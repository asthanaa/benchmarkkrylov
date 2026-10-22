"""Validate matched hexatriene grids, cached dispatch, workers and launchers."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_qrte_mean_field_dt_scan_hexatriene_exact as exact
import run_qrte_mean_field_dt_scan_hexatriene_trotter2 as trotter
import run_qrte_mean_field_active_dt_scan_upto12_exact_worker as worker
import collect_qrte_mean_field_active_dt_scan4 as collector
import plot_qrte_mean_field_dt_scan_upto12_exact_loglog as style
import plot_qrte_mean_field_dt_scan_upto12_exact_vs_trotter2 as overlay
from collect_qrte_mean_field_dt_scan_hexatriene_exact import LABELS

SCRIPTS = Path(__file__).resolve().parent


class HexatrieneScanTests(unittest.TestCase):
    def test_all_44_conditions_match_saved_hamiltonians_and_hf_widths(self):
        a, b = exact.load_tasks(exact.parse_args([])), trotter.load_tasks(trotter.parse_args([]))
        self.assertEqual(len(a), 22)
        self.assertEqual(len(b), 22)
        self.assertEqual([t["task_index"] for t in a], list(range(22)))
        for left, right in zip(a, b):
            self.assertEqual(left["case"], right["case"])
            self.assertEqual(left["delta_t"], right["delta_t"])
            self.assertEqual(left["case"]["active_orbitals"], 6)
            baseline = exact.qrte.load_json(exact.qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2"
                                             / left["case"]["case_id"] / "summary.json")
            self.assertEqual(left["base_delta_t"], baseline["delta_t"])
            self.assertEqual(left["case"]["hamiltonian_sha256"], baseline["hamiltonian_sha256"])
            self.assertEqual(left["delta_t"], left["factor"] * baseline["delta_t"])
        for start in (0, 11):
            self.assertEqual([t["factor"] for t in a[start:start + 11]], list(exact.FACTORS))
        self.assertEqual({t["evolution_mode"] for t in a}, {"sparse"})
        self.assertEqual({(t["evolution_mode"], t["trotter_order"], t["trotter_steps"]) for t in b}, {("trotter", 2, 1)})
        self.assertTrue({t["outdir"] for t in a}.isdisjoint({t["outdir"] for t in b}))

    def test_invalid_settings_and_unfinished_plot_data_are_rejected(self):
        for driver, wrong_mode in ((exact, "trotter"), (trotter, "sparse")):
            for options in (["--evolution-mode", wrong_mode], ["--trotter-steps", "2"], ["--delta-t", ".05"]):
                with self.assertRaises(SystemExit):
                    driver.parse_args(options)
            for options in (["--case-index", "22"], ["--case-id", "H4_near_equi_sto3g_0"]):
                with self.assertRaises(ValueError):
                    driver.load_tasks(driver.parse_args(options))
            with tempfile.TemporaryDirectory() as tmp:
                args = driver.parse_args(["--outdir", tmp])
                tasks = driver.load_tasks(args)
                collector.write_plan(tasks, args)
                plan = driver.qrte.load_json(Path(tmp) / "scan_plan.json")
                self.assertEqual(plan["task_count"], 22)
                self.assertEqual(plan["factors"], list(exact.FACTORS))
                self.assertEqual({r["status"] for r in collector.result_rows(tasks, args)}, {"pending"})
                with self.assertRaisesRegex(ValueError, "22 completed"):
                    style.validated_rows(Path(tmp), scan_module=driver)

    def test_both_methods_dispatch_the_pinned_cache_and_resume(self):
        for driver in (exact, trotter):
            args = driver.parse_args(["--case-index", "0"])
            task = driver.load_tasks(args)[0]
            reference = task["case"]["local_reference_energy_hartree"]
            payload = {"pinned": True}
            qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())
            def simulate(**kwargs):
                self.assertIs(qa._build_hamiltonian_payload(), payload)
                self.assertEqual(kwargs["evolution_mode"], args.evolution_mode)
                self.assertEqual(kwargs["trotter_order"], args.trotter_order)
                self.assertEqual(kwargs["trotter_steps"], 1)
                self.assertEqual(kwargs["delta_t"], task["delta_t"])
                self.assertEqual(kwargs["early_stop_energy"], reference + .0016)
                history = np.array([reference + .0005])
                return history, None, None, history
            qa.qrte_qulacs = Mock(side_effect=simulate)
            fake = types.ModuleType("QCANT")
            fake.qulacs_accel = qa
            with tempfile.TemporaryDirectory() as tmp:
                task["outdir"] = Path(tmp)
                with patch.dict(sys.modules, {"QCANT": fake}), \
                        patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload), \
                        patch.object(driver.qrte, "compute_fock_spectrum") as scf:
                    driver.run_task(task, args)
                    driver.run_task(task, args)
                    self.assertEqual(qa.qrte_qulacs.call_count, 1)
                    scf.assert_not_called()
                self.assertEqual(collector.result_rows([task], args)[0]["status"], "completed_hit")

    def test_workers_cover_22_conditions_and_launchers_pin_propagation(self):
        for driver, mode in ((exact, "exact"), (trotter, "trotter2")):
            tasks = driver.load_tasks(driver.parse_args([]))
            parts = [worker.select_tasks(tasks, i, 4) for i in range(4)]
            self.assertEqual([len(part) for part in parts], [6, 6, 5, 5])
            self.assertEqual(sorted(t["task_index"] for part in parts for t in part), list(range(22)))
            stem = f"qrte_mean_field_dt_scan_hexatriene_{mode}"
            batch = SCRIPTS / f"run_{stem}_array.sbatch"
            submit = SCRIPTS / f"submit_talon_{stem}.sh"
            for path in (batch, submit):
                subprocess.run(["bash", "-n", str(path)], check=True)
            with tempfile.TemporaryDirectory() as tmp:
                env = {k: v for k, v in os.environ.items() if not k.startswith("QCANT_")}
                env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="3", MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp,
                           QCANT_EVOLUTION_MODE="sparse" if driver is trotter else "trotter", QCANT_TROTTER_STEPS="4")
                result = subprocess.run(["bash", str(batch)], env=env, check=True, capture_output=True, text=True)
                command = result.stdout.split()
                self.assertEqual(command[0], f"scripts/Mushir_script/run_{stem}_worker.py")
                self.assertEqual(command[command.index("--evolution-mode") + 1], "sparse" if driver is exact else "trotter")
                self.assertEqual(command[command.index("--trotter-order") + 1], "1" if driver is exact else "2")
                self.assertEqual(command[command.index("--trotter-steps") + 1], "1")
                scripts = Path(tmp) / "scripts/Mushir_script"
                scripts.mkdir(parents=True)
                local = scripts / submit.name
                shutil.copyfile(submit, local)
                shell = '''
python_stub() { printf '%s\\n' "$@" >&2; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1"
'''
                env.update(PYTHON="python_stub", TEST_COUNT="22")
                result = subprocess.run(["bash", "-c", shell, "test", str(local)], env=env,
                                        check=True, capture_output=True, text=True)
                self.assertIn("--array=0-3%4", result.stderr)
                self.assertIn("--prepare-only", result.stderr)
                self.assertEqual((Path(tmp) / f"provenance/latest_{stem}_jobid.txt").read_text(), "12345\n")
                env["TEST_COUNT"] = "88"
                result = subprocess.run(["bash", "-c", shell, "test", str(local)], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("--parsable", result.stderr)

    def test_hexatriene_overlay_labels_and_bounds(self):
        cid = exact.CASE_IDS[0]
        points = [{"case_id": cid, "dt_scale_factor": f, "unscaled_mean_field_delta_t": 1.,
                   "status": "completed_hit", "hit_iteration": 4, "plotted_y": 4, "n_steps_completed": 4}
                  for f in exact.FACTORS]
        fig, ax = style.plt.subplots()
        overlay.draw_overlay(ax, points, points, labels=LABELS)
        self.assertIn("Hexatriene", ax.get_title())
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
        self.assertEqual(len(ax.lines[0].get_xdata()), 11)
        style.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
