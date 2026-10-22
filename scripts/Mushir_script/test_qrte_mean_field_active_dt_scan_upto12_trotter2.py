"""Verify matched Trotter grid, dispatch, worker submission and overlay validation."""

import copy
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

import run_qrte_mean_field_active_dt_scan_upto12_trotter2 as scan
import run_qrte_mean_field_active_dt_scan_upto12_exact_worker as worker
import collect_qrte_mean_field_active_dt_scan4 as collector
import plot_qrte_mean_field_dt_scan_upto12_exact_loglog as style
import plot_qrte_mean_field_dt_scan_upto12_exact_vs_trotter2 as overlay

SCRIPTS = Path(__file__).resolve().parent
STEM = "qrte_mean_field_active_dt_scan_upto12_trotter2"


class MatchedTrotterTests(unittest.TestCase):
    def test_all_88_conditions_match_exact_and_pin_order_two_one_slice(self):
        args = scan.parse_args([])
        tasks = scan.load_tasks(args)
        exact_args = scan.exact.parse_args([])
        exact = scan.exact.load_tasks(exact_args)
        self.assertEqual(len(tasks), 88)
        self.assertEqual((args.evolution_mode, args.trotter_order, args.trotter_steps), ("trotter", 2, 1))
        for a, b in zip(collector.plan_rows(tasks), collector.plan_rows(exact)):
            for field in scan.MATCH_FIELDS:
                self.assertEqual(a[field], b[field])
            self.assertNotEqual(a["output_directory"], b["output_directory"])
        for options in (["--evolution-mode", "sparse"], ["--trotter-order", "1"],
                        ["--trotter-steps", "2"], ["--delta-t", "0.05"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                scan.parse_args(options)
        with self.assertRaisesRegex(ValueError, "n_steps must match"):
            scan.load_tasks(scan.parse_args(["--n-steps", "200"]))
        changed = scan.qrte.load_json(scan.EXACT_PLAN)
        changed["tasks"][0]["hamiltonian_sha256"] = "changed"
        # Mock only the selected exact-plan read, retaining manifest/reference reads.
        original = scan.qrte.load_json
        def load(path):
            return changed if path == scan.EXACT_PLAN else original(path)
        with patch.object(scan.qrte, "load_json", side_effect=load), self.assertRaisesRegex(ValueError, "hamiltonian_sha256"):
            scan.load_tasks(args)

    def test_trotter_dispatch_pins_cache_and_resumes(self):
        args = scan.parse_args(["--case-index", "51"])
        task = scan.load_tasks(args)[0]
        reference = task["case"]["local_reference_energy_hartree"]
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())
        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual((kwargs["evolution_mode"], kwargs["trotter_order"], kwargs["trotter_steps"]), ("trotter", 2, 1))
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
                    patch.object(scan.qrte, "_payload_from_cache_path", return_value=payload), \
                    patch.object(scan.qrte, "compute_fock_spectrum") as scf:
                scan.run_task(task, args)
                scan.run_task(task, args)
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                scf.assert_not_called()
            self.assertEqual(collector.result_rows([task], args)[0]["status"], "completed_hit")

    def test_plan_preparation_and_workers_never_use_exact_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = scan.parse_args(["--outdir", tmp])
            tasks = scan.load_tasks(args)
            collector.write_plan(tasks, args)
            scan.record_comparison(tasks, args)
            self.assertEqual({r["status"] for r in collector.result_rows(tasks, args)}, {"pending"})
            audit = scan.qrte.load_json(Path(tmp) / "exact_comparison.json")
            self.assertEqual(audit["trotter_order"], 2)
            exact = scan.qrte.load_json(scan.exact.DEFAULT_OUTDIR / "factor_0p25/N2_equi_cc_pVTZ_0/summary.json")
            with self.assertRaisesRegex(ValueError, "propagation"):
                collector.validate_summary(exact, tasks[4], args)
            with self.assertRaisesRegex(ValueError, "88 completed"):
                style.validated_rows(Path(tmp), scan_module=scan)
        with patch.object(scan, "run_task") as run:
            worker.main(["--worker-index", "3", "--worker-count", "4"], scan_module=scan)
        self.assertEqual([c.args[0]["task_index"] for c in run.call_args_list], list(range(3, 88, 4)))

    def test_overlay_matches_pairs_and_keeps_no_hit_as_bound(self):
        rows, args, _, _ = style.validated_rows(scan.exact.DEFAULT_OUTDIR)
        overlay.validate_pairs(rows, copy.deepcopy(rows))
        bad = copy.deepcopy(rows)
        bad[0]["hamiltonian_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hamiltonian_sha256"):
            overlay.validate_pairs(rows, bad)
        with self.assertRaisesRegex(ValueError, "grids differ"):
            overlay.validate_pairs(rows, rows[:-1])
        points = [r for r in rows if r["case_id"] == "N2_equi_cc_pVTZ_2"]
        approximate = copy.deepcopy(points)
        approximate[-1]["hit_iteration"] = 12
        approximate[-1]["plotted_y"] = 12
        fig, ax = style.plt.subplots()
        overlay.draw_overlay(ax, points, approximate)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()[0]))
        self.assertTrue(np.isnan(ax.lines[-1].get_ydata()[0]))
        self.assertEqual(ax.lines[-1].get_ydata()[-1], 12)
        self.assertGreater(ax.get_ylim()[1], 100)
        style.plt.close(fig)

    def test_slurm_uses_four_workers_and_pins_trotter_two(self):
        batch = SCRIPTS / f"run_{STEM}_array.sbatch"
        submit = SCRIPTS / f"submit_talon_{STEM}.sh"
        for path in (batch, submit):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --array=0-3%4", batch.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("QCANT_")}
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="3", QCANT_EVOLUTION_MODE="sparse",
                       QCANT_TROTTER_ORDER="1", MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            command = subprocess.run(["bash", str(batch)], env=env, check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(command[0], f"scripts/Mushir_script/run_{STEM}_worker.py")
            self.assertEqual(command[command.index("--evolution-mode") + 1], "trotter")
            self.assertEqual(command[command.index("--trotter-order") + 1], "2")
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
            env.update(PYTHON="python_stub", TEST_COUNT="88")
            output = subprocess.run(["bash", "-c", shell, "test", str(local)], env=env, check=True, capture_output=True, text=True)
            self.assertIn("--array=0-3%4", output.stderr)
            self.assertIn("--prepare-only", output.stderr)
            self.assertEqual((Path(tmp) / f"provenance/latest_{STEM}_jobid.txt").read_text(), "12345\n")
            env["TEST_COUNT"] = "0"
            output = subprocess.run(["bash", "-c", shell, "test", str(local)], env=env, capture_output=True, text=True)
            self.assertNotEqual(output.returncode, 0)
            self.assertNotIn("--parsable", output.stderr)


if __name__ == "__main__":
    unittest.main()
