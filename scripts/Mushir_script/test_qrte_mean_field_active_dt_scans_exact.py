"""Verify exact scan inputs, dispatch/resume, collectors and Slurm without jobs."""

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

import run_qrte_mean_field_active_dt_scan4_exact as four
import run_qrte_mean_field_active_dt_scan_beh2_hchains_exact as seven
import collect_qrte_mean_field_active_dt_scan4 as collector

SCRIPTS = Path(__file__).resolve().parent
DRIVERS = ((four, 28, "scan4"), (seven, 49, "scan_beh2_hchains"))


class ExactScanTests(unittest.TestCase):
    def test_all_77_tasks_match_previous_trotter_scans(self):
        for driver, count, group in DRIVERS:
            with self.subTest(group=group):
                args = driver.parse_args([])
                tasks = driver.load_tasks(args)
                previous = driver.qrte.load_json(driver.qrte.ROOT / (
                    f"runs/qrte_mean_field_active_dt_{group}_trotter2/scan_plan.json"))
                self.assertEqual(len(tasks), count)
                self.assertEqual(args.evolution_mode, "sparse")
                self.assertEqual((args.n_steps, args.overlap_tol), (100, 1e-10))
                self.assertEqual([t["task_index"] for t in tasks], list(range(count)))
                for new, old in zip(collector.plan_rows(tasks), previous["tasks"]):
                    for key in ("task_index", "case_id", "dt_scale_factor", "delta_t",
                                "mean_field_active_bandwidth_hartree", "unscaled_mean_field_delta_t",
                                "cached_reference_energy_hartree", "hamiltonian_sha256",
                                "mean_field_reference_sha256"):
                        self.assertEqual(new[key], old[key], key)
                    self.assertNotEqual(new["output_directory"], old["output_directory"])
                    self.assertFalse(new["trotter_parameters_apply"])
                selected = driver.load_tasks(driver.parse_args(["--case-index", str(count - 1)]))
                self.assertEqual(selected[0]["task_index"], count - 1)
                with self.assertRaises(ValueError):
                    driver.load_tasks(driver.parse_args(["--case-index", str(count)]))

    def test_exact_arguments_and_plan_reject_trotter_results(self):
        for driver, count, group in DRIVERS:
            for options in (["--evolution-mode", "trotter"], ["--delta-t", "0.05"],
                            ["--trotter-order", "2"], ["--trotter-steps", "4"]):
                with self.subTest(group=group, options=options), self.assertRaises(SystemExit):
                    driver.parse_args(options)
            with tempfile.TemporaryDirectory() as tmp:
                args = driver.parse_args(["--outdir", tmp])
                tasks = driver.load_tasks(args)
                collector.write_plan(tasks, args)
                plan = driver.qrte.load_json(Path(tmp) / "scan_plan.json")
                self.assertEqual(plan["evolution_mode"], "sparse")
                self.assertFalse(plan["trotter_parameters_apply"])
                self.assertEqual(len(collector.result_rows(tasks, args)), count)
                self.assertEqual({r["status"] for r in collector.result_rows(tasks, args)}, {"pending"})
                old_root = driver.qrte.ROOT / f"runs/qrte_mean_field_active_dt_{group}_trotter2"
                task = tasks[0]
                old_summary = driver.qrte.load_json(old_root / "factor_0p25" / driver.CASE_IDS[0] / "summary.json")
                with self.assertRaisesRegex(ValueError, "propagation"):
                    collector.validate_summary(old_summary, task, args)
                changed = copy.copy(args)
                changed.n_steps = 200
                with self.assertRaisesRegex(ValueError, "Existing scan plan differs"):
                    collector.write_plan(tasks, changed)

    def test_exact_dispatch_uses_pinned_cache_scaled_dt_and_cached_target_and_resumes(self):
        args = four.parse_args(["--case-index", "14"])
        task = four.load_tasks(args)[0]  # Al3-minus has a distinct dataset reference.
        reference = task["case"]["local_reference_energy_hartree"]
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["evolution_mode"], "sparse")
            self.assertEqual(kwargs["delta_t"], task["delta_t"])
            self.assertEqual(kwargs["early_stop_energy"], reference + .0016)
            self.assertNotEqual(kwargs["early_stop_energy"], task["case"]["dataset_casscf_energy_hartree"] + .0016)
            history = np.array([reference + .0005])
            return history, None, None, history

        qa.qrte_qulacs = Mock(side_effect=simulate)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            task["outdir"] = Path(tmp)
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(four.qrte, "_payload_from_cache_path", return_value=payload), \
                    patch.object(four.qrte, "compute_fock_spectrum") as scf:
                four.run_task(task, args)
                four.run_task(task, args)
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                scf.assert_not_called()
            summary = four.qrte.load_json(Path(tmp) / task["case"]["case_id"] / "summary.json")
            collector.validate_summary(summary, task, args)
            self.assertEqual(summary["evolution_mode"], "sparse")
            self.assertEqual(summary["dt_scale_factor"], .25)
            self.assertEqual(collector.result_rows([task], args)[0]["status"], "completed_hit")

    def test_slurm_spool_pins_exact_mode_and_submission_preflight(self):
        for driver, count, group in DRIVERS:
            stem = f"qrte_mean_field_active_dt_{group}_exact"
            batch = SCRIPTS / f"run_{stem}_array.sbatch"
            submit = SCRIPTS / f"submit_talon_{stem}.sh"
            for path in (batch, submit):
                subprocess.run(["bash", "-n", str(path)], check=True)
            self.assertIn("#SBATCH --time=2-00:00:00", batch.read_text())
            memory = "64G" if driver is four else "8G"
            self.assertIn(f"#SBATCH --mem={memory}", batch.read_text())
            with tempfile.TemporaryDirectory() as tmp:
                spool = Path(tmp) / "slurm_script"
                shutil.copyfile(batch, spool)
                env = {k: v for k, v in os.environ.items() if not k.startswith("QCANT_")}
                env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID=str(count - 1),
                           QCANT_EVOLUTION_MODE="trotter", QCANT_TROTTER_ORDER="2",
                           QCANT_DELTA_T="0.05", MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
                options = subprocess.run(["bash", str(spool)], env=env, check=True,
                                         capture_output=True, text=True).stdout.split()
                self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")
                self.assertEqual(options[options.index("--evolution-mode") + 1], "sparse")
                self.assertNotIn("--delta-t", options)
                scripts = Path(tmp) / "scripts/Mushir_script"
                scripts.mkdir(parents=True)
                local_submit = scripts / submit.name
                shutil.copyfile(submit, local_submit)
                command = '''
python_stub() { printf '%s\\n' "$@" >&2; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1" --time=02:00:00
'''
                env.update(PYTHON="python_stub", TEST_COUNT=str(count))
                result = subprocess.run(["bash", "-c", command, "test", str(local_submit)],
                                        env=env, check=True, capture_output=True, text=True)
                self.assertIn(f"--array=0-{count - 1}%4", result.stderr)
                self.assertIn("--prepare-only", result.stderr)
                self.assertIn("--time=02:00:00", result.stderr)
                record = Path(tmp) / f"provenance/latest_{stem}_jobid.txt"
                self.assertEqual(record.read_text(), "12345\n")
                env["TEST_COUNT"] = "0"
                result = subprocess.run(["bash", "-c", command, "test", str(local_submit)],
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
