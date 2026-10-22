"""Saved-input, task-mapping, reference, collector, and mocked Slurm checks."""

import copy
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_qrte_mean_field_active_dt_scan4 as driver
import collect_qrte_mean_field_active_dt_scan4 as collector

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH = SCRIPT_DIR / "run_qrte_mean_field_active_dt_scan4_array.sbatch"
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_mean_field_active_dt_scan4.sh"


class ScanTests(unittest.TestCase):
    def test_grid_and_centre_points_match_saved_mean_field_results(self):
        args = driver.parse_args([])
        tasks = driver.load_tasks(args)
        self.assertEqual(len(tasks), 28)
        self.assertEqual([task["task_index"] for task in tasks], list(range(28)))
        self.assertEqual((args.trotter_order, args.trotter_steps, args.n_steps, args.overlap_tol), (2, 1, 100, 1e-10))
        roots = [driver.mf.DEFAULT_OUTDIR, driver.qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2"]
        for offset, cid in enumerate(driver.CASE_IDS):
            block = tasks[offset * 7:(offset + 1) * 7]
            self.assertEqual([task["factor"] for task in block], list(driver.FACTORS))
            path = next(root / cid / "summary.json" for root in roots if (root / cid / "summary.json").exists())
            baseline = driver.qrte.load_json(path)
            for task in block:
                self.assertEqual(task["case"]["case_id"], cid)
                self.assertEqual(task["base_delta_t"], baseline["delta_t"])
                self.assertEqual(task["case"]["hamiltonian_sha256"], baseline["hamiltonian_sha256"])
                self.assertEqual(task["case"]["local_reference_energy_hartree"], baseline["local_reference_energy_hartree"])
                spectrum = driver.compute_scan_spectrum(task["case"], None, args)
                width = driver.scan_width(spectrum)
                self.assertEqual(width["delta_t"], task["factor"] * baseline["delta_t"])
                self.assertEqual(width["mean_field_active_bandwidth_hartree"], baseline["mean_field_active_bandwidth_hartree"])
                self.assertEqual(width["spectrum_scope"], driver.SPECTRUM_SCOPE)
        self.assertEqual(len({str(task["outdir"] / task["case"]["case_id"]) for task in tasks}), 28)
        selected = driver.load_tasks(driver.parse_args(["--case-index", "27"]))
        self.assertEqual((selected[0]["case"]["case_id"], selected[0]["factor"]), (driver.CASE_IDS[3], 2))

    def test_invalid_indices_and_incompatible_propagation_are_rejected(self):
        for options in (["--trotter-order", "1"], ["--trotter-steps", "4"],
                        ["--evolution-mode", "sparse"], ["--delta-t", "0.05"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                driver.parse_args(options)
        for options in (["--case-index", "28"], ["--case-id", "FeO"]):
            with self.assertRaises(ValueError):
                driver.load_tasks(driver.parse_args(options))

    def test_scaled_run_pins_original_payload_reference_and_resumes(self):
        args = driver.parse_args(["--case-index", "14"])
        task = driver.load_tasks(args)[0]
        reference = task["case"]["local_reference_energy_hartree"]
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["delta_t"], task["delta_t"])
            self.assertEqual((kwargs["trotter_order"], kwargs["trotter_steps"]), (2, 1))
            self.assertEqual(kwargs["early_stop_energy"], reference + 0.0016)
            self.assertNotEqual(kwargs["early_stop_energy"], task["case"]["dataset_casscf_energy_hartree"] + 0.0016)
            history = np.array([reference + 0.0005])
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
            summary = driver.qrte.load_json(Path(tmp) / task["case"]["case_id"] / "summary.json")
            self.assertEqual(summary["dt_scale_factor"], 0.25)
            self.assertEqual(summary["delta_t"], task["delta_t"])
            self.assertEqual(summary["run_profile"]["fock_profile"]["case"]["dt_scale_factor"], 0.25)
            self.assertEqual(summary["hit_iteration"], 1)

    def test_collector_distinguishes_pending_hit_no_hit_failed_and_invalid(self):
        args = driver.parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            tasks = driver.load_tasks(args)
            collector.write_plan(tasks, args)
            self.assertEqual({row["status"] for row in collector.result_rows(tasks, args)}, {"pending"})
            for task, status in zip(tasks[:4], ("completed", "completed", "failed", "completed")):
                case = task["case"]
                reference = case["local_reference_energy_hartree"]
                hit = task["task_index"] == 0
                count = 1 if hit else args.n_steps
                summary = {**case, "status": status, "evolution_mode": "trotter", "trotter_order": 2,
                           "trotter_steps": 1, "n_steps_requested": 100, "overlap_tol": args.overlap_tol,
                           "delta_t": task["delta_t"], "n_steps_completed": count,
                           "hit_iteration": 1 if hit else None, "hit_chemical_accuracy": hit}
                if task["task_index"] == 3:
                    summary["trotter_order"] = 1
                path = task["outdir"] / case["case_id"]
                driver.qrte.write_json(path / "summary.json", summary)
                with (path / "min_energy_history.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["iteration", "min_energy_hartree"])
                    writer.writerows((idx, reference + (0.0005 if hit else 0.01)) for idx in range(1, count + 1))
            rows = collector.result_rows(tasks, args)
            self.assertEqual([row["status"] for row in rows[:5]],
                             ["completed_hit", "completed_no_hit", "failed", "invalid_result", "pending"])
            self.assertEqual(rows[1]["hit_iteration"], "")
            self.assertEqual(rows[1]["n_steps_completed"], 100)
            collector.plot_results(rows, args)
            self.assertTrue((Path(tmp) / "iterations_vs_dt_scale.png").exists())
            changed = copy.copy(args)
            changed.n_steps = 200
            with self.assertRaisesRegex(ValueError, "Existing scan plan differs"):
                collector.write_plan(tasks, changed)

    def test_relative_and_absolute_output_paths_produce_same_plan(self):
        relative = driver.parse_args(["--outdir", "runs/qrte_mean_field_active_dt_scan4_trotter2"])
        absolute = driver.parse_args([])
        self.assertEqual(relative.outdir, absolute.outdir)


class SubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_batch_pins_order_and_slice_count_from_slurm_spool(self):
        for path in (SUBMIT, BATCH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --time=06:00:00", BATCH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(BATCH, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="27", QCANT_TROTTER_ORDER="1",
                       QCANT_TROTTER_STEPS="4", QCANT_DELTA_T="0.05", QCANT_EVOLUTION_MODE="sparse",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            output = subprocess.run(["bash", str(spool)], env=env, check=True,
                                    capture_output=True, text=True).stdout.split()
            self.assertEqual(output[0], f"scripts/Mushir_script/{driver.__name__}.py")
            for key, value in {"--case-index": "27", "--trotter-order": "2", "--trotter-steps": "1",
                               "--evolution-mode": "trotter", "--n-steps": "100"}.items():
                self.assertEqual(output[output.index(key) + 1], value)
            self.assertNotIn("--delta-t", output)

    def test_submission_preflight_and_options_with_stubbed_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            command = '''
python_stub() { printf '%s\\n' "$@" >&2; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1" --time=02:00:00
'''
            env = self.clean_env()
            env.update(PYTHON="python_stub", TEST_COUNT="28")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    check=True, capture_output=True, text=True)
            self.assertIn("--array=0-27%4", result.stderr)
            self.assertIn("--time=02:00:00", result.stderr)
            self.assertIn("--prepare-only", result.stderr)
            self.assertIn(BATCH.name, result.stderr)
            record = Path(tmp) / "provenance/latest_qrte_mean_field_active_dt_scan4_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "27"
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
