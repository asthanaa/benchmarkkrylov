"""Saved-input checks for the seven-case scan; QRTE and Slurm are mocked."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_qrte_mean_field_active_dt_scan_beh2_hchains as driver
import collect_qrte_mean_field_active_dt_scan_beh2_hchains as collect

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_mean_field_active_dt_scan_beh2_hchains.sh"
BATCH = SCRIPT_DIR / "run_qrte_mean_field_active_dt_scan_beh2_hchains_array.sbatch"


class ExtendedScanTests(unittest.TestCase):
    def test_all_49_conditions_match_saved_centres_and_provenance(self):
        args = driver.parse_args([])
        tasks = driver.load_tasks(args)
        self.assertEqual(len(tasks), 49)
        self.assertEqual(len(driver.CASE_IDS), 7)
        self.assertEqual([task["task_index"] for task in tasks], list(range(49)))
        self.assertEqual((args.trotter_order, args.trotter_steps, args.n_steps, args.overlap_tol), (2, 1, 100, 1e-10))
        self.assertNotEqual(args.outdir, driver.base.DEFAULT_OUTDIR)
        roots = [driver.mf.DEFAULT_OUTDIR, driver.qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2"]
        for offset, cid in enumerate(driver.CASE_IDS):
            block = tasks[7 * offset:7 * (offset + 1)]
            self.assertEqual([task["factor"] for task in block], list(driver.FACTORS))
            path = next(root / cid / "summary.json" for root in roots if (root / cid / "summary.json").exists())
            baseline = driver.qrte.load_json(path)
            for task in block:
                self.assertEqual(task["case"]["case_id"], cid)
                self.assertEqual(task["base_delta_t"], baseline["delta_t"])
                self.assertEqual(task["case"]["hamiltonian_sha256"], baseline["hamiltonian_sha256"])
                self.assertEqual(task["case"]["local_reference_energy_hartree"], baseline["local_reference_energy_hartree"])
                result = driver.base.compute_scan_spectrum(task["case"], None, args)
                self.assertEqual(driver.base.scan_width(result)["delta_t"], task["factor"] * baseline["delta_t"])
        self.assertEqual(len({str(t["outdir"] / t["case"]["case_id"]) for t in tasks}), 49)

    def test_main_count_and_final_array_index_use_seven_cases(self):
        with patch.object(sys, "argv", ["scan", "--count-cases"]), redirect_stdout(io.StringIO()) as out:
            driver.main()
        self.assertEqual(out.getvalue().strip(), "49")
        last = driver.load_tasks(driver.parse_args(["--case-index", "48"]))[0]
        self.assertEqual((last["case"]["case_id"], last["factor"]), ("H7_near_equi_sto3g_1", 2))
        with self.assertRaises(ValueError):
            driver.load_tasks(driver.parse_args(["--case-index", "49"]))
        with self.assertRaises(ValueError):
            driver.load_tasks(driver.parse_args(["--case-id", "N2_equi_cc_pVTZ_0"]))
        for options in (["--trotter-order", "1"], ["--trotter-steps", "4"], ["--delta-t", "0.05"]):
            with self.assertRaises(SystemExit):
                driver.parse_args(options)

    def test_collector_prepares_seven_cases_and_renders_seven_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys, "argv", ["collect", "--outdir", tmp, "--prepare-only"]):
                collect.main()
            plan = driver.qrte.load_json(Path(tmp) / "scan_plan.json")
            self.assertEqual(plan["case_ids"], list(driver.CASE_IDS))
            self.assertEqual(plan["task_count"], 49)
            args = driver.parse_args(["--outdir", tmp])
            tasks = driver.load_tasks(args)
            rows = collect.collector.result_rows(tasks, args)
            self.assertEqual({row["status"] for row in rows}, {"pending"})
            import matplotlib.pyplot as plt
            with patch.object(plt, "close") as close:
                collect.collector.plot_results(rows, args, case_ids=driver.CASE_IDS, labels=collect.LABELS)
                figure = close.call_args.args[0]
                self.assertEqual(sum(ax.get_visible() for ax in figure.axes), 7)
            plt.close(figure)
            self.assertTrue((Path(tmp) / "iterations_vs_dt_scale.png").exists())


class SubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_batch_spool_pins_second_order_one_slice_and_six_hours(self):
        for path in (SUBMIT, BATCH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --time=06:00:00", BATCH.read_text())
        self.assertIn("#SBATCH --array=0-48%4", BATCH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(BATCH, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="48", QCANT_TROTTER_ORDER="1",
                       QCANT_TROTTER_STEPS="4", QCANT_DELTA_T="0.05", QCANT_EVOLUTION_MODE="sparse",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            options = subprocess.run(["bash", str(spool)], env=env, check=True,
                                     capture_output=True, text=True).stdout.split()
            self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")
            for key, value in {"--case-index": "48", "--trotter-order": "2", "--trotter-steps": "1",
                               "--evolution-mode": "trotter"}.items():
                self.assertEqual(options[options.index(key) + 1], value)
            self.assertNotIn("--delta-t", options)

    def test_submission_checks_49_and_records_separate_job_with_mocked_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            command = '''
python_stub() { printf '%s\\n' "$@" >&2; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1"
'''
            env = self.clean_env()
            env.update(PYTHON="python_stub", TEST_COUNT="49")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    check=True, capture_output=True, text=True)
            self.assertIn("--array=0-48%4", result.stderr)
            self.assertIn("--prepare-only", result.stderr)
            self.assertIn(BATCH.name, result.stderr)
            record = Path(tmp) / "provenance/latest_qrte_mean_field_active_dt_scan_beh2_hchains_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "28"
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
