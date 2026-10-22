"""Remaining-case selection, saved-input readiness, and mocked Slurm checks."""

import copy
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_qrte_mean_field_active_pi_over_w_remaining17 as driver

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_mean_field_active_pi_over_w_remaining17_trotter2.sh"
ARRAY = SCRIPT_DIR / "run_qrte_mean_field_active_pi_over_w_remaining17_trotter2_array.sbatch"


class Remaining17Tests(unittest.TestCase):
    def args(self, *options):
        with patch.object(sys, "argv", ["driver", *options]):
            return driver.parse_args()

    def test_partition_and_order_are_fixed_when_cache_rows_reordered(self):
        args = self.args()
        all_cases = driver.qrte.load_cases(args)
        self.assertEqual(len(driver.CASE_IDS), 17)
        self.assertFalse(set(driver.CASE_IDS) & set(driver.mean_field.CASE_IDS))
        self.assertEqual(set(driver.CASE_IDS) | set(driver.mean_field.CASE_IDS),
                         {case["case_id"] for case in all_cases})
        with patch.object(driver.qrte, "load_cases", return_value=list(reversed(all_cases))):
            cases = driver.load_cases(args, validate_saved=False)
        self.assertEqual(tuple(case["case_id"] for case in cases), driver.CASE_IDS)
        for index, case_id in enumerate(driver.CASE_IDS):
            cases = driver.load_cases(self.args("--case-index", str(index)), validate_saved=False)
            self.assertEqual([case["case_id"] for case in cases], [case_id])
        with self.assertRaisesRegex(ValueError, "outside"):
            driver.load_cases(self.args("--case-id", driver.mean_field.CASE_IDS[0]))
        with self.assertRaisesRegex(ValueError, "--case-index"):
            driver.load_cases(self.args("--case-index", "17"))

    def test_available_widths_use_active_sums_and_preserve_special_references(self):
        args = self.args()
        cases = driver.load_cases(args, validate_saved=False)
        checked = []
        for case in cases:
            if not Path(case["mean_field_reference_path"]).exists():
                continue
            result = driver.inspect_case(case, args)
            self.assertEqual(result["status"], "ready", result)
            energies = sorted(result["mean_field_active_estimate"]["alpha_orbital_energies_hartree"])
            low = sum(sum(energies[:n]) for n in case["nelecas"])
            high = sum(sum(energies[-n:]) if n else 0 for n in case["nelecas"])
            self.assertAlmostEqual(result["mean_field_active_bandwidth_hartree"], high - low, places=12)
            self.assertAlmostEqual(result["delta_t"], math.pi / (high - low), places=12)
            checked.append(case["case_id"])
            if case["active_orbital_indices"]:
                self.assertEqual(result["active_orbital_indices_zero_based"], case["active_orbital_indices"])
            if case["case_id"] in {"feno_2plus", "fenoco3_minus"}:
                self.assertEqual(case["reference_settings"]["reference_scf"], "sfX2C1eROHF")
                self.assertIn("open_shell_caveat", result["mean_field_active_estimate"])
            if case["case_id"] == "fenoco3_minus":
                self.assertFalse(case["local_reference_converged"])
            if case["case_id"] == "FeO":
                self.assertEqual(case["reference_settings"]["geometry_unit"], "Bohr")
        self.assertGreaterEqual(len(checked), 15)
        selected = self.args(*[arg for cid in checked for arg in ("--case-id", cid)])
        self.assertEqual(len(driver.load_cases(selected)), len(checked))

    def test_missing_inputs_report_all_cases_and_prevent_any_qrte_work(self):
        args = self.args()
        cases = driver.load_cases(args, validate_saved=False)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                case["mean_field_reference_path"] = str(Path(tmp) / case["case_id"] / "fock_reference.json")
            with patch.object(driver.mean_field, "FOCK_ROOT", Path(tmp)), \
                    patch.object(driver.qrte, "run_case") as run, \
                    patch.object(driver.qrte, "compute_fock_spectrum") as scf:
                with self.assertRaises(ValueError) as error:
                    driver.load_cases(args)
                for cid in driver.CASE_IDS:
                    self.assertIn(cid, str(error.exception))
                with patch.object(sys, "argv", ["driver", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(SystemExit) as exit_status:
                        driver.main()
                self.assertEqual(exit_status.exception.code, 1)
                self.assertIn("ready=0/17", out.getvalue())
                self.assertEqual(out.getvalue().count("missing_saved_reference:"), 17)
                run.assert_not_called()
                scf.assert_not_called()

    def test_changed_source_and_changed_dataset_are_rejected(self):
        args = self.args("--case-id", "FeO")
        case = driver.load_cases(args)[0]
        changed = copy.copy(case)
        changed["mean_field_reference_sha256"] = "wrong"
        self.assertEqual(driver.inspect_case(changed, args)["status"], "invalid_saved_reference")
        all_cases = driver.qrte.load_cases(self.args())
        with patch.object(driver.qrte, "load_cases", return_value=all_cases[:-1]):
            with self.assertRaisesRegex(ValueError, "original 27"):
                driver.load_cases(args)

    def test_main_passes_same_cached_reference_builders_and_settings(self):
        with patch.object(sys, "argv", ["driver", "--case-id", "FeO"]), \
                patch.object(driver.qrte, "run_case") as run:
            driver.main()
        case, args = run.call_args.args
        self.assertEqual(case["case_id"], "FeO")
        self.assertEqual(case["hamiltonian_sha256"], driver.qrte.load_cases(args)[0]["hamiltonian_sha256"])
        self.assertEqual((args.trotter_order, args.trotter_steps, args.n_steps, args.overlap_tol),
                         (2, 1, 100, 1e-10))
        self.assertIs(run.call_args.kwargs["spectrum_builder"], driver.mean_field.compute_mean_field_spectrum)
        self.assertIs(run.call_args.kwargs["width_builder"], driver.mean_field.width_from_reference)
        self.assertEqual(args.outdir, driver.DEFAULT_OUTDIR)
        for options in (["--trotter-order", "1"], ["--evolution-mode", "sparse"], ["--delta-t", "0.05"]):
            with self.assertRaises(SystemExit):
                self.args(*options)


class RemainingSubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_batch_spool_execution_pins_workflow_and_defaults(self):
        for script in (SUBMIT, ARRAY):
            subprocess.run(["bash", "-n", str(script)], check=True)
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="16", QCANT_EVOLUTION_MODE="sparse",
                       QCANT_TROTTER_ORDER="1", QCANT_DELTA_T="0.05", MPLCONFIGDIR=tmp,
                       PYTHONPYCACHEPREFIX=tmp)
            result = subprocess.run(["bash", str(spool)], cwd=tmp, env=env,
                                    check=True, capture_output=True, text=True)
            options = result.stdout.split()
            for key, value in {"--case-index": "16", "--outdir": str(driver.DEFAULT_OUTDIR.relative_to(driver.qrte.ROOT)),
                               "--trotter-order": "2", "--evolution-mode": "trotter",
                               "--trotter-steps": "1", "--n-steps": "100", "--overlap-tol": "1e-10"}.items():
                self.assertEqual(options[options.index(key) + 1], value)
            self.assertNotIn("--delta-t", options)
            self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")

    def test_submission_preflight_blocks_missing_inputs_and_preserves_separate_job_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            command = '''
python_stub() { if [[ "$TEST_COUNT" == missing ]]; then printf 'Missing saved spectrum\\n' >&2; return 1; fi; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1" --time=01:00:00
'''
            env = self.clean_env()
            env.update(PYTHON="python_stub", TEST_COUNT="17")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    check=True, capture_output=True, text=True)
            self.assertIn("--array=0-16%4", result.stderr)
            self.assertIn("--time=01:00:00", result.stderr)
            record = Path(tmp) / "provenance/latest_qrte_mean_field_active_pi_over_w_remaining17_trotter2_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            for count in ("15", "missing"):
                env["TEST_COUNT"] = count
                result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("--parsable", result.stderr)
                self.assertEqual(record.read_text(), "12345\n")


if __name__ == "__main__":
    unittest.main()
