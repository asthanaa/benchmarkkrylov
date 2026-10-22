"""Correct-reference regression checks with mocked QRTE and Slurm."""

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

import run_qrte_co_spin2_trotter1_dt0p05 as driver

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_co_spin2_trotter1_dt0p05.sh"
BATCH = SCRIPT_DIR / "run_qrte_co_spin2_trotter1_dt0p05.sbatch"


class COReferenceTests(unittest.TestCase):
    def args(self, *options):
        with patch.object(sys, "argv", ["driver", *options]):
            return driver.parse_args()

    def test_defaults_select_co_and_preserve_both_distinct_reference_energies(self):
        args = self.args()
        self.assertEqual((args.delta_t, args.trotter_order, args.trotter_steps), (0.05, 1, 1))
        self.assertEqual((args.n_steps, args.overlap_tol), (100, 1e-10))
        cases = driver.load_cases(args)
        self.assertEqual([case["case_id"] for case in cases], [driver.CASE_ID])
        case = cases[0]
        self.assertAlmostEqual(case["local_reference_energy_hartree"], -112.65639016965865, places=12)
        self.assertAlmostEqual(case["dataset_casscf_energy_hartree"], -112.60166830789687, places=12)
        self.assertAlmostEqual(case["saved_exact_sector_ground_energy_hartree"], case["local_reference_energy_hartree"], places=10)
        self.assertEqual(driver.load_cases(self.args("--case-index", "0"))[0]["case_id"], driver.CASE_ID)

    def test_incompatible_propagation_and_other_cases_are_rejected(self):
        for options in (["--delta-t", "0.1"], ["--trotter-order", "2"],
                        ["--trotter-steps", "4"], ["--evolution-mode", "sparse"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)
        for options in (["--case-id", "CO_equi_cc_pVTZ_0"], ["--case-index", "1"]):
            with self.assertRaises(ValueError):
                driver.load_cases(self.args(*options))

    def test_invalid_saved_ground_or_changed_source_is_rejected(self):
        case = driver.load_cases(self.args())[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spectrum.json"
            spectrum = driver.qrte.load_json(driver.SAVED_SPECTRUM)
            bad = copy.deepcopy(spectrum)
            bad["sector_ground_energy_hartree"] = case["dataset_casscf_energy_hartree"]
            driver.qrte.write_json(path, bad)
            unchecked = dict(case)
            unchecked.pop("saved_cas_spectrum_sha256")
            with patch.object(driver, "SAVED_SPECTRUM", path):
                with self.assertRaisesRegex(ValueError, "differs from the cached reference"):
                    driver.read_saved_spectrum(unchecked)
                with self.assertRaisesRegex(ValueError, "changed after validation"):
                    driver.read_saved_spectrum(case)

    def test_old_false_hit_cannot_stop_rerun_and_resume_preserves_correct_target(self):
        args = self.args()
        case = driver.load_cases(args)[0]
        old = driver.qrte.load_json(driver.qrte.ROOT / "runs/qrte_natorb_dt0.05" / driver.CASE_ID / "summary.json")
        old_energy = old["final_min_history_energy_hartree"]
        reference = case["local_reference_energy_hartree"]
        self.assertGreater(old_energy - reference, 0.0016)
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())
        original_builder = qa._build_hamiltonian_payload
        payload = {"pinned": True}

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["early_stop_energy"], reference + 0.0016)
            self.assertGreater(old_energy, kwargs["early_stop_energy"])
            self.assertEqual((kwargs["delta_t"], kwargs["trotter_order"], kwargs["trotter_steps"]), (0.05, 1, 1))
            values = np.array([old_energy, reference + 0.0005])
            return values, None, None, values

        qa.qrte_qulacs = Mock(side_effect=simulate)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload), \
                    patch.object(driver.qrte, "compute_fock_spectrum") as scf:
                def run():
                    driver.qrte.run_case(case, args, spectrum_builder=driver.saved_spectrum,
                                         width_builder=driver.cas.width_from_reference,
                                         spectrum_scope=driver.cas.SPECTRUM_SCOPE,
                                         spectrum_filename=driver.cas.SPECTRUM_FILENAME)
                run()
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                scf.assert_not_called()
                self.assertIs(qa._build_hamiltonian_payload, original_builder)
            summary = driver.qrte.load_json(args.outdir / driver.CASE_ID / "summary.json")
            self.assertEqual(summary["hit_iteration"], 2)
            self.assertEqual(summary["casscf_energy_hartree"], reference)
            self.assertEqual(summary["dataset_casscf_energy_hartree"], case["dataset_casscf_energy_hartree"])
            with (args.outdir / driver.CASE_ID / "min_energy_history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["within_chemical_accuracy"], "false")
            self.assertEqual(rows[1]["within_chemical_accuracy"], "true")


class COSubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_single_case_spool_execution_uses_six_hours_and_pins_settings(self):
        for path in (SUBMIT, BATCH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --time=06:00:00", BATCH.read_text())
        self.assertNotIn("#SBATCH --array", BATCH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(BATCH, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", QCANT_DELTA_T="1", QCANT_TROTTER_ORDER="2",
                       QCANT_TROTTER_STEPS="4", QCANT_EVOLUTION_MODE="sparse",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            result = subprocess.run(["bash", str(spool)], env=env, check=True, capture_output=True, text=True)
            options = result.stdout.split()
            for key, value in {"--delta-t": "0.05", "--trotter-order": "1", "--trotter-steps": "1",
                               "--evolution-mode": "trotter", "--n-steps": "100"}.items():
                self.assertEqual(options[options.index(key) + 1], value)
            self.assertNotIn("--case-index", options)
            self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")

    def test_submission_preflight_with_stubbed_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            command = '''
python_stub() { printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1"
'''
            env = self.clean_env()
            env.update(PYTHON="python_stub", TEST_COUNT="1")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    check=True, capture_output=True, text=True)
            self.assertNotIn("--array", result.stderr)
            self.assertIn(BATCH.name, result.stderr)
            record = Path(tmp) / "provenance/latest_qrte_co_spin2_trotter1_dt0p05_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "10"
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
