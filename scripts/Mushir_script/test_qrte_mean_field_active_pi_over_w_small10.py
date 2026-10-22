"""Saved-spectrum and submission checks; no molecular simulations are run."""

import copy
import hashlib
import math
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

import run_qrte_mean_field_active_pi_over_w_small10 as driver

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_mean_field_active_pi_over_w_small10_trotter2.sh"
ARRAY = SCRIPT_DIR / "run_qrte_mean_field_active_pi_over_w_small10_trotter2_array.sbatch"
OUTDIR = "runs/qrte_mean_field_active_pi_over_w_small10_trotter2"


class MeanFieldActiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["driver"]):
            cls.args = driver.parse_args()
        cls.cases = driver.load_cases(cls.args)

    def test_defaults_and_all_ten_widths_match_saved_comparison(self):
        self.assertEqual(self.args.trotter_order, 2)
        self.assertEqual(self.args.n_steps, 100)
        self.assertEqual(self.args.overlap_tol, 1e-10)
        self.assertEqual(self.args.trotter_steps, 1)
        self.assertIsNone(self.args.delta_t)
        self.assertEqual(tuple(c["case_id"] for c in self.cases), driver.CASE_IDS)
        comparison = driver.qrte.load_json(driver.qrte.ROOT / "runs/active_fock_width_small10/comparison.json")
        by_id = {row["case_id"]: row["canonical_hf_window"] for row in comparison["cases"]}
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                result = driver.compute_mean_field_spectrum(case, None, self.args)
                width = driver.width_from_reference(result)
                expected = by_id[case["case_id"]]["determinant_width_hartree"]
                self.assertAlmostEqual(width["mean_field_active_bandwidth_hartree"], expected, places=12)
                self.assertAlmostEqual(width["delta_t"], math.pi / expected, places=12)
                self.assertEqual(result["reference_scf"], case["reference_settings"]["reference_scf"])
                self.assertFalse(result["new_reference_calculation"])

    def test_incompatible_workflow_overrides_are_rejected(self):
        for options in (["--trotter-order", "1"], ["--evolution-mode", "sparse"], ["--delta-t", "0.05"]):
            with self.subTest(options=options), patch.object(sys, "argv", ["driver", *options]):
                with self.assertRaisesRegex(SystemExit, "requires dt=pi/W_MF"):
                    driver.parse_args()

    def test_width_recomputed_from_orbital_sums_and_caveats_preserved(self):
        for case in self.cases:
            result = driver.compute_mean_field_spectrum(case, None, self.args)
            estimate = result["active_orbital_estimate"]
            expected = estimate["determinant_width_hartree"]
            estimate["determinant_width_hartree"] = 999
            width = driver.width_from_reference(result)
            self.assertAlmostEqual(width["fock_bandwidth_hartree"], expected)
            self.assertEqual(width["mean_field_active_estimate"]["determinant_width_hartree"], expected)
            if case["active_orbital_labels"]:
                self.assertIn("avas_caveat", width["mean_field_active_estimate"])
            if result["reference_scf"] == "ROHF":
                self.assertIn("open_shell_caveat", width["mean_field_active_estimate"])

    def test_saved_reference_provenance_and_convergence_are_checked(self):
        case = self.cases[0]
        original = driver.qrte.load_json(case["mean_field_reference_path"])
        mutations = (
            (lambda r: r.update(scf_converged=False), "converged"),
            (lambda r: r.update(status="failed"), "completed"),
            (lambda r: r.update(scf_energy_hartree="nan"), "finite"),
            (lambda r: r.update(reference_scf="UHF"), "method"),
            (lambda r: r.update(nelecas=[1, 1]), "occupations"),
            (lambda r: r["fock_profile"]["case"].update(hamiltonian_sha256="wrong"), "hamiltonian_sha256"),
            (lambda r: r["fock_profile"]["case"].update(spin=2), "spin"),
            (lambda r: r["fock_profile"]["case"].update(active_orbital_indices=[0, 1]), "active_orbital_indices"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            local_case = {**case, "mean_field_reference_path": str(Path(tmp) / "reference.json")}
            local_case.pop("mean_field_reference_sha256")
            for mutate, message in mutations:
                with self.subTest(message=message):
                    reference = copy.deepcopy(original)
                    mutate(reference)
                    driver.qrte.write_json(Path(local_case["mean_field_reference_path"]), reference)
                    with self.assertRaisesRegex(ValueError, message):
                        driver.read_saved_reference(local_case, self.args)
            driver.qrte.write_json(Path(local_case["mean_field_reference_path"]), original)
            local_case["mean_field_reference_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "changed after input validation"):
                driver.read_saved_reference(local_case, self.args)

    def test_driver_uses_original_payload_and_resumes_only_with_same_source(self):
        case = dict(self.cases[0])
        args = copy.copy(self.args)
        reference_energy = case["local_reference_energy_hartree"]
        payload = {"pinned": True}
        original_builder = Mock()
        qa = types.SimpleNamespace(_build_hamiltonian_payload=original_builder)

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["trotter_order"], 2)
            self.assertEqual(kwargs["trotter_steps"], 1)
            self.assertEqual(kwargs["n_steps"], 100)
            self.assertEqual(kwargs["overlap_tol"], 1e-10)
            self.assertEqual(kwargs["evolution_mode"], "trotter")
            self.assertEqual(kwargs["early_stop_energy"], reference_energy + driver.qrte.CHEMICAL_ACCURACY)
            self.assertEqual(kwargs["hamiltonian_source"], "casscf_natorb")
            self.assertEqual(kwargs["spin"], case["spin"])
            width = driver.width_from_reference(driver.compute_mean_field_spectrum(case, None, args))
            self.assertEqual(kwargs["delta_t"], width["delta_t"])
            history = np.array([reference_energy + 0.0005])
            return history, None, None, history

        qa.qrte_qulacs = Mock(side_effect=simulate)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp) / "outputs"
            path = Path(tmp) / "reference.json"
            shutil.copyfile(case["mean_field_reference_path"], path)
            case["mean_field_reference_path"] = str(path)
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload) as build_payload:
                def run():
                    driver.qrte.run_case(case, args, spectrum_builder=driver.compute_mean_field_spectrum,
                                        width_builder=driver.width_from_reference,
                                        spectrum_scope=driver.SPECTRUM_SCOPE,
                                        spectrum_filename=driver.SPECTRUM_FILENAME)
                run()
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                self.assertEqual(build_payload.call_args.args[0], Path(case["hamiltonian_cache_path"]))
                self.assertIs(qa._build_hamiltonian_payload, original_builder)
                summary = driver.qrte.load_json(args.outdir / case["case_id"] / "summary.json")
                config = driver.qrte.load_json(args.outdir / case["case_id"] / "run_config.json")
                self.assertEqual(summary["trotter_order"], 2)
                self.assertEqual(config["trotter_order"], 2)
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["hit_iteration"], 1)
                self.assertEqual(summary["source_mean_field_reference_sha256"], case["mean_field_reference_sha256"])
                self.assertEqual(summary["hamiltonian_sha256"], case["hamiltonian_sha256"])
                reference = driver.qrte.load_json(path)
                reference["used_newton_scf"] = not reference["used_newton_scf"]
                driver.qrte.write_json(path, reference)
                case["mean_field_reference_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 2)


class SubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {k: v for k, v in os.environ.items() if not k.startswith("QCANT_")}

    def test_shell_syntax(self):
        for script in (SUBMIT, ARRAY):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_batch_from_spool_pins_second_order_and_no_fixed_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool)
            for script, steps in ((ARRAY, "1"), (spool, "3")):
                env = self.clean_env()
                env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="7",
                           QCANT_EVOLUTION_MODE="sparse", QCANT_TROTTER_ORDER="1",
                           QCANT_DELTA_T="0.05", QCANT_TROTTER_STEPS=steps,
                           MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
                result = subprocess.run(["bash", str(script)], cwd=tmp, env=env,
                                        check=True, capture_output=True, text=True)
                options = result.stdout.split()
                for key, value in {"--case-index": "7", "--outdir": OUTDIR,
                                   "--trotter-order": "2", "--evolution-mode": "trotter",
                                   "--trotter-steps": steps, "--n-steps": "100",
                                   "--overlap-tol": "1e-10"}.items():
                    self.assertEqual(options[options.index(key) + 1], value)
                self.assertNotIn("--delta-t", options)
                self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")

    def test_submit_preflight_forwards_options_and_records_separate_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            command = '''
python_stub() { printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1" --time=01:00:00
'''
            env = self.clean_env()
            env.update(PYTHON="python_stub", TEST_COUNT="10")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)],
                                    env=env, check=True, capture_output=True, text=True)
            self.assertIn("--array=0-9%4", result.stderr)
            self.assertIn("--time=01:00:00", result.stderr)
            self.assertIn(ARRAY.name, result.stderr)
            record = Path(tmp) / "provenance/latest_qrte_mean_field_active_pi_over_w_small10_trotter2_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "9"
            failure = subprocess.run(["bash", "-c", command, "test", str(submit)],
                                     env=env, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("Expected ten", failure.stderr)
            self.assertNotIn("--parsable", failure.stderr)


if __name__ == "__main__":
    unittest.main()
