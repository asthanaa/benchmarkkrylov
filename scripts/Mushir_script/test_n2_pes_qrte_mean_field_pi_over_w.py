"""Read saved PES inputs and mock every SCF, QRTE, and sbatch calculation."""

import copy
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_n2_pes_qrte_mean_field_pi_over_w as driver

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH = SCRIPT_DIR / "run_n2_pes_qrte_mean_field_pi_over_w_array.sbatch"
SUBMIT = SCRIPT_DIR / "submit_talon_n2_pes_qrte_mean_field_pi_over_w.sh"


def fake_scf(case, case_dir, args):
    # Four closed core orbitals, six active-window orbitals. These are test data.
    return {"reference_scf": case["reference_settings"]["reference_scf"],
            "scf_converged": True, "scf_energy_hartree": case["cached_scf_energy_hartree"],
            "nelecas": case["nelecas"], "fock_operator": "test spatial Fock",
            "full_scf_orbital_energies_hartree": [-20, -19, -18, -17, -1.2, -.9, -.5, .1, .7, 1.4],
            "full_scf_orbital_occupations": [2] * 4 + ([2, 2, 2, 0, 0, 0] if case["spin"] == 0
                                                                else [2, 2, 1, 1, 0, 0])}


class PESInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = driver.load_cases(driver.parse_args([]))

    def test_all_16_points_match_old_caches_and_local_targets(self):
        self.assertEqual(len(self.cases), 16)
        self.assertEqual([c["task_index"] for c in self.cases], list(range(16)))
        self.assertEqual(sorted({c["bond_length_angstrom"] for c in self.cases}),
                         [.9, 1, 1.1, 1.35, 1.5, 1.7, 2, 2.4])
        for case in self.cases:
            saved = driver.qrte.load_json(Path(case["source_run_dir"]) / "summary.json")
            self.assertEqual(case["hamiltonian_cache_path"], saved["hamiltonian_cache_path"])
            self.assertEqual(case["local_reference_energy_hartree"], saved["local_reference_energy_hartree"])
            self.assertEqual(case["cached_scf_energy_hartree"], saved["scf_energy_hartree"])
            self.assertEqual(case["hamiltonian_sha256"], driver.sha256(case["hamiltonian_cache_path"]))
            self.assertIsNone(case["dataset_casscf_energy_hartree"])
        last = driver.load_cases(driver.parse_args(["--case-index", "15"]))[0]
        self.assertEqual(last["case_id"], "N2_R2p40_cc_pVTZ_2")
        for options in (["--case-index", "16"], ["--case-index", "-1"], ["--case-id", "N2_equi_cc_pVTZ_0"]):
            with self.assertRaises(ValueError):
                driver.load_cases(driver.parse_args(options))

    def test_dry_run_and_plan_never_compute_and_preserve_settings(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(driver.qrte, "compute_fock_spectrum") as scf, \
                patch.object(driver, "run_case") as run, redirect_stdout(io.StringIO()):
            driver.main(["--outdir", tmp, "--dry-run"])
            self.assertEqual(list(Path(tmp).iterdir()), [])
            driver.main(["--outdir", tmp, "--prepare-only"])
            scf.assert_not_called()
            run.assert_not_called()
            args = driver.parse_args(["--outdir", tmp])
            driver.check_plan(self.cases, args)
            plan = driver.qrte.load_json(Path(tmp) / "pes_plan.json")
            self.assertEqual((plan["task_count"], plan["n_steps"], plan["overlap_tol"],
                              plan["trotter_order"], plan["trotter_steps"]), (16, 100, 1e-10, 2, 1))
            changed = copy.copy(args)
            changed.n_steps = 200
            with self.assertRaisesRegex(ValueError, "Existing PES plan differs"):
                driver.check_plan(self.cases, changed)
            with self.assertRaisesRegex(ValueError, "Existing PES plan differs"):
                driver.check_plan([{**self.cases[0], "hamiltonian_sha256": "changed"}] + self.cases[1:], args)

    def test_reject_incompatible_propagation_and_source_overwrite(self):
        for options in (["--trotter-order", "1"], ["--trotter-steps", "4"], ["--delta-t", ".05"],
                        ["--evolution-mode", "sparse"], ["--n-steps", "0"], ["--overlap-tol", "nan"],
                        ["--outdir", str(driver.SOURCE_ROOT)], ["--outdir", str(driver.CACHE_ROOT / "new")]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(options)

    def test_geometry_mismatch_in_saved_source_is_rejected(self):
        load = driver.qrte.load_json

        def altered(path):
            value = load(path)
            if Path(path).name == "input_snapshot.json":
                value["geometry_angstrom"][1][2] = 1.20
            return value

        with patch.object(driver.qrte, "load_json", side_effect=altered), \
                self.assertRaisesRegex(ValueError, "geometry_angstrom"):
            driver.load_cases(driver.parse_args([]))


class SpectrumAndRunTests(unittest.TestCase):
    def test_each_geometry_gets_its_own_scf_and_timestep(self):
        cases = driver.load_cases(driver.parse_args([]))[::2]

        def geometry_scf(case, case_dir, args):
            result = fake_scf(case, case_dir, args)
            scale = case["bond_length_angstrom"]
            result["full_scf_orbital_energies_hartree"] = [scale * e for e in result["full_scf_orbital_energies_hartree"]]
            return result

        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(["--outdir", tmp])
            timesteps = []
            with patch.object(driver.qrte, "compute_fock_spectrum", side_effect=geometry_scf) as scf:
                for case in cases:
                    folder = Path(tmp) / case["case_id"]
                    folder.mkdir()
                    result = driver.compute_spectrum(case, folder, args)
                    dt = driver.width_from_reference(result)["delta_t"]
                    self.assertAlmostEqual(dt, np.pi / (9.6 * case["bond_length_angstrom"]))
                    timesteps.append(dt)
                self.assertEqual([call.args[0]["geometry_angstrom"][1][2] for call in scf.call_args_list],
                                 list(driver.old.DEFAULT_DISTANCES_ANGSTROM))
            self.assertEqual(len(set(timesteps)), 8)

    def test_active_determinant_widths_for_both_spins_and_reuse(self):
        cases = driver.load_cases(driver.parse_args([]))[:2]
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(["--outdir", tmp])
            with patch.object(driver.qrte, "compute_fock_spectrum", side_effect=fake_scf) as scf:
                for case, expected_w in zip(cases, (9.6, 8.4)):
                    path = Path(tmp) / case["case_id"]
                    path.mkdir()
                    result = driver.compute_spectrum(case, path, args)
                    width = driver.width_from_reference(result)
                    self.assertAlmostEqual(width["mean_field_active_bandwidth_hartree"], expected_w)
                    self.assertAlmostEqual(width["delta_t"], np.pi / expected_w)
                    self.assertEqual(width["active_orbital_indices_zero_based"], list(range(4, 10)))
                    self.assertEqual(driver.compute_spectrum(case, path, args), result)
                self.assertEqual(scf.call_count, 2)

    def test_wrong_geometry_method_energy_and_modified_spectrum_are_rejected(self):
        case = driver.load_cases(driver.parse_args(["--case-index", "0"]))[0]
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(["--outdir", tmp])
            path = Path(tmp) / case["case_id"]
            path.mkdir()
            raw_path = path / "fock_reference.json"
            with patch.object(driver.qrte, "compute_fock_spectrum", side_effect=fake_scf):
                active = driver.compute_spectrum(case, path, args)
            original = driver.qrte.load_json(raw_path)
            for field, value in (("reference_scf", "ROHF"), ("scf_energy_hartree", -1), ("scf_converged", False)):
                driver.qrte.write_json(raw_path, {**original, field: value})
                with self.assertRaises(ValueError):
                    driver.read_raw(case, args, raw_path)
            wrong_geometry = copy.deepcopy(original)
            wrong_geometry["fock_profile"]["case"]["geometry_angstrom"][1][2] = 1.20
            driver.qrte.write_json(raw_path, wrong_geometry)
            with self.assertRaises(ValueError):
                driver.read_raw(case, args, raw_path)
            driver.qrte.write_json(raw_path, original)
            active.update(status="completed", fock_profile={"case": case})
            driver.qrte.write_json(path / driver.mf.SPECTRUM_FILENAME, active)
            driver.validate_saved_spectrum(case, args)
            changed = copy.deepcopy(original)
            changed["full_scf_orbital_energies_hartree"][-1] += .1
            driver.qrte.write_json(raw_path, changed)
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                driver.validate_saved_spectrum(case, args)

    def test_reference_only_then_qrte_pins_geometry_payload_and_resumes(self):
        case = driver.load_cases(driver.parse_args(["--case-index", "15"]))[0]
        reference = case["local_reference_energy_hartree"]
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            np.testing.assert_equal(kwargs["geometry"], case["geometry_angstrom"])
            self.assertEqual(kwargs["spin"], 2)
            self.assertAlmostEqual(kwargs["delta_t"], np.pi / 8.4)
            self.assertEqual((kwargs["trotter_order"], kwargs["trotter_steps"], kwargs["n_steps"],
                              kwargs["overlap_tol"]), (2, 1, 100, 1e-10))
            self.assertEqual(kwargs["early_stop_energy"], reference + .0016)
            history = np.array([reference + .01, reference + .0005])
            return history, None, None, history

        qa.qrte_qulacs = Mock(side_effect=simulate)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(["--outdir", tmp, "--reference-only"])
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload) as loader, \
                    patch.object(driver.qrte, "compute_fock_spectrum", side_effect=fake_scf) as scf:
                driver.run_case(case, args)
                scf.assert_called_once()
                qa.qrte_qulacs.assert_not_called()
                args.reference_only = False
                driver.run_case(case, args)
                driver.run_case(case, args)
                qa.qrte_qulacs.assert_called_once()
                scf.assert_called_once()
                self.assertEqual(loader.call_args.args[0], Path(case["hamiltonian_cache_path"]))
            folder = Path(tmp) / case["case_id"]
            summary = driver.qrte.load_json(folder / "summary.json")
            self.assertEqual(summary["hit_iteration"], 2)
            with (folder / "min_energy_history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["error_vs_molvqe_casscf_hartree"] == "" for row in rows))
            self.assertEqual(rows[-1]["within_chemical_accuracy"], "true")


class SubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_spooled_batch_pins_propagation_and_six_hour_limit(self):
        for path in (BATCH, SUBMIT):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --time=06:00:00", BATCH.read_text())
        self.assertIn("#SBATCH --array=0-15%4", BATCH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(BATCH, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="15", QCANT_TROTTER_ORDER="1",
                       QCANT_TROTTER_STEPS="4", QCANT_DELTA_T=".05", PYTHONPYCACHEPREFIX=tmp)
            output = subprocess.run(["bash", str(spool)], env=env, check=True,
                                    text=True, capture_output=True).stdout.split()
            for key, value in {"--case-index": "15", "--trotter-order": "2", "--trotter-steps": "1",
                               "--evolution-mode": "trotter", "--scf-match-tolerance": "1e-6"}.items():
                self.assertEqual(output[output.index(key) + 1], value)
            self.assertNotIn("--delta-t", output)

    def test_submitter_preflights_before_mock_submission(self):
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
            env.update(PYTHON="python_stub", TEST_COUNT="16")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    text=True, capture_output=True, check=True)
            self.assertIn("--prepare-only", result.stderr)
            self.assertIn("--array=0-15%4", result.stderr)
            record = Path(tmp) / "provenance/latest_n2_pes_qrte_mean_field_pi_over_w_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "15"
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
