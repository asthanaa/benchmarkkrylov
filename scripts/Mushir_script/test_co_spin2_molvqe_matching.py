"""Check the selected CO solution, timestep convention, and pinned run dispatch."""

import copy
import itertools
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_co_spin2_molvqe_matching as driver


class MatchingCOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = driver.load_reference()

    def test_prepared_hamiltonian_matches_dataset_and_exact_triplet_ground(self):
        reference = self.reference
        case = reference["case"]
        self.assertAlmostEqual(case["local_reference_energy_hartree"], -112.60166830789687, places=8)
        self.assertAlmostEqual(reference["exact_sector_ground_energy_hartree"],
                               case["local_reference_energy_hartree"], places=8)
        self.assertAlmostEqual(reference["exact_sector_ground_spin_square"], 2, places=9)
        self.assertEqual(case["nelecas"], [4, 2])

    def test_original_lower_energy_hamiltonian_is_rejected(self):
        reference = copy.deepcopy(self.reference)
        with (driver.ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv").open(newline="") as handle:
            import csv
            original = next(row for row in csv.DictReader(handle) if row["case_id"] == driver.CASE_ID)
        path = driver.ROOT / original["consolidated_cache_path"]
        reference["case"].update(hamiltonian_cache_path=str(path), hamiltonian_sha256=driver.digest(path))
        with patch.object(driver.qrte, "load_json", return_value=reference):
            with self.assertRaisesRegex(ValueError, "accuracy reference differ"):
                driver.load_reference()

    def test_mean_field_time_uses_fixed_spin_determinant_sums(self):
        args = driver.parse_args(["--dry-run"])
        configs = driver.configurations(args, self.reference)
        eps = np.asarray(self.reference["full_scf_orbital_energies_hartree"])[4:10]
        sums = [sum(eps[list(a)]) + sum(eps[list(b)])
                for a in itertools.combinations(range(6), 4)
                for b in itertools.combinations(range(6), 2)]
        self.assertAlmostEqual(configs[1]["delta_t"], math.pi / (max(sums) - min(sums)), places=12)
        self.assertEqual([c["trotter_order"] for c in configs], [1, 2, 1])
        self.assertEqual([c["trotter_steps"] for c in configs], [1, 1, 1])

    def test_all_three_jobs_pin_new_payload_and_matching_accuracy_target(self):
        case = self.reference["case"]
        reference_energy = case["local_reference_energy_hartree"]
        payload = {"prepared_CO_hamiltonian": True}
        original_builder = Mock(side_effect=AssertionError("Must not rebuild a Hamiltonian"))
        qa = types.SimpleNamespace(_build_hamiltonian_payload=original_builder)
        calls = []

        def simulate(method, **kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertAlmostEqual(kwargs["early_stop_energy"], reference_energy + .0016)
            self.assertEqual(kwargs["trotter_steps"], 1)
            self.assertEqual(kwargs["evolution_mode"], "trotter")
            self.assertEqual(kwargs["spin"], 2)
            calls.append((method, kwargs))
            history = np.array([reference_energy + .005, reference_energy + .0005])
            return (history, None, None, history) if method == "qrte" else (history, None, history)

        qa.qrte_qulacs = lambda **kwargs: simulate("qrte", **kwargs)
        qa.qkud_qulacs = lambda **kwargs: simulate("qkud", **kwargs)
        module = types.ModuleType("QCANT")
        module.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            for index, task in enumerate(driver.TASKS):
                with patch.dict(sys.modules, {"QCANT": module}), \
                        patch.object(sys, "argv", ["driver", "--task-index", str(index), "--outdir", tmp]), \
                        patch.object(driver, "_payload_from_cache_path", return_value=payload) as qkud_payload, \
                        patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload) as qrte_payload:
                    driver.main()
                    loader = qkud_payload if index == 2 else qrte_payload
                    self.assertEqual(loader.call_args.args[0], driver.CACHE_PATH)
                self.assertIs(qa._build_hamiltonian_payload, original_builder)
                summary = driver.qrte.load_json(Path(tmp) / task["name"] / driver.CASE_ID / "summary.json")
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["hamiltonian_sha256"], case["hamiltonian_sha256"])
                self.assertEqual(summary["trotter_order"], task["trotter_order"])
                self.assertEqual(summary["hit_iteration"], 2)
            self.assertEqual(calls[0][1]["delta_t"], .05)
            self.assertEqual(calls[0][1]["trotter_order"], 1)
            self.assertAlmostEqual(calls[1][1]["delta_t"], .9985305567033425, places=12)
            self.assertEqual(calls[1][1]["trotter_order"], 2)
            self.assertEqual(calls[2][1]["epsilon"], .05)
            original_builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
