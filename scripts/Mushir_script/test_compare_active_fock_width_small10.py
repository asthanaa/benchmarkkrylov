"""Synthetic checks for the active-orbital width comparison."""

import math
import unittest

import numpy as np

import compare_active_fock_width_small10 as compare


class ActiveFockWidthTests(unittest.TestCase):
    def test_determinant_width_is_not_the_one_electron_range(self):
        result = compare.orbital_estimates([-1, 0, 1], [-1, 0, 1], [1, 1])
        self.assertEqual(result["one_electron_range_hartree"], 2)
        self.assertEqual(result["determinant_width_hartree"], 4)
        self.assertEqual(result["delta_t_pi_over_estimated_w"], math.pi / 4)

    def test_empty_spin_channel_contributes_zero_width(self):
        result = compare.orbital_estimates([-1, 0, 1], [-2, 0, 2], [1, 0])
        self.assertEqual(result["determinant_width_hartree"], 2)

    def test_canonical_window_excludes_core_and_external_virtuals(self):
        case = {"active_electrons": 2, "active_orbitals": 2, "nelecas": [1, 1],
                "active_orbital_indices": [], "active_orbital_index_base": 0,
                "active_orbital_labels": None}
        reference = {"full_scf_orbital_energies_hartree": [-10, -1, 1, 20],
                     "full_scf_orbital_occupations": [2, 2, 0, 0],
                     "reference_scf": "RHF", "fock_operator": "restricted spatial Fock"}
        result = compare.canonical_window(case, reference)
        self.assertEqual(result["active_orbital_indices_zero_based"], [1, 2])
        self.assertEqual(result["determinant_width_hartree"], 4)
        self.assertEqual(result["ncore"], 1)

    def test_recorded_indices_and_avas_rohf_caveats(self):
        case = {"active_electrons": 1, "active_orbitals": 2, "nelecas": [1, 0],
                "active_orbital_indices": [2, 4], "active_orbital_index_base": 1,
                "active_orbital_labels": ["C 2p"]}
        reference = {"full_scf_orbital_energies_hartree": [-10, -1, 1, 20],
                     "full_scf_orbital_occupations": [2, 1, 0, 0],
                     "reference_scf": "ROHF", "fock_operator": "effective Roothaan Fock"}
        result = compare.canonical_window(case, reference)
        self.assertEqual(result["active_orbital_indices_zero_based"], [1, 3])
        self.assertIn("avas_caveat", result)
        self.assertIn("open_shell_caveat", result)

    def test_frozen_cas_fock_uses_coulomb_and_spin_resolved_exchange(self):
        h1 = np.diag([-1.0, 0.5])
        eri = np.zeros((2, 2, 2, 2))
        eri[0, 0, 0, 0] = 0.7
        eri[1, 1, 1, 1] = 0.3
        eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2
        closed = compare.frozen_cas_fock(h1, eri, [1, 1])
        np.testing.assert_allclose(closed["alpha_orbital_energies_hartree"], [-0.3, 0.9])
        np.testing.assert_allclose(closed["beta_orbital_energies_hartree"], [-0.3, 0.9])
        self.assertAlmostEqual(closed["determinant_width_hartree"], 2.4)
        opened = compare.frozen_cas_fock(h1, eri, [1, 0])
        np.testing.assert_allclose(opened["alpha_orbital_energies_hartree"], [-1.0, 0.7])
        np.testing.assert_allclose(opened["beta_orbital_energies_hartree"], [-0.3, 0.7])
        self.assertAlmostEqual(opened["determinant_width_hartree"], 1.7)
        self.assertTrue(opened["active_subspace_matches_optimized_casscf"])

    def test_invalid_integrals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "integrals"):
            compare.frozen_cas_fock(np.eye(2), np.zeros((2, 2)), [1, 1])
        with self.assertRaisesRegex(ValueError, "occupations"):
            compare.frozen_cas_fock(np.eye(2), np.zeros((2, 2, 2, 2)), [3, 1])


if __name__ == "__main__":
    unittest.main()
