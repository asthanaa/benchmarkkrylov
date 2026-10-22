"""Synthetic determinant-spectrum tests; no molecular SCF or QRTE runs."""

import itertools
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_qrte_uhf_determinant_pi_over_w_small10 as driver


class UHFDeterminantTests(unittest.TestCase):
    def args(self):
        with patch.object(sys, "argv", ["run_qrte_uhf_determinant_pi_over_w_small10.py"]):
            return driver.parse_args()

    def test_width_matches_enumerated_fixed_spin_determinants(self):
        alpha, beta = np.array([3.0, -5.0, 0.5, -1.0]), np.array([1.0, -4.0, 2.0, -0.5])
        energies = [float(alpha[list(a)].sum() + beta[list(b)].sum())
                    for a in itertools.combinations(range(4), 2)
                    for b in itertools.combinations(range(4), 1)]
        width = driver.determinant_bandwidth(alpha, beta, [2, 1])
        self.assertEqual(width["lowest_determinant_orbital_sum_hartree"], min(energies))
        self.assertEqual(width["highest_determinant_orbital_sum_hartree"], max(energies))
        self.assertAlmostEqual(width["delta_t"], math.pi / (max(energies) - min(energies)))
        self.assertNotEqual(width["fock_bandwidth_hartree"], max(alpha.max(), beta.max()) - min(alpha.min(), beta.min()))

    def test_empty_spin_sector_and_invalid_spectra(self):
        width = driver.determinant_bandwidth([-2, 1], [-3, 4], [1, 0])
        self.assertEqual(width["fock_bandwidth_hartree"], 3)
        self.assertEqual(width["highest_determinant_beta_orbital_indices"], [])
        for alpha, beta, occupations in (
            ([], [0, 1], [1, 1]), ([0, float("nan")], [0, 1], [1, 1]),
            ([0, 1], [0, 1], [3, 1]), ([0, 1], [0, 1], [1.5, 1]),
            ([0, 1], [0, 1], [-1, 1]), ([0, 1], [0, 1], [2, 2]),
        ):
            with self.subTest(occupations=occupations), self.assertRaises(ValueError):
                driver.determinant_bandwidth(alpha, beta, occupations)

    def test_common_offset_cancels_in_width(self):
        result = {"full_uhf_alpha_orbital_energies_hartree": [-3, -1, 2],
                  "full_uhf_beta_orbital_energies_hartree": [-2, 0, 4],
                  "full_system_spin_occupations": [2, 1],
                  "uhf_fock_constant_offset_hartree": 10.0}
        first = driver.width_from_reference(result)
        result["uhf_fock_constant_offset_hartree"] = -100.0
        second = driver.width_from_reference(result)
        self.assertEqual(first["fock_bandwidth_hartree"], second["fock_bandwidth_hartree"])
        self.assertEqual(first["delta_t"], second["delta_t"])
        self.assertAlmostEqual(first["fock_spectral_max_hartree"] - first["fock_spectral_min_hartree"],
                               first["fock_bandwidth_hartree"])

    def test_subset_indices_are_stable_and_qrte_caches_are_identical(self):
        args = self.args()
        cases = driver.load_cases(args)
        self.assertEqual(tuple(case["case_id"] for case in cases), driver.CASE_IDS)
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(case["active_orbitals"] <= 6 for case in cases))
        by_id = {case["case_id"]: case for case in driver.qrte.load_cases(args)}
        for i, case in enumerate(cases):
            self.assertEqual(case, by_id[case["case_id"]])
            args.case_index = i
            self.assertEqual(driver.load_cases(args), [case])
        self.assertEqual(args.outdir, driver.DEFAULT_OUTDIR)
        self.assertEqual((args.n_steps, args.overlap_tol, args.evolution_mode, args.trotter_steps),
                         (100, 1e-10, "trotter", 1))

    def test_uhf_uses_full_electron_counts_without_rhf_energy_matching(self):
        args = self.args()
        args.case_id = ["N2_equi_cc_pVTZ_2"]
        case = driver.load_cases(args)[0]
        case["cached_scf_energy_hartree"] = -1000.0
        alpha, beta = np.linspace(-10, 3, 12), np.linspace(-9, 4, 12)
        mo_occ = np.zeros((2, 12))
        mo_occ[0, :8], mo_occ[1, :6] = 1, 1
        mf = types.SimpleNamespace(converged=True, e_tot=-100.0, mo_energy=(alpha, beta),
                                   mo_occ=mo_occ, init_guess="minao", init_guess_breaksym=1,
                                   kernel=Mock(), spin_square=Mock(return_value=(2.1, 3.06)))
        mol = types.SimpleNamespace(nelec=(8, 6))
        fake_pyscf = types.ModuleType("pyscf")
        fake_pyscf.gto = types.SimpleNamespace(M=Mock(return_value=mol))
        fake_pyscf.scf = types.SimpleNamespace(UHF=Mock(return_value=mf))
        with patch.dict(sys.modules, {"pyscf": fake_pyscf}):
            result = driver.compute_uhf_spectrum(case, Path("/tmp"), args)
        fake_pyscf.scf.UHF.assert_called_once_with(mol)
        self.assertEqual(result["full_system_spin_occupations"], [8, 6])
        self.assertNotEqual(result["full_system_spin_occupations"], case["nelecas"])
        self.assertEqual(result["n_scf_spatial_orbitals_per_spin"], 12)
        self.assertEqual(result["reference_scf"], "UHF")
        self.assertEqual(mf.level_shift, case["reference_settings"]["scf_level_shift"])
        width = driver.width_from_reference(result)
        self.assertAlmostEqual(width["fock_spectral_min_hartree"], mf.e_tot)

    def test_nonconverged_uhf_cannot_start_qrte(self):
        args = self.args()
        case = driver.load_cases(args)[0]
        case["reference_settings"]["scf_newton_if_not_converged"] = False
        mf = types.SimpleNamespace(converged=False, e_tot=-1.0, kernel=Mock(),
                                   init_guess="minao", init_guess_breaksym=1)
        fake_pyscf = types.ModuleType("pyscf")
        fake_pyscf.gto = types.SimpleNamespace(M=Mock())
        fake_pyscf.scf = types.SimpleNamespace(UHF=Mock(return_value=mf))
        with patch.dict(sys.modules, {"pyscf": fake_pyscf}), self.assertRaises(RuntimeError):
            driver.compute_uhf_spectrum(case, Path("/tmp"), args)

    def test_shared_driver_changes_only_width_source(self):
        args = self.args()
        case = driver.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        fock_result = {"full_uhf_alpha_orbital_energies_hartree": [-3, -1, 1, 2],
                       "full_uhf_beta_orbital_energies_hartree": [-2, 0, 2, 4],
                       "full_system_spin_occupations": [2, 2],
                       "uhf_fock_constant_offset_hartree": 0.5}
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=lambda *a, **kw: None)

        def qrte(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertAlmostEqual(kwargs["delta_t"], driver.width_from_reference(fock_result)["delta_t"])
            self.assertEqual(kwargs["early_stop_energy"], reference + driver.qrte.CHEMICAL_ACCURACY)
            self.assertEqual((kwargs["n_steps"], kwargs["overlap_tol"], kwargs["evolution_mode"], kwargs["trotter_steps"]),
                             (100, 1e-10, "trotter", 1))
            history = np.array([reference + 0.003, reference + 0.0005])
            return history[-1:], None, None, history

        qa.qrte_qulacs = Mock(side_effect=qrte)
        fake_qcant = types.ModuleType("QCANT")
        fake_qcant.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            with patch.dict(sys.modules, {"QCANT": fake_qcant}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload) as loader:
                builder = Mock(return_value=dict(fock_result))
                for _ in range(2):
                    driver.qrte.run_case(case, args, spectrum_builder=builder,
                                         width_builder=driver.width_from_reference,
                                         spectrum_scope=driver.SPECTRUM_SCOPE)
                self.assertEqual(builder.call_count, 1)
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                loader.assert_called_once_with(Path(case["hamiltonian_cache_path"]),
                                               hamiltonian_cutoff=1e-20, qa=qa)
                summary = driver.qrte.load_json(args.outdir / case["case_id"] / "summary.json")
                self.assertEqual(summary["hit_iteration"], 2)
                self.assertEqual(summary["width_scf_method"], "UHF")
                self.assertEqual(summary["run_profile"]["spectrum_scope"], driver.SPECTRUM_SCOPE)


if __name__ == "__main__":
    unittest.main()
