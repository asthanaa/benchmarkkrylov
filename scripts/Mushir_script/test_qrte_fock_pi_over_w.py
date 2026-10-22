"""Configuration and synthetic-data tests; no molecular calculations."""

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_qrte_fock_pi_over_w as driver


class FockQRTEConfigurationTests(unittest.TestCase):
    def args(self):
        with patch.object(sys, "argv", ["run_qrte_fock_pi_over_w.py"]):
            return driver.parse_args()

    def test_full_orbital_width_includes_core_and_virtuals(self):
        width = driver.fock_bandwidth([3.0, -20.0, -0.2, 0.1])
        self.assertEqual(width["fock_spectral_min_hartree"], -20.0)
        self.assertEqual(width["fock_spectral_max_hartree"], 3.0)
        self.assertEqual(width["fock_bandwidth_hartree"], 23.0)
        self.assertAlmostEqual(width["delta_t"], math.pi / 23.0)

    def test_invalid_spectra_cannot_select_a_timestep(self):
        for values in ([], [1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0], [[1.0, 2.0]], 1.0):
            with self.subTest(values=values), self.assertRaises(ValueError):
                driver.fock_bandwidth(values)

    def test_reference_and_corrected_orbital_provenance(self):
        cases = {case["case_id"]: case for case in driver.load_cases(self.args())}
        self.assertEqual(len(cases), 27)
        self.assertNotIn("Ferrocene_ceo", cases)
        self.assertNotIn("feo4_2minus_ceo", cases)
        for case_id in ("feno_2plus", "fenoco3_minus"):
            self.assertEqual(cases[case_id]["reference_settings"]["reference_scf"], "sfX2C1eROHF")
        self.assertEqual(cases["FeO"]["reference_settings"]["reference_scf"], "ROHF")
        self.assertEqual(cases["FeO"]["reference_settings"]["geometry_unit"], "Bohr")
        self.assertEqual(cases["FeO"]["active_orbital_labels"], ["Fe 3d", "O 2p"])
        for case_id, indices in {
            "cyclobutadiene_ceo": [11, 13, 14, 19],
            "m_Benzyne_equi_cc_pVTZ_0": [15, 18, 19, 20, 24, 27],
            "m_Benzyne_equi_cc_pVTZ_2": [16, 18, 19, 20, 24, 27],
        }.items():
            self.assertEqual(cases[case_id]["active_orbital_indices"], indices)
        self.assertFalse(cases["fenoco3_minus"]["local_reference_converged"])
        self.assertNotEqual(cases["CO_equi_cc_pVTZ_2"]["local_reference_energy_hartree"],
                            cases["CO_equi_cc_pVTZ_2"]["dataset_casscf_energy_hartree"])

    def test_cache_builder_is_restored_on_failure(self):
        original = lambda *args, **kwargs: "original"
        qa = types.SimpleNamespace(_build_hamiltonian_payload=original)
        with self.assertRaises(RuntimeError):
            with driver.cached_hamiltonian(qa, {"pinned": True}):
                self.assertEqual(qa._build_hamiltonian_payload(), {"pinned": True})
                raise RuntimeError("synthetic failure")
        self.assertIs(qa._build_hamiltonian_payload, original)

    def test_scf_uses_original_method_geometry_and_every_orbital(self):
        args = self.args()
        cases = {case["case_id"]: case for case in driver.load_cases(args)}
        for case_id in ("H4_near_equi_sto3g_0", "FeO", "feno_2plus", "fenoco3_minus"):
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                settings = case["reference_settings"]
                mf = types.SimpleNamespace(
                    converged=True, e_tot=case["cached_scf_energy_hartree"] or -1.0,
                    mo_energy=np.linspace(-20.0, 3.0, 101), mo_occ=np.zeros(101),
                    kernel=Mock(),
                )
                mol = object()
                gto = types.SimpleNamespace(M=Mock(return_value=mol))
                scf = types.SimpleNamespace(RHF=Mock(return_value=mf), ROHF=Mock(return_value=mf),
                                            sfx2c1e=Mock(side_effect=lambda value: value))
                fake_pyscf = types.ModuleType("pyscf")
                fake_pyscf.gto, fake_pyscf.scf = gto, scf
                with patch.dict(sys.modules, {"pyscf": fake_pyscf}):
                    result = driver.compute_fock_spectrum(case, Path("/tmp"), args)
                self.assertEqual(result["n_scf_spatial_orbitals"], 101)
                self.assertEqual(len(result["full_scf_orbital_energies_hartree"]), 101)
                self.assertEqual(gto.M.call_args.kwargs["unit"], settings["geometry_unit"])
                self.assertEqual(mf.level_shift, settings["scf_level_shift"])
                self.assertEqual(mf.max_cycle, settings["scf_max_cycle"])
                if settings["reference_scf"] == "RHF":
                    scf.RHF.assert_called_once_with(mol)
                    scf.ROHF.assert_not_called()
                else:
                    scf.ROHF.assert_called_once_with(mol)
                    scf.RHF.assert_not_called()
                self.assertEqual(scf.sfx2c1e.call_count,
                                 int(settings["reference_scf"] == "sfX2C1eROHF"))

    def test_run_pins_cache_uses_local_target_and_resumes_by_settings(self):
        args = self.args()
        args.case_id = ["CO_equi_cc_pVTZ_2"]
        case = driver.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        original_builder = lambda *a, **kw: self.fail("Attempted to rebuild the Hamiltonian")
        qa = types.SimpleNamespace(_build_hamiltonian_payload=original_builder)
        payload = {"pinned": True}
        calls = []

        def qrte(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertAlmostEqual(kwargs["delta_t"], math.pi / 23.0)
            self.assertEqual(kwargs["n_steps"], args.n_steps)
            self.assertEqual(kwargs["early_stop_energy"], reference + driver.CHEMICAL_ACCURACY)
            calls.append(kwargs)
            history = np.array([reference + 0.01, reference + 0.0008])
            return history[-1:], None, None, history

        qa.qrte_qulacs = qrte
        fake_module = types.ModuleType("QCANT")
        fake_module.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            with patch.dict(sys.modules, {"QCANT": fake_module}), \
                    patch.object(driver, "_payload_from_cache_path", return_value=payload) as loader, \
                    patch.object(driver, "compute_fock_spectrum", return_value={
                        "full_scf_orbital_energies_hartree": [-20.0, -0.2, 0.1, 3.0],
                    }) as scf:
                driver.run_case(case, args)
                loader.assert_called_once_with(Path(case["hamiltonian_cache_path"]),
                                               hamiltonian_cutoff=1e-20, qa=qa)
                summary = driver.load_json(args.outdir / case["case_id"] / "summary.json")
                self.assertEqual(summary["hit_iteration"], 2)
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["casscf_energy_hartree"], reference)
                driver.run_case(case, args)
                self.assertEqual(len(calls), 1)
                args.overlap_tol = 1e-9
                driver.run_case(case, args)
                self.assertEqual(len(calls), 2)
                self.assertEqual(scf.call_count, 1)
            self.assertIs(qa._build_hamiltonian_payload, original_builder)


if __name__ == "__main__":
    unittest.main()
