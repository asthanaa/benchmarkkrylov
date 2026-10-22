"""Active-space spectrum tests using synthetic integrals, not dataset runs."""

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_qrte_casscf_natorb_pi_over_w_small10 as driver


class CASSCFBandwidthTests(unittest.TestCase):
    def args(self):
        with patch.object(sys, "argv", ["run_qrte_casscf_natorb_pi_over_w_small10.py"]):
            return driver.parse_args()

    def synthetic_case(self, folder):
        h1 = np.diag([-1.0, 0.5])
        eri = np.zeros((2, 2, 2, 2))
        eri[0, 0, 0, 0] = 0.7
        eri[1, 1, 1, 1] = 0.3
        eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2
        cache_path = Path(folder) / "synthetic.npz"
        np.savez(cache_path, ncas=2, nelecas=[1, 1], h1ecas=h1,
                 two_mo=np.swapaxes(eri, 1, 3), ecore=7.0,
                 reference_energy=5.7, hamiltonian_source="casscf_natorb")
        return {"case_id": "synthetic", "hamiltonian_cache_path": str(cache_path),
                "hamiltonian_sha256": "synthetic-test", "active_orbitals": 2,
                "nelecas": [1, 1], "local_reference_energy_hartree": 5.7}

    def test_interacting_ci_extrema_restore_cache_integral_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self.synthetic_case(tmp)
            result = driver.compute_active_space_spectrum(case, Path(tmp), self.args())
        self.assertEqual(result["sector_dim"], 4)
        self.assertEqual(result["n_eigenvalues"], 4)
        np.testing.assert_allclose(result["sector_energies_hartree"], [5.7, 6.7, 6.7, 8.3])
        width = driver.width_from_reference(result)
        self.assertAlmostEqual(width["active_space_bandwidth_hartree"], 2.6)
        self.assertAlmostEqual(width["delta_t"], math.pi / 2.6)
        self.assertAlmostEqual(result["sector_ground_minus_cached_reference_hartree"], 0.0)

    def test_cached_reference_mismatch_prevents_qrte(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self.synthetic_case(tmp)
            case["local_reference_energy_hartree"] = -5.0
            with self.assertRaisesRegex(RuntimeError, "cached CASSCF reference"):
                driver.compute_active_space_spectrum(case, Path(tmp), self.args())

    def test_truncated_ci_matrix_is_rejected(self):
        from pyscf import fci

        with tempfile.TemporaryDirectory() as tmp:
            case = self.synthetic_case(tmp)
            with patch.object(fci.direct_spin1, "pspace", return_value=(np.arange(2), np.eye(2))):
                with self.assertRaisesRegex(RuntimeError, "complete fixed-occupation sector"):
                    driver.compute_active_space_spectrum(case, Path(tmp), self.args())

    def test_invalid_bandwidth_is_rejected(self):
        for ground, highest in ((1, 1), (2, 1), (float("nan"), 2), (1, float("inf"))):
            with self.subTest(ground=ground, highest=highest), self.assertRaises(ValueError):
                driver.width_from_reference({"sector_ground_energy_hartree": ground,
                                             "sector_highest_energy_hartree": highest})

    def test_same_ten_cases_and_qrte_defaults(self):
        args = self.args()
        cases = driver.load_cases(args)
        self.assertEqual(tuple(case["case_id"] for case in cases), driver.CASE_IDS)
        self.assertEqual(len(cases), 10)
        self.assertEqual(args.outdir, driver.DEFAULT_OUTDIR)
        self.assertEqual((args.n_steps, args.overlap_tol, args.evolution_mode, args.trotter_steps),
                         (100, 1e-10, "trotter", 1))
        self.assertNotEqual(driver.DEFAULT_OUTDIR, driver.qrte.DEFAULT_OUTDIR)

    def test_shared_qrte_driver_uses_active_spectrum_file_and_keeps_cache(self):
        args = self.args()
        case = driver.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        result = {"sector_ground_energy_hartree": reference,
                  "sector_highest_energy_hartree": reference + 2.0,
                  "sector_dim": 36, "active_spin_occupations": [2, 2]}
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=lambda *a, **kw: None)

        def qrte(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertAlmostEqual(kwargs["delta_t"], math.pi / 2.0)
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
            builder = Mock(return_value=dict(result))
            with patch.dict(sys.modules, {"QCANT": fake_qcant}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload) as loader:
                for _ in range(2):
                    driver.qrte.run_case(case, args, spectrum_builder=builder,
                                         width_builder=driver.width_from_reference,
                                         spectrum_scope=driver.SPECTRUM_SCOPE,
                                         spectrum_filename=driver.SPECTRUM_FILENAME)
                self.assertEqual(builder.call_count, 1)
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                loader.assert_called_once_with(Path(case["hamiltonian_cache_path"]),
                                               hamiltonian_cutoff=1e-20, qa=qa)
                case_dir = args.outdir / case["case_id"]
                self.assertTrue((case_dir / "active_space_reference.json").exists())
                self.assertFalse((case_dir / "fock_reference.json").exists())
                summary = driver.qrte.load_json(case_dir / "summary.json")
                self.assertEqual(summary["hit_iteration"], 2)
                self.assertEqual(summary["active_space_bandwidth_hartree"], 2.0)
                self.assertEqual(summary["run_profile"]["spectrum_scope"], driver.SPECTRUM_SCOPE)


if __name__ == "__main__":
    unittest.main()
