"""Fixed-timestep checks without molecular calculations or real submissions."""

import contextlib
import io
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

import run_qrte_casscf_natorb_pi_over_w_small10 as driver


SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_casscf_natorb_small10_trotter2_dt0p05.sh"
ARRAY = SCRIPT_DIR / "run_qrte_casscf_natorb_small10_trotter2_dt0p05_array.sbatch"
OUTDIR = "runs/qrte_casscf_natorb_small10_trotter2_dt0p05"


class FixedTimestepTests(unittest.TestCase):
    def args(self, *options):
        with patch.object(sys, "argv", ["driver", *options]):
            return driver.parse_args()

    def clean_env(self):
        return {key: value for key, value in os.environ.items()
                if not key.startswith("QCANT_")}

    def test_fixed_timestep_is_opt_in_and_invalid_values_are_rejected(self):
        self.assertIsNone(self.args().delta_t)
        self.assertEqual(self.args("--delta-t", "0.05").delta_t, 0.05)
        for value in ("0", "-0.05", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.args("--delta-t", value)
                self.assertEqual(error.exception.code, 2)

    def test_shell_syntax(self):
        for script in (SUBMIT, ARRAY):
            with self.subTest(script=script):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_array_pins_dt_and_order_from_original_and_spool_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool)
            for script in (ARRAY, spool):
                with self.subTest(script=script):
                    env = self.clean_env()
                    env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="4",
                               QCANT_EVOLUTION_MODE="sparse", QCANT_TROTTER_ORDER="1",
                               QCANT_DELTA_T="0.9", MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
                    result = subprocess.run(["bash", str(script)], cwd=tmp, env=env,
                                            check=True, capture_output=True, text=True)
                    args = result.stdout.split()
                    expected = {"--case-index": "4", "--outdir": OUTDIR,
                                "--evolution-mode": "trotter", "--trotter-order": "2",
                                "--trotter-steps": "1", "--delta-t": "0.05",
                                "--n-steps": "100", "--overlap-tol": "1e-10"}
                    for option, value in expected.items():
                        self.assertEqual(args[args.index(option) + 1], value)

    def test_submit_uses_fixed_dt_array_and_separate_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts" / "Mushir_script"
            scripts.mkdir(parents=True)
            submit = scripts / SUBMIT.name
            shutil.copyfile(SUBMIT, submit)
            env = self.clean_env()
            env["PYTHON"] = "python_stub"
            command = '''
python_stub() { printf '10\\n'; }
sbatch() {
  printf '%s\\n' "$@" >&2
  printf 'mode=%s order=%s dt=%s outdir=%s\\n' "$QCANT_EVOLUTION_MODE" "$QCANT_TROTTER_ORDER" "$QCANT_DELTA_T" "$QCANT_OUTDIR" >&2
  printf '12345;cluster\\n'
}
export -f python_stub sbatch
exec bash "$1" --time=01:00:00
'''
            result = subprocess.run(["bash", "-c", command, "test", str(submit)],
                                    env=env, check=True, capture_output=True, text=True)
            self.assertIn("--array=0-9%4", result.stderr)
            self.assertIn("--time=01:00:00", result.stderr)
            self.assertIn(ARRAY.name, result.stderr)
            self.assertIn(f"mode=trotter order=2 dt=0.05 outdir={OUTDIR}", result.stderr)
            record = root / "provenance" / "latest_qrte_casscf_natorb_small10_trotter2_dt0p05_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            self.assertEqual(len(list(record.parent.iterdir())), 1)

    def test_driver_overrides_dt_keeps_reference_and_separates_resume_profiles(self):
        args = self.args("--trotter-order", "2", "--delta-t", "0.05")
        case = driver.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        result = {"sector_ground_energy_hartree": reference,
                  "sector_highest_energy_hartree": reference + 2.0,
                  "sector_dim": 36, "active_spin_occupations": [2, 2]}
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=lambda *a, **kw: None)

        def qrte(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["delta_t"], args.delta_t if args.delta_t is not None else math.pi / 2)
            self.assertEqual((kwargs["trotter_order"], kwargs["trotter_steps"], kwargs["n_steps"]), (2, 1, 100))
            self.assertEqual(kwargs["early_stop_energy"], reference + driver.qrte.CHEMICAL_ACCURACY)
            history = np.array([reference + 0.0005])
            return history, None, None, history

        qa.qrte_qulacs = Mock(side_effect=qrte)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            builder = Mock(return_value=dict(result))
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value=payload):
                def run():
                    driver.qrte.run_case(case, args, spectrum_builder=builder,
                                         width_builder=driver.width_from_reference,
                                         spectrum_scope=driver.SPECTRUM_SCOPE,
                                         spectrum_filename=driver.SPECTRUM_FILENAME)

                run()
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                for filename in ("summary.json", "run_config.json"):
                    saved = driver.qrte.load_json(args.outdir / case["case_id"] / filename)
                    self.assertEqual(saved["delta_t"], 0.05)
                    self.assertEqual(saved["delta_t_rule"], "fixed timestep (independent of spectral width)")
                    self.assertAlmostEqual(saved["bandwidth_delta_t"], math.pi / 2)
                    self.assertEqual(saved["active_space_bandwidth_hartree"], 2.0)
                summary = driver.qrte.load_json(args.outdir / case["case_id"] / "summary.json")
                self.assertEqual(summary["run_profile"]["fixed_delta_t"], 0.05)
                self.assertEqual(summary["hamiltonian_sha256"], case["hamiltonian_sha256"])
                args.delta_t = 0.1
                run()
                args.delta_t = None
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 3)
                self.assertEqual(builder.call_count, 1)
                summary = driver.qrte.load_json(args.outdir / case["case_id"] / "summary.json")
                self.assertNotIn("fixed_delta_t", summary["run_profile"])
                self.assertNotIn("bandwidth_delta_t", summary)
                self.assertEqual(summary["delta_t"], math.pi / 2)


if __name__ == "__main__":
    unittest.main()
