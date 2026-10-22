"""Second-order submission and driver checks without dataset calculations."""

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
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_casscf_natorb_pi_over_w_small10_trotter2.sh"
ARRAY = SCRIPT_DIR / "run_qrte_casscf_natorb_pi_over_w_small10_trotter2_array.sbatch"
OUTDIR = "runs/qrte_casscf_natorb_pi_over_w_small10_trotter2"


class SecondOrderSubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items()
                if not key.startswith("QCANT_")}

    def test_shell_syntax(self):
        for script in (SUBMIT, ARRAY):
            with self.subTest(script=script):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_array_pins_order_and_supports_slurm_spool_and_slice_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool)
            for script, steps in ((ARRAY, "1"), (spool, "3")):
                with self.subTest(script=script, steps=steps):
                    env = self.clean_env()
                    env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="4",
                               QCANT_EVOLUTION_MODE="sparse", QCANT_TROTTER_ORDER="1",
                               MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
                    if steps != "1":
                        env["QCANT_TROTTER_STEPS"] = steps
                    result = subprocess.run(["bash", str(script)], cwd=tmp, env=env,
                                            check=True, capture_output=True, text=True)
                    args = result.stdout.split()
                    expected = {"--case-index": "4", "--outdir": OUTDIR,
                                "--evolution-mode": "trotter", "--trotter-order": "2",
                                "--trotter-steps": steps, "--n-steps": "100",
                                "--overlap-tol": "1e-10"}
                    for option, value in expected.items():
                        self.assertEqual(args[args.index(option) + 1], value)

    def test_submit_records_a_separate_job_and_forwards_slurm_arguments(self):
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
  printf 'mode=%s order=%s outdir=%s\\n' "$QCANT_EVOLUTION_MODE" "$QCANT_TROTTER_ORDER" "$QCANT_OUTDIR" >&2
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
            self.assertIn(f"mode=trotter order=2 outdir={OUTDIR}", result.stderr)
            record = root / "provenance" / "latest_qrte_casscf_natorb_pi_over_w_small10_trotter2_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            self.assertEqual(len(list(record.parent.iterdir())), 1)

    def test_driver_passes_order_records_settings_and_separates_resume_profiles(self):
        with patch.object(sys, "argv", ["driver", "--trotter-order", "2"]):
            args = driver.parse_args()
        case = driver.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        result = {"sector_ground_energy_hartree": reference,
                  "sector_highest_energy_hartree": reference + 2.0,
                  "sector_dim": 36, "active_spin_occupations": [2, 2]}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=lambda *a, **kw: None)

        def qrte(**kwargs):
            self.assertEqual(kwargs["trotter_order"], args.trotter_order)
            self.assertEqual(kwargs["trotter_steps"], 1)
            self.assertEqual(kwargs["evolution_mode"], "trotter")
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
                    patch.object(driver.qrte, "_payload_from_cache_path", return_value={"pinned": True}):
                for _ in range(2):
                    driver.qrte.run_case(case, args, spectrum_builder=builder,
                                         width_builder=driver.width_from_reference,
                                         spectrum_scope=driver.SPECTRUM_SCOPE,
                                         spectrum_filename=driver.SPECTRUM_FILENAME)
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                summary = driver.qrte.load_json(args.outdir / case["case_id"] / "summary.json")
                config = driver.qrte.load_json(args.outdir / case["case_id"] / "run_config.json")
                self.assertEqual(summary["trotter_order"], 2)
                self.assertEqual(summary["run_profile"]["trotter_order"], 2)
                self.assertEqual(config["trotter_order"], 2)
                self.assertEqual(summary["hamiltonian_sha256"], case["hamiltonian_sha256"])
                args.trotter_order = 1
                driver.qrte.run_case(case, args, spectrum_builder=builder,
                                     width_builder=driver.width_from_reference,
                                     spectrum_scope=driver.SPECTRUM_SCOPE,
                                     spectrum_filename=driver.SPECTRUM_FILENAME)
                self.assertEqual(qa.qrte_qulacs.call_count, 2)
                self.assertEqual(builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
