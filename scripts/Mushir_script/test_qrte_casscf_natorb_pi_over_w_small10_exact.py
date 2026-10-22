"""Submission checks only: no molecular calculations or real Slurm jobs."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_casscf_natorb_pi_over_w_small10_exact.sh"
ARRAY = SCRIPT_DIR / "run_qrte_casscf_natorb_pi_over_w_small10_exact_array.sbatch"
OUTDIR = "runs/qrte_casscf_natorb_pi_over_w_small10_exact"


class ExactSubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items()
                if not key.startswith("QCANT_")}

    def test_shell_syntax(self):
        for script in (SUBMIT, ARRAY):
            with self.subTest(script=script):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_array_pins_sparse_and_preserves_qrte_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="7",
                       QCANT_EVOLUTION_MODE="trotter", MPLCONFIGDIR=tmp,
                       PYTHONPYCACHEPREFIX=tmp)
            result = subprocess.run(["bash", str(ARRAY)], env=env, check=True,
                                    capture_output=True, text=True)
        args = result.stdout.split()
        expected = {"--case-index": "7", "--outdir": OUTDIR,
                    "--evolution-mode": "sparse", "--n-steps": "100",
                    "--overlap-tol": "1e-10", "--max-workers": "18",
                    "--cache-manifest": "hamiltonian_cache/correct_27_systems/manifest.csv"}
        for option, value in expected.items():
            self.assertEqual(args[args.index(option) + 1], value)
        self.assertEqual(args[0], "scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py")

    def test_array_runs_from_a_slurm_spool_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool_script = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool_script)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="7",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            result = subprocess.run(["bash", str(spool_script)], cwd=tmp,
                                    env=env, check=True, capture_output=True, text=True)
        self.assertIn("scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py", result.stdout)
        self.assertIn("--case-index 7", result.stdout)
        self.assertIn("--evolution-mode sparse", result.stdout)
        self.assertIn(f"--outdir {OUTDIR}", result.stdout)

    def test_submit_uses_exact_array_and_separate_provenance(self):
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
  printf 'mode=%s outdir=%s\\n' "$QCANT_EVOLUTION_MODE" "$QCANT_OUTDIR" >&2
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
            self.assertIn(f"mode=sparse outdir={OUTDIR}", result.stderr)
            self.assertIn(f"Output: {OUTDIR}/<case_id>/", result.stdout)
            record = root / "provenance" / "latest_qrte_casscf_natorb_pi_over_w_small10_exact_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            self.assertEqual(len(list(record.parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
