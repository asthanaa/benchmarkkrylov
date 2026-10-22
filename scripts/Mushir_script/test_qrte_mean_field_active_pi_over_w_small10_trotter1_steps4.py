"""Saved-input and mocked execution checks; no molecular calculations."""

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

import run_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4 as driver

mf = driver.mean_field
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT = SCRIPT_DIR / "submit_talon_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4.sh"
ARRAY = SCRIPT_DIR / "run_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4_array.sbatch"


class FirstOrderFourSlicesTests(unittest.TestCase):
    def args(self, *options):
        with patch.object(sys, "argv", ["driver", *options]):
            return driver.parse_args()

    def test_defaults_and_saved_timesteps_match_previous_ten_runs(self):
        args = self.args()
        self.assertEqual((args.trotter_order, args.trotter_steps), (1, 4))
        self.assertEqual((args.n_steps, args.overlap_tol), (100, 1e-10))
        self.assertIsNone(args.delta_t)
        self.assertEqual(args.outdir, driver.DEFAULT_OUTDIR)
        cases = mf.load_cases(args)
        self.assertEqual(tuple(case["case_id"] for case in cases), mf.CASE_IDS)
        for case in cases:
            with self.subTest(case=case["case_id"]):
                width = mf.width_from_reference(mf.compute_mean_field_spectrum(case, None, args))
                previous = mf.qrte.load_json(mf.DEFAULT_OUTDIR / case["case_id"] / "summary.json")
                self.assertEqual(width["delta_t"], previous["delta_t"])
                self.assertEqual(width["mean_field_active_bandwidth_hartree"], previous["mean_field_active_bandwidth_hartree"])
                self.assertEqual(case["hamiltonian_sha256"], previous["hamiltonian_sha256"])
                self.assertEqual(case["mean_field_reference_sha256"], previous["mean_field_reference_sha256"])
                self.assertEqual(case["local_reference_energy_hartree"], previous["local_reference_energy_hartree"])

    def test_incompatible_overrides_are_rejected(self):
        for options in (["--trotter-order", "2"], ["--trotter-steps", "1"],
                        ["--trotter-steps", "8"], ["--evolution-mode", "sparse"], ["--delta-t", "0.05"]):
            with self.subTest(options=options), self.assertRaisesRegex(SystemExit, "requires dt=pi/W_MF"):
                self.args(*options)

    def test_cached_payload_full_timestep_and_distinct_resume_profile(self):
        args = self.args("--case-index", "4")
        case = mf.load_cases(args)[0]
        reference = case["local_reference_energy_hartree"]
        width = mf.width_from_reference(mf.compute_mean_field_spectrum(case, None, args))
        payload = {"pinned": True}
        qa = types.SimpleNamespace(_build_hamiltonian_payload=Mock())

        def simulate(**kwargs):
            self.assertIs(qa._build_hamiltonian_payload(), payload)
            self.assertEqual(kwargs["delta_t"], width["delta_t"])
            self.assertEqual((kwargs["trotter_order"], kwargs["trotter_steps"]), (1, 4))
            self.assertEqual((kwargs["n_steps"], kwargs["overlap_tol"]), (100, 1e-10))
            self.assertEqual(kwargs["early_stop_energy"], reference + 0.0016)
            history = np.array([reference + 0.0005])
            return history, None, None, history

        qa.qrte_qulacs = Mock(side_effect=simulate)
        fake = types.ModuleType("QCANT")
        fake.qulacs_accel = qa
        with tempfile.TemporaryDirectory() as tmp:
            args.outdir = Path(tmp)
            path = args.outdir / case["case_id"] / "summary.json"
            previous = mf.qrte.load_json(mf.DEFAULT_OUTDIR / case["case_id"] / "summary.json")
            mf.qrte.write_json(path, previous)
            with patch.dict(sys.modules, {"QCANT": fake}), \
                    patch.object(mf.qrte, "_payload_from_cache_path", return_value=payload) as build:
                def run():
                    mf.qrte.run_case(case, args, spectrum_builder=mf.compute_mean_field_spectrum,
                                     width_builder=mf.width_from_reference,
                                     spectrum_scope=mf.SPECTRUM_SCOPE, spectrum_filename=mf.SPECTRUM_FILENAME)
                run()
                run()
                self.assertEqual(qa.qrte_qulacs.call_count, 1)
                self.assertEqual(build.call_args.args[0], Path(case["hamiltonian_cache_path"]))
            summary = mf.qrte.load_json(path)
            config = mf.qrte.load_json(path.parent / "run_config.json")
            for result in (summary, config):
                self.assertEqual((result["trotter_order"], result["trotter_steps"]), (1, 4))
                self.assertEqual(result["delta_t"], previous["delta_t"])
            self.assertNotEqual(summary["run_profile"], previous["run_profile"])

    def test_main_uses_shared_runner_without_second_order_parser(self):
        with patch.object(sys, "argv", ["driver", "--dry-run"]), patch.object(mf, "run") as run:
            driver.main()
        args = run.call_args.args[0]
        self.assertEqual((args.trotter_order, args.trotter_steps), (1, 4))
        self.assertTrue(args.dry_run)


class SubmissionTests(unittest.TestCase):
    def clean_env(self):
        return {key: value for key, value in os.environ.items() if not key.startswith("QCANT_")}

    def test_batch_pins_order_and_slices_even_with_inherited_overrides(self):
        for script in (SUBMIT, ARRAY):
            subprocess.run(["bash", "-n", str(script)], check=True)
        self.assertIn("#SBATCH --time=06:00:00", ARRAY.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "slurm_script"
            shutil.copyfile(ARRAY, spool)
            env = self.clean_env()
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="7", QCANT_TROTTER_ORDER="2",
                       QCANT_TROTTER_STEPS="1", QCANT_EVOLUTION_MODE="sparse", QCANT_DELTA_T="0.05",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            result = subprocess.run(["bash", str(spool)], cwd=tmp, env=env,
                                    check=True, capture_output=True, text=True)
            options = result.stdout.split()
            self.assertEqual(options[0], f"scripts/Mushir_script/{driver.__name__}.py")
            for key, value in {"--trotter-order": "1", "--trotter-steps": "4",
                               "--case-index": "7", "--evolution-mode": "trotter",
                               "--outdir": str(driver.DEFAULT_OUTDIR.relative_to(mf.qrte.ROOT))}.items():
                self.assertEqual(options[options.index(key) + 1], value)
            self.assertNotIn("--delta-t", options)

    def test_submission_preflight_and_separate_job_record_with_stubbed_sbatch(self):
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
            env.update(PYTHON="python_stub", TEST_COUNT="10", QCANT_TROTTER_STEPS="1")
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    check=True, capture_output=True, text=True)
            self.assertIn("--array=0-9%4", result.stderr)
            self.assertIn("--time=01:00:00", result.stderr)
            self.assertIn("order: 1; slices per step: 4", result.stdout)
            record = Path(tmp) / "provenance/latest_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "9"
            result = subprocess.run(["bash", "-c", command, "test", str(submit)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)


if __name__ == "__main__":
    unittest.main()
