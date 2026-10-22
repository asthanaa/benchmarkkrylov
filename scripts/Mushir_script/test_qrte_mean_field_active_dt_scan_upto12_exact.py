"""Check the <=12-qubit exact grid, safe reuse, and Slurm dispatch without jobs."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run_qrte_mean_field_active_dt_scan_upto12_exact as scan
import run_qrte_mean_field_active_dt_scan_upto12_exact_worker as worker
import collect_qrte_mean_field_active_dt_scan4 as collector

SCRIPTS = Path(__file__).resolve().parent
STEM = "qrte_mean_field_active_dt_scan_upto12_exact"


class SmallExactScanTests(unittest.TestCase):
    def test_88_tasks_cover_eight_small_cases_and_full_range(self):
        args = scan.parse_args([])
        tasks = scan.load_tasks(args)
        self.assertEqual(len(tasks), 88)
        self.assertEqual([t["task_index"] for t in tasks], list(range(88)))
        self.assertEqual((min(scan.FACTORS), max(scan.FACTORS)), (.01, 2.))
        self.assertTrue(set(scan.base.FACTORS).issubset(scan.FACTORS))
        self.assertEqual(len(scan.CASE_IDS), 8)
        for offset, cid in enumerate(scan.CASE_IDS):
            block = tasks[offset * 11:(offset + 1) * 11]
            self.assertEqual([t["factor"] for t in block], list(scan.FACTORS))
            self.assertTrue(all(t["case"]["case_id"] == cid for t in block))
            self.assertTrue(all(2 * t["case"]["active_orbitals"] <= 12 for t in block))
            self.assertTrue(all(t["delta_t"] == t["factor"] * t["base_delta_t"] for t in block))
        with patch.object(sys, "argv", ["scan", "--count-cases"]), redirect_stdout(io.StringIO()) as output:
            scan.main()
        self.assertEqual(output.getvalue().strip(), "88")
        self.assertEqual(scan.load_tasks(scan.parse_args(["--case-index", "87"]))[0]["factor"], 2.)
        for options in (["--case-index", "88"], ["--case-id", "H7_near_equi_sto3g_1"]):
            with self.assertRaises(ValueError):
                scan.load_tasks(scan.parse_args(options))

    def test_plan_and_plots_use_new_factors_and_old_plans_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = scan.parse_args(["--outdir", tmp])
            tasks = scan.load_tasks(args)
            collector.write_plan(tasks, args)
            plan = scan.qrte.load_json(Path(tmp) / "scan_plan.json")
            self.assertEqual(plan["factors"], list(scan.FACTORS))
            self.assertEqual(plan["task_count"], 88)
            rows = collector.result_rows(tasks, args)
            import matplotlib.pyplot as plt
            with patch.object(plt, "close") as close:
                collector.plot_results(rows, args, case_ids=scan.CASE_IDS, loglog=True)
                fig = close.call_args.args[0]
                self.assertEqual(len(fig.axes), 8)
                self.assertEqual(list(fig.axes[0].get_xticks()), list(scan.FACTORS))
            plt.close(fig)
        import run_qrte_mean_field_active_dt_scan4_exact as old
        args = old.parse_args([])
        # Writing the same old plan must remain idempotent after shared changes.
        tasks = old.load_tasks(args)
        old_plan = scan.qrte.load_json(args.outdir / "scan_plan.json")
        self.assertEqual(list(dict.fromkeys(t["factor"] for t in tasks)), old_plan["factors"])

    def test_reuse_validates_history_and_profile_preserves_sources_and_skips_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = scan.parse_args(["--outdir", str(root / "destination"), "--case-index", "4"])
            task = scan.load_tasks(args)[0]
            self.assertEqual(task["factor"], .25)
            source_root = root / "source"
            source = source_root / "factor_0p25" / task["case"]["case_id"]
            original = scan.REUSE_ROOTS[0] / "factor_0p25" / task["case"]["case_id"]
            shutil.copytree(original, source)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
            with patch.object(scan, "REUSE_ROOTS", (source_root,)):
                self.assertEqual(scan.reuse_completed([task], args), 1)
                self.assertEqual(scan.reuse_completed([task], args), 0)
                destination = task["outdir"] / task["case"]["case_id"]
                audit = scan.qrte.load_json(destination / "reused_from.json")
                self.assertEqual(audit["source_directory"], str(source))
                for name, digest in audit["copied_file_sha256"].items():
                    self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
                    self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), before[name])
                self.assertEqual(collector.result_rows([task], args)[0]["status"], "completed_hit")
                bad = scan.qrte.load_json(source / "summary.json")
                bad["evolution_mode"] = "trotter"
                scan.qrte.write_json(source / "summary.json", bad)
                rejected = {**task, "outdir": root / "rejected" / "factor_0p25"}
                self.assertEqual(scan.reuse_completed([rejected], args), 0)
                self.assertFalse((rejected["outdir"] / task["case"]["case_id"]).exists())

    def test_slurm_pins_exact_mode_and_checks_88_before_submission(self):
        batch = SCRIPTS / f"run_{STEM}_array.sbatch"
        submit = SCRIPTS / f"submit_talon_{STEM}.sh"
        for path in (batch, submit):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("#SBATCH --array=0-3%4", batch.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("QCANT_")}
            env.update(PYTHON="/bin/echo", SLURM_ARRAY_TASK_ID="3", QCANT_EVOLUTION_MODE="trotter",
                       MPLCONFIGDIR=tmp, PYTHONPYCACHEPREFIX=tmp)
            options = subprocess.run(["bash", str(batch)], env=env, check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(options[0], f"scripts/Mushir_script/{worker.__name__}.py")
            self.assertEqual(options[options.index("--worker-index") + 1], "3")
            self.assertEqual(options[options.index("--worker-count") + 1], "4")
            self.assertEqual(options[options.index("--evolution-mode") + 1], "sparse")
            scripts = Path(tmp) / "scripts/Mushir_script"
            scripts.mkdir(parents=True)
            local = scripts / submit.name
            shutil.copyfile(submit, local)
            command = '''
python_stub() { printf '%s\\n' "$@" >&2; printf '%s\\n' "$TEST_COUNT"; }
sbatch() { printf '%s\\n' "$@" >&2; printf '12345;cluster\\n'; }
export -f python_stub sbatch
exec bash "$1"
'''
            env.update(PYTHON="python_stub", TEST_COUNT="88")
            result = subprocess.run(["bash", "-c", command, "test", str(local)],
                                    env=env, check=True, capture_output=True, text=True)
            self.assertIn("--array=0-3%4", result.stderr)
            self.assertIn("--prepare-only", result.stderr)
            record = Path(tmp) / f"provenance/latest_{STEM}_jobid.txt"
            self.assertEqual(record.read_text(), "12345\n")
            env["TEST_COUNT"] = "77"
            result = subprocess.run(["bash", "-c", command, "test", str(local)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--parsable", result.stderr)

    def test_workers_cover_all_conditions_once_and_preserve_task_parameters(self):
        args = scan.parse_args([])
        tasks = scan.load_tasks(args)
        for count in (1, 3, 4, 6):
            partitions = [worker.select_tasks(tasks, index, count) for index in range(count)]
            flattened = [t for part in partitions for t in part]
            self.assertEqual(sorted(t["task_index"] for t in flattened), list(range(88)))
            self.assertEqual(len({t["task_index"] for t in flattened}), 88)
        for index, count in ((0, 0), (4, 4), (-1, 4), (0, 89)):
            with self.assertRaises(ValueError):
                worker.select_tasks(tasks, index, count)
        selected = worker.select_tasks(tasks, 0, 4)
        with patch.object(worker.scan, "run_task") as run:
            worker.main(["--worker-index", "0", "--worker-count", "4"])
        self.assertEqual([call.args[0] for call in run.call_args_list], selected)


if __name__ == "__main__":
    unittest.main()
