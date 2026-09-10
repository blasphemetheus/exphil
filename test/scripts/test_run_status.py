import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
STATUS_TOOL = ROOT / "scripts/run_status.py"
WATCHDOG = ROOT / "scripts/train_watchdog.sh"


class RunStatusTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.status = self.directory / "status.json"
        self.environment = dict(os.environ)
        self.environment.pop("EXPHIL_RUN_ID", None)
        self.environment.pop("EXPHIL_RUN_STATUS", None)

    def tool(self, operation, *arguments, **kwargs):
        return subprocess.run(
            [sys.executable, str(STATUS_TOOL), operation, "--status", str(self.status),
             "--run-id", "test-run", *arguments],
            env=self.environment, capture_output=True, text=True, timeout=10, **kwargs
        )

    def launch(self, code):
        return self.tool("run", "--", sys.executable, "-c", code)

    def record(self):
        return json.loads(self.status.read_text())

    def watchdog(self, pid, *, identified=True):
        binaries = self.directory / "bin"
        binaries.mkdir(exist_ok=True)
        for name in ("nvidia-smi", "notify-send"):
            stub = binaries / name
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
        log = self.directory / "run.log"
        log.write_text("Policy exported\nConverged\n")
        checkpoint = self.directory / "old.bin"
        checkpoint.write_bytes(b"intermediate checkpoint")
        command = ["bash", str(WATCHDOG), "--pid", str(pid), "--log", str(log),
                   "--checkpoint", str(checkpoint)]
        if identified:
            command.extend(["--status", str(self.status), "--run-id", "test-run"])
        return subprocess.run(
            command, env=dict(self.environment, PATH=str(binaries) + os.pathsep + os.environ["PATH"]),
            capture_output=True, text=True, timeout=5
        )

    def test_success_and_matching_watchdog(self):
        result = self.launch("pass")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = self.record()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["child_exit_code"], 0)
        self.assertIsNotNone(record["ended_at"])
        self.assertEqual(self.tool("check").returncode, 0)
        self.assertEqual(self.watchdog(record["launcher_pid"]).returncode, 0)

    def test_crash_after_checkpoint_is_failure(self):
        result = self.launch("raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.record()["status"], "failed")
        self.assertEqual(self.watchdog(self.record()["launcher_pid"]).returncode, 1)

    def test_checkpoint_and_export_log_alone_are_unknown(self):
        self.launch("pass")
        result = self.watchdog(self.record()["launcher_pid"], identified=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("UNKNOWN", result.stdout)

    def test_stale_pid_and_run_identity_are_rejected(self):
        self.launch("pass")
        self.assertEqual(self.tool("check", "--pid", "2147483647").returncode, 2)
        self.assertEqual(self.tool("check", "--run-id", "another-run").returncode, 2)

    def test_existing_status_is_not_overwritten(self):
        self.launch("pass")
        original = self.status.read_bytes()
        self.assertEqual(self.launch("raise SystemExit(9)").returncode, 2)
        self.assertEqual(self.status.read_bytes(), original)

    def test_missing_truncated_and_nonterminal_records_are_unknown(self):
        self.assertEqual(self.tool("check").returncode, 2)
        self.status.write_text('{"schema_version":')
        self.assertEqual(self.tool("check").returncode, 2)
        self.status.unlink()
        self.launch("pass")
        record = self.record()
        record.update(status="running", child_exit_code=None, ended_at=None)
        self.status.write_text(json.dumps(record))
        self.assertEqual(self.watchdog(record["launcher_pid"]).returncode, 2)

    def child_update(self, *options):
        arguments = [sys.executable, str(STATUS_TOOL), "update", *options]
        return f"import subprocess; subprocess.run({arguments!r}, check=True)"

    def test_progress_and_intentional_early_stop(self):
        result = self.launch(self.child_update("--epoch", "3", "--artifact", "best.bin",
                                                "--diagnostics", "passed", "--early-stopped"))
        self.assertEqual(result.returncode, 0, result.stderr)
        record = self.record()
        self.assertEqual(record["status"], "early_stopped")
        self.assertEqual(record["last_successful_epoch"], 3)
        self.assertEqual(record["artifacts"], ["best.bin"])
        self.assertEqual(self.tool("check").returncode, 0)

    def test_diagnostics_failure_is_not_success(self):
        result = self.launch(self.child_update("--diagnostics", "failed"))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.record()["child_exit_code"], 0)
        self.assertEqual(self.record()["status"], "diagnostics_failed")
        self.assertEqual(self.tool("check").returncode, 1)

    def test_early_stop_does_not_hide_child_failure(self):
        result = self.launch(self.child_update("--early-stopped") + "; raise SystemExit(8)")
        self.assertEqual(result.returncode, 8)
        self.assertEqual(self.record()["status"], "failed")

    def test_finished_run_cannot_be_updated(self):
        self.launch("pass")
        self.assertEqual(self.tool("update", "--epoch", "4").returncode, 2)

    def test_invalid_terminal_records_are_unknown(self):
        self.launch("pass")
        original = self.record()
        for change in ({"status": []}, {"diagnostics": []}, {"ended_at": 12},
                       {"ended_at": "not-a-time"}, {"child_exit_code": True},
                       {"child_exit_code": 7}, {"diagnostics": "failed"}):
            with self.subTest(change=change):
                self.status.write_text(json.dumps(dict(original, **change)))
                result = self.tool("check")
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_progress_cannot_decrease(self):
        code = self.child_update("--epoch", "3") + "; " + self.child_update("--epoch", "2")
        self.assertNotEqual(self.launch(code).returncode, 0)
        self.assertEqual(self.record()["last_successful_epoch"], 3)

    def test_diagnostic_failure_cannot_be_erased(self):
        code = self.child_update("--diagnostics", "failed") + "; " + self.child_update("--diagnostics", "passed")
        self.assertNotEqual(self.launch(code).returncode, 0)
        self.assertEqual(self.record()["diagnostics"], "failed")

    def test_killed_launcher_leaves_unknown_status(self):
        child = subprocess.Popen(
            [sys.executable, str(STATUS_TOOL), "run", "--status", str(self.status),
             "--run-id", "test-run", "--", sys.executable, "-c", "import time; time.sleep(10)"],
            env=self.environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        worker_pid = None
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.status.exists():
                    worker_pid = self.record().get("child_pid")
                    if worker_pid:
                        break
                time.sleep(0.01)
            else:
                self.fail("launcher did not start child")
            child.kill()
            child.wait(timeout=5)
            self.assertEqual(self.watchdog(child.pid).returncode, 2)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            if worker_pid:
                try:
                    os.killpg(worker_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def test_missing_command_records_launch_failure(self):
        result = self.tool("run", "--", str(self.directory / "nonexistent-command"))
        self.assertEqual(result.returncode, 127)
        self.assertEqual(self.record()["status"], "failed")

    def test_signal_records_interruption(self):
        result = self.launch("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
        self.assertEqual(result.returncode, 128 + signal.SIGTERM)
        self.assertEqual(self.record()["status"], "interrupted")

    def test_wrapper_forwards_termination(self):
        child = subprocess.Popen(
            [sys.executable, str(STATUS_TOOL), "run", "--status", str(self.status),
             "--run-id", "test-run", "--", sys.executable, "-c", "import time; time.sleep(10)"],
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.status.exists() and self.record().get("child_pid"):
                    break
                time.sleep(0.01)
            else:
                self.fail("launcher did not start child")
            child.terminate()
            child.communicate(timeout=5)
            self.assertEqual(child.returncode, 128 + signal.SIGTERM)
            self.assertEqual(self.record()["status"], "interrupted")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
