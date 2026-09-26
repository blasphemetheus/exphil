import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/watch_cuda_diagnostic.py"
spec = importlib.util.spec_from_file_location("cuda_watcher", SCRIPT)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class CudaWatcherTest(unittest.TestCase):
    completed = {"steps": 1879, "chunks": 2, "mode": "parallel", "start": 12}
    clean_log = "Completed 1879 steps\n========= ERROR SUMMARY: 0 errors\n"

    def test_pass_requires_both_completion_and_final_sanitizer_summary(self):
        self.assertTrue(watcher.classify(self.clean_log, self.completed)[0].startswith("PASSED"))
        for log, completed in [(self.clean_log, None), ("Completed 1879 steps", self.completed),
                               (self.clean_log, {**self.completed, "steps": 100})]:
            self.assertTrue(watcher.classify(log, completed)[0].startswith("INCONCLUSIVE"))

    def test_disabled_checker_cannot_pass(self):
        verdict, _ = watcher.classify("Sanitizer will be disabled\n" + self.clean_log, self.completed)
        self.assertTrue(verdict.startswith("INCONCLUSIVE"))

    def test_memory_error_or_fatal_overrides_successful_training(self):
        for log in [self.clean_log + "ERROR SUMMARY: 2 errors", self.clean_log + "[FATAL]"]:
            self.assertTrue(watcher.classify(log, self.completed)[0].startswith("FAILED"))

    def test_publication_does_not_overwrite_an_existing_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "CUDA.md"
            target.write_text("another agent's report")
            with patch.multiple(watcher, ROOT=root, TARGET=target, LOG=root / "missing.log", ARTIFACTS=root):
                with self.assertRaises(FileExistsError):
                    watcher.publish()
            self.assertEqual(target.read_text(), "another agent's report")
            self.assertEqual(list(root.glob(".CUDA.md.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
