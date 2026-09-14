import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "multishine_readout", Path(__file__).resolve().parents[2] / "scripts/multishine_readout.py")
READOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READOUT)


class ReadoutTest(unittest.TestCase):
    def test_missing_map_does_not_become_success(self):
        with tempfile.TemporaryDirectory() as directory:
            report = READOUT.build_report({"version": 1, "checkpoints": [
                {"id": "candidate", "policy": "model.bin", "coverage": "pending.json"}]}, Path(directory))
            self.assertEqual(report["checkpoints"][0]["evidence"]["coverage"]["status"], "pending")

    def test_epoch_filter_and_duplicate_runs(self):
        content = b"ep45 [18:37:42] r1 5621 92 81 11 51.9 2\nep60 [18:38:42] r1 5623 101 97 4 62.1 6\n"
        summary = READOUT.summarize("cpu", content, {"epoch": 60}, Path("/tmp"))
        self.assertEqual(summary["rows"][0]["self_per_minute"], 62.1)
        self.assertEqual(len(summary["rows"]), 1)
        summary = READOUT.summarize("cpu", content + content, {"epoch": 60}, Path("/tmp"))
        self.assertEqual(summary["status"], "ambiguous")
        self.assertEqual(READOUT.summarize("cpu", content, {"epoch": 30}, Path("/tmp"))["status"], "no_matching_epoch")

    def test_policy_mismatch_cannot_attach_another_models_map(self):
        content = json.dumps({"policy": "other.bin"}).encode()
        summary = READOUT.summarize("coverage", content, {"policy": "candidate.bin"}, Path("/tmp"))
        self.assertEqual(summary["status"], "policy_mismatch")

    def test_partial_json_is_reported_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.json").write_text('{"policy":')
            report = READOUT.build_report({"version": 1, "checkpoints": [
                {"id": "candidate", "policy": "model.bin", "coverage": "map.json"}]}, root)
            self.assertEqual(report["checkpoints"][0]["evidence"]["coverage"]["status"], "invalid")

    def test_map_preserves_weakest_state_and_protocol(self):
        baseline = {"axis": "baseline", "mean": 0.9, "min": 0.3, "min_key": "365/3a"}
        content = json.dumps({"policy": "model.bin", "label_offset": 4, "delay_id": 4,
                              "rows": [baseline]}).encode()
        summary = READOUT.summarize("coverage", content, {"policy": "model.bin"}, Path("/tmp"))
        self.assertEqual(summary["baseline"]["min_key"], "365/3a")
        self.assertEqual(summary["label_offset"], 4)

    def test_stand_does_not_parse_argmax_selection_as_an_extra_run(self):
        content = b"ep40: 92.7/min chain 2\n=== ARGMAX: model_ep40.bin at 92.7/min\n"
        result = READOUT.summarize("stand", content, {"epoch": 40}, Path("/tmp"))
        self.assertEqual(len(result["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
