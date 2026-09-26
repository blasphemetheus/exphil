import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('degeneracy', Path(__file__).resolve().parents[2] / 'scripts/degeneracy_check.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def rows(aerial=10, roll=1, grab=2, back=0.05, ledge=0, character='Mewtwo'):
    values = dict(roll_backward_per_min=roll, roll_forward_per_min=roll, grab_per_min=grab,
                  throw_back_mix=back, ledge_roll_mix=ledge, ledge_getup_mix=0, aerial_per_min=aerial)
    return [dict(character=character, fingerprint=dict(values)) for _ in range(8)]


class DriftChecks(unittest.TestCase):
    def test_compound_signature_and_aerial_only_warning_differ(self):
        result = check.compare(rows(), rows(aerial=0.8, roll=12, grab=8, back=0.25, ledge=0.2))
        self.assertIn('zoo_1_roll_grab_backthrow_signature', result['alerts'])
        reduced = check.compare(rows(), rows(aerial=2))
        self.assertEqual(len(reduced['alerts']), 1)
        self.assertNotIn('zoo_1_roll_grab_backthrow_signature', reduced['alerts'])

    def test_unchanged_policy_has_no_alert(self):
        self.assertEqual(check.compare(rows(), rows())['status'], 'no_known_signature_detected')

    def test_missing_nan_and_wrong_character_are_not_passes(self):
        for bad in [rows()[:2], rows(character='Fox'), rows(aerial=float('nan'))]:
            self.assertEqual(check.compare(rows(), bad)['status'], 'insufficient_evidence')

    def test_zero_baseline_does_not_make_infinite_ratios(self):
        result = check.compare(rows(aerial=0, back=0), rows(aerial=0, back=1))
        self.assertIsNone(result['ratios']['aerial_per_min'])
        self.assertIsNone(result['ratios']['throw_back_mix'])
        self.assertEqual(result['alerts'], [])

    def test_features_schema_supported(self):
        other = [dict(character=r['character'], features=r['fingerprint']) for r in rows()]
        self.assertEqual(check.compare(rows(), other)['status'], 'no_known_signature_detected')


if __name__ == '__main__':
    unittest.main()
