#!/usr/bin/env python3
"""Compare PPO fingerprints with a same-harness prior control (CPU only).

Retrospective drift alerts, not a human-likeness or strength verdict. Thresholds
encode DEGENERATE_ZOO Entry 1; validate them on future runs before treating them
as a general classifier. Exit 0 normally; --fail-on-alert returns 2 on alerts
and 3 when evidence is insufficient. Never starts/stops a training process.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

FEATURES = ('roll_backward_per_min', 'roll_forward_per_min', 'grab_per_min',
            'throw_back_mix', 'ledge_roll_mix', 'ledge_getup_mix', 'aerial_per_min')
RULE = dict(roll_ratio=4.0, grab_ratio=3.0, back_throw_ratio=3.0,
            aerial_collapse_ratio=0.3, aerial_warning_ratio=0.5,
            baseline_aerial_floor=1.0, minimum_games=8)


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'fingerprint.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return path, rows


def summarize(rows):
    means, counts = {}, {}
    for key in FEATURES:
        values = [(r.get('fingerprint') or r.get('features') or {}).get(key) for r in rows]
        values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        means[key] = statistics.mean(values) if values else None
        counts[key] = len(values)
    return dict(games=len(rows), characters=sorted({str(r.get('character', 'unknown')).lower() for r in rows}),
                means=means, counts=counts)


def compare(baseline, candidate):
    base, arm = summarize(baseline), summarize(candidate)
    reasons = []
    if base['characters'] != arm['characters'] or len(base['characters']) != 1 or 'unknown' in base['characters']:
        reasons.append('character mismatch, missing character, or mixed characters')
    for label, stats in [('baseline', base), ('candidate', arm)]:
        if any(n < RULE['minimum_games'] for n in stats['counts'].values()):
            reasons.append(f'{label}: fewer than {RULE["minimum_games"]} finite game values for at least one required feature')
    if reasons:
        return dict(status='insufficient_evidence', reasons=reasons, baseline=base, candidate=arm, alerts=[])
    b, a = base['means'], arm['means']
    ratios = {k: a[k] / b[k] if b[k] > 0 else None for k in FEATURES}
    rolls_b = b['roll_backward_per_min'] + b['roll_forward_per_min']
    rolls_a = a['roll_backward_per_min'] + a['roll_forward_per_min']
    ratios['total_rolls_per_min'] = rolls_a / rolls_b if rolls_b > 0 else None
    above = lambda key, threshold: ratios[key] is not None and ratios[key] > threshold
    aerial_supported = b['aerial_per_min'] >= RULE['baseline_aerial_floor']
    ledge_appeared = any(b[k] == 0 and a[k] > 0 for k in ('ledge_roll_mix', 'ledge_getup_mix'))
    signature = (above('total_rolls_per_min', RULE['roll_ratio'])
                 and above('grab_per_min', RULE['grab_ratio'])
                 and above('throw_back_mix', RULE['back_throw_ratio'])
                 and ledge_appeared and aerial_supported
                 and ratios['aerial_per_min'] < RULE['aerial_collapse_ratio'])
    alerts = []
    if signature:
        alerts.append('zoo_1_roll_grab_backthrow_signature')
    if aerial_supported and ratios['aerial_per_min'] < RULE['aerial_warning_ratio']:
        alerts.append('aerial_rate_reduced_by_more_than_half_review_required')
    return dict(status='review_required' if alerts else 'no_known_signature_detected', alerts=alerts,
                baseline=base, candidate=arm, ratios=ratios, ledge_behavior_appeared=ledge_appeared,
                caveat='A drift alert is not proof of degeneracy. No alert is not a strength/style pass. Ratios of sparse event mixtures are noisy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fail-on-alert', action='store_true')
    args = parser.parse_args()
    base_path, base = load(args.baseline)
    reports = []
    for candidate in args.candidate:
        path, rows = load(candidate)
        result = compare(base, rows)
        result.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reports.append(result)
        print(f'{path.parent.name}: {result["status"]}; {", ".join(result["alerts"] or result.get("reasons", [])) or "no alerts"}')
    report = dict(schema_version=1, baseline_path=str(base_path),
        baseline_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        rule=RULE, rule_provenance='retrospective: DEGENERATE_ZOO.md Entry 1, 2026-09-25',
        comparison_requirement='Caller must use the same evaluator, opponent, stage, temperature, capture window and port protocol. Different seeds are allowed. These settings are not all recoverable from historical JSONL.',
        candidates=reports)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.fail_on_alert:
        if any(r['status'] == 'insufficient_evidence' for r in reports):
            raise SystemExit(3)
        if any(r['alerts'] for r in reports):
            raise SystemExit(2)


if __name__ == '__main__':
    main()
