#!/usr/bin/env python3
"""R3 style half — raw habit tells, read-only (no mix, no GPU).

RL_ON_PRIOR R3 has two halves. The win-rate half passed (91.0 %, with a 49.0 %
prior-vs-prior control). This scores the other half: does the PPO head's
fingerprint stay inside the human range on the identity tells, and do the
roll/spotdodge rates *fall* toward the humans rather than away?

Arms compared, all port 1:
  ppo            the trained head   (eval_runs/0923_ppo/eval_iter200)
  prior_control  the untouched prior through the SAME harness, same day
                 (eval_runs/0923_ppo/eval_control) — the matched baseline
  prior_r1       the prior's earlier sim spread (eval_runs/0921_sim_r1)
  prior_dolphin  the prior in Dolphin, the R1 reference arm
  humans         erickfm + yeti Fox games

Two criteria, both reported per tell:
  (1) R1's bound, reused: is the PPO mean within 2 sd of the prior's Dolphin arm?
  (2) the human range: is the PPO mean inside the humans' mean +/- 2 sd?
and, for roll_forward_per_min / spotdodge_per_min, whether PPO moved toward the
human mean relative to prior_control.

The learned-metric (NCA) half of the comparison needs `StyleMetric`, so it is
NOT covered here — run scripts/sim_r1_compare.exs for that once the GPU is free.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TELLS = [
    "jump_x_ratio", "short_hop_frac", "cstick_aerial_frac", "aerial_per_min",
    "roll_forward_per_min", "spotdodge_per_min",
]
CONTEXT = [
    "dashdance_per_min", "wavedash_per_min", "lightshield_frac",
    "grab_per_min", "airdodge_per_min", "lcancel_press_offset_mean",
]


def load(path, port: int | None = 1, path_contains=None):
    """Rows carry their features under 'fingerprint' (sim) or 'features' (replays)."""
    out = []
    p = REPO / path
    if not p.exists():
        return out
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if port is not None and r.get("port") != port:
            continue
        if path_contains and path_contains not in (r.get("path") or ""):
            continue
        f = r.get("fingerprint") or r.get("features")
        if isinstance(f, dict):
            out.append(f)
    return out


def stat(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var), n


def main():
    arms = {
        "ppo": load("eval_runs/0923_ppo/eval_iter200/fingerprint.jsonl"),
        "prior_control": load("eval_runs/0923_ppo/eval_control/fingerprint.jsonl"),
        "prior_r1": load("eval_runs/0921_sim_r1/anon_self_n10/sim_fingerprints.jsonl"),
        "prior_dolphin": load(
            "checkpoints/fox_v3_1_20260919_022008_ep3/style_probe/bot_fingerprints.jsonl",
            path_contains="/anon/",
        ),
        "humans": load("eval_runs/0917_style_identity/erickfm_fox.jsonl", port=None)
        + load("eval_runs/0917_style_identity/yeti_fox.jsonl", port=None),
    }

    print("R3 STYLE HALF — raw habit tells (learned-metric half not covered here)\n")
    print("games per arm: " + ", ".join(f"{k} {len(v)}" for k, v in arms.items()))
    if not arms["prior_dolphin"]:
        print("\n!! prior_dolphin arm is empty — the 2-sd reference is missing; "
              "criterion (1) cannot be scored.")

    names = ["ppo", "prior_control", "prior_r1", "prior_dolphin", "humans"]
    width = max(len(f) for f in TELLS + CONTEXT) + 2

    def table(keys, title):
        print(f"\n{title}\n" + "tell".ljust(width) + "".join(n.rjust(16) for n in names))
        for k in keys:
            row = k.ljust(width)
            for n in names:
                s = stat(arms[n], k)
                row += ("--".rjust(16) if s is None
                        else f"{s[0]:.3f}({s[1]:.3f})".rjust(16))
            print(row)

    table(TELLS, "IDENTITY TELLS (the pre-registered six)")
    table(CONTEXT, "CONTEXT (not gated)")

    # The same control logic the win-rate half used: score the UNTOUCHED prior
    # through the same harness against the same bounds. A tell the prior already
    # fails is a property of the prior or the harness, not drift PPO caused.
    print("\nVERDICT per tell")
    print("tell".ljust(width) + "ppo in human+/-2sd".rjust(20)
          + "prior in human+/-2sd".rjust(22) + "attribution".rjust(16)
          + "ppo within 2sd dolphin".rjust(24))
    passed_dolphin, passed_human, attribution = [], [], {}
    for k in TELLS:
        p = stat(arms["ppo"], k)
        c = stat(arms["prior_control"], k)
        d = stat(arms["prior_dolphin"], k)
        h = stat(arms["humans"], k)
        ok_d = ok_h = ok_c = None
        if p and d and d[1] > 0:
            ok_d = abs(p[0] - d[0]) <= 2 * d[1]
        if p and h and h[1] > 0:
            ok_h = abs(p[0] - h[0]) <= 2 * h[1]
        if c and h and h[1] > 0:
            ok_c = abs(c[0] - h[0]) <= 2 * h[1]
        passed_dolphin.append(ok_d)
        passed_human.append(ok_h)
        if ok_h is False:
            attribution[k] = "PPO drift" if ok_c else "inherited"
        fmt = lambda b: ("--" if b is None else ("PASS" if b else "OUT"))
        print(k.ljust(width) + fmt(ok_h).rjust(20) + fmt(ok_c).rjust(22)
              + attribution.get(k, "").rjust(16) + fmt(ok_d).rjust(24))

    print("\nDIRECTION on the two rates R3 asks to FALL toward the humans")
    for k in ["roll_forward_per_min", "spotdodge_per_min"]:
        p, c, h = stat(arms["ppo"], k), stat(arms["prior_control"], k), stat(arms["humans"], k)
        if not (p and c and h):
            print(f"  {k}: missing data")
            continue
        before, after = abs(c[0] - h[0]), abs(p[0] - h[0])
        verdict = "toward humans" if after < before else "away from humans"
        print(f"  {k}: prior {c[0]:.3f} -> ppo {p[0]:.3f}, humans {h[0]:.3f} "
              f"(|gap| {before:.3f} -> {after:.3f}) = {verdict}")

    scored_h = [b for b in passed_human if b is not None]
    scored_d = [b for b in passed_dolphin if b is not None]
    print(f"\nidentity tells inside the human range: {sum(scored_h)}/{len(scored_h)}")
    print(f"identity tells within 2 sd of the prior's Dolphin arm: {sum(scored_d)}/{len(scored_d)}")
    if scored_h and all(scored_h):
        print("\nSTYLE HALF (raw tells): PASSED — every identity tell is inside the human range")
    else:
        out = [k for k, b in zip(TELLS, passed_human) if b is False]
        drift = [k for k, v in attribution.items() if v == "PPO drift"]
        inherited = [k for k, v in attribution.items() if v == "inherited"]
        print(f"\nSTYLE HALF (raw tells): NOT PASSED — outside the human range on {out}")
        print(f"  caused by PPO (prior is inside, head is not): {drift or 'none'}")
        print(f"  inherited (the untouched prior is outside too): {inherited or 'none'}")
    print("\nNOTE: n=16 games per sim arm. The learned-metric comparison "
          "(scripts/sim_r1_compare.exs) is still unrun and is part of the same gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
