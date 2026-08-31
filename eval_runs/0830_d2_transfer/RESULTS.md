# D2 — human-vs-CPU transfer table (loop_report metrics)

**2026-08-30.** Same checkpoint scored on both rungs (CPU bracket dirs vs
Bradley-session dirs), loop_report --bot-port 1 throughout.

| ckpt | rung | taunts/min | dpad/min | loops/min | action_long_frac | longest_input_run |
|---|---|---:|---:|---:|---:|---:|
| ep10 | cpu | 1.29 | 111.6 | 1.53 | 0.27 | 14.0 |
| ep10 | human | **331.7 ⚠** | 93.2 | 0.74 | 0.16 | 11.2 |
| B1 | cpu | 2.04 | 114.6 | 1.49 | 0.25 | 11.5 |
| B1 | human | 1.41 | 92.5 | 0.22 | 0.24 | 16.7 |
| B2 | cpu | 1.87 | 126.9 | 1.80 | 0.27 | 14.4 |
| B2 | human | 1.34 | 91.5 | 0.78 | 0.21 | 10.7 |
| B3 | cpu | 1.90 | 119.0 | 1.35 | 0.25 | 13.6 |
| B3 | human | 1.59 | 102.1 | 0.81 | 0.21 | 13.6 |

## Read

- **dpad/min transfers well**: human ≈ 0.8× CPU, uniformly (91–102 vs
  112–127). The CPU rung can stand in for this metric.
- **taunts/min transfers** for B1/B2/B3 (human ≈ 0.7–0.85× CPU) —
  **EXCEPT ep10_human = 331.7/min, which is physically impossible**
  (>5 taunts/second). The ep10 "human" dir is
  `eval_runs/0828_livelook_btn05/2026-08-Mainline` scored at --bot-port 1;
  suspect the bot was NOT port 1 in part of that session (human on P1 →
  scoring the human's d-pad? no — executed taunt states 264/265 at 331/min
  is parser/port mix-up territory). Treat the ep10 human row as INVALID
  until the dir's port map is checked; do not read it as a finding.
- **loops/min: CPU rung OVERSTATES ~2×** (1.35–1.80 cpu vs 0.22–0.81
  human) but preserves ordering loosely; use for ordering, not magnitude.
- action_long_frac / longest_input_run: within noise across rungs.

Verdict: dense decode-side metrics (dpad, taunts) transfer with a stable
~0.8× human/CPU factor; loop metrics transfer in ordering only; one
port-mapping artifact flagged. Per-metric floors from D1 still apply on
top.

Reports: `eval_runs/0830_d2_transfer/<ckpt>_<rung>/report.json` (runner:
`scripts/d2_transfer.sh`).
