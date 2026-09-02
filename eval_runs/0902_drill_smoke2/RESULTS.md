# Drill bank score — eval_runs/0902_drill_smoke2

20 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 10/20.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot (this bank) | 20 | 2.6 | 20 | 16.1 | 0 |
| expert cell (uthrow/0-19%/mid) | 39 | 3.9 | 87 | 27.0 | 0 |

hits histogram: 1:2  2:14  3:2  7:1  10:1

## Smoke-run notes (2026-09-02, build step 2 complete)

- Driver: `scripts/drill_episode.exs` (single console, ~7 s/episode wall,
  4 episodes/game via edge-suicide percent resets, zero failed openers in
  20 attempts). Scorer: `scripts/drill_score.exs` (the mining detector,
  anchored at recorded handoffs; 0 anchor mismatches).
- **The headline is the histogram: 14/20 episodes stop at exactly 2 hits**
  (uthrow + one follow-up, then the confirm is dropped). That is F4's
  "live punishes cap at 1-2 hits" reproduced on demand in the drill cell —
  the state-visitation gap the 500-episode bank + AWBC retrain targets.
- The 7/10-hit outliers are laser-string episodes (each laser = a hitstun
  rising edge; the expert reference counts lasers identically, so the
  columns stay comparable, but the histogram is the honest read).
- Live-counter vs offline disagreements (10/20) are expected: the live
  bridge's hitstun stream has no gaps between linked hits, the parsed
  stream does. The offline pass is pre-registered as authoritative.
- Build lesson (added to the driver header): a HELD stick direction never
  triggers a throw — the up must EDGE into an actionable grab frame
  (07-08 press-edges lesson, grab-throw variant). 2-up/2-neutral pulse.
- Dummy DI = none (first-run knob per DRILL_HITCONFIRM.md); expert faced
  human DI, so the bot's deficit here is if anything UNDERSTATED.
