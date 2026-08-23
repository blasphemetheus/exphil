# 0824_stateful_live — live qtrace validation of --stateful-step (JIT lever 2b)

The deploy-rung gate from JIT_WARMUP.md option 2b: one live session on
the stateful step path with a qtrace lag check, paired with a windowed
control run under identical settings the same night.

## Setup (both arms identical except the flag)

ms_g19_ep4.bin, local windowed Dolphin (netplay-beta-nixos), fox vs
CPU-3 fox, FD, `--frame-delay 3 --deterministic`, EXPHIL_QUEUE_TRACE=1,
`--seconds 150 --on-game-end stop`, slippi-port 51442 (51441 was held
by a stray launcher Dolphin from earlier tonight).

## Results

| metric | stateful (live.log) | windowed (windowed_control.log) |
|---|---|---|
| JIT warmup | **1,482ms** | 19,973ms |
| frames | 9,192 @ 60.0 fps | 9,153 @ 60.0 fps |
| staleness | 1/9192 (0.0%) | 7/9153 (0.1%) |
| qtrace peak | lag 5 @ **99.9%** | lag 5 @ 99.6% |
| verdict | sharp (nominal d3+2) | sharp (nominal d3+2) |

- Decision vs applied B-runs match in both arms (queue applies
  faithfully); no trace of the exec-cache hang class (counter frozen /
  latched inputs) at any point.
- Stateful is marginally SHARPER than windowed (higher peak, lower
  off-peak correlation).
- `Conf: 0.0 (avg ~0.05)` in the stats line appears in BOTH arms —
  pre-existing display quirk, not a stateful artifact; not chased.
- Session also exercised the same-night warmup fix (fused sampler now
  warmed on the stateful branch — heads_sample stage 503ms live).

## Verdict

Gate PASSED: the stateful step path has nominal decision→apply latency
live, at 13x less warmup. Recommend `--stateful-step` in local deploy
recipes now; netplay-vs-human remains the final rung before flipping
the script default (deploy-rung rule).
