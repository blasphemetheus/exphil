# Deploy knobs — which flags in which context

Single source of truth for launch knobs (Bradley's ask, 2026-08-24,
after a netplay test silently ran at an untrained delay-id). **Rule:
copy the knob row for your context; when in doubt, copy the knob set
from the last record/crowned session's RESULTS.md — never reconstruct
from memory.** Update this table whenever a rung changes the recipe.

## Per-context knob table (ms_g15 / ms_g19 line — trained delay-ids {2,3})

| Context | Knobs | Why / source |
|---|---|---|
| **Netplay (Direct, vs human)** | `--frame-delay 4 --delay-id-override 3 --deterministic` + `EXPHIL_QUEUE_TRACE=1` + `--replay-dir eval_runs/<name>/` | d4 = netplay latency regime; **id4 is UNTRAINED — the override to 3 is mandatory** (bare `--frame-delay 4` sets id4 and collapses chaining: 0824_stateful_netplay). Knobs of the chain-62 record (0822_netplay_crown). |
| **Local windowed play (vs human/CPU)** | `--frame-delay 3 --deterministic` (id3 implied = trained) | d3 local crown regime (ms_g15 card). 5090 harness note: laptop-trained policies need `--frame-delay 1` extra (project_5090_harness_delay_offset). |
| **Headless probes (offense-scored)** | `pace_hz: 60` (AsyncRunner) + `--stateful-step` | GOTCHA #69: both required — unpaced runs mistime inputs; windowed inference drops below every-frame acting at 3 arms/GPU. Stateful is FINE here (chains still form; relative comparisons valid) but see behavior caveat below. |
| **Sync headless eval (record-equivalent)** | `--emulation-speed 0 --blocking-input` | CLAUDE.md fast-eval recipe; 3-run determinism only on FD/BF. Sync pools CANNOT measure offense — async is the offense rung. |
| **Scenario replay (frame-locked)** | scenario_suite defaults | Input replay is pacing-immune (#69). |

## Behavior caveats that ride on these knobs

- **`--stateful-step` is NOT deploy-equivalent**: latency-identical
  (qtrace sharp) but chains ~4x weaker than windowed even at a
  trained id (0824_stateful_live canonical rescore). Default-off for
  play; mandatory for headless probes. `--stateful-resync` at a
  trained id is the untested repair.
- **Never deploy at an untrained delay-id** (standing CLAUDE.md rule);
  the trained set is a property of the CHECKPOINT's training recipe —
  for the g15/g19 line it is {2,3}.
- **Chain claims come from ShineChain over replays or the qtrace
  `act=` field (2026-08-24+) — never from B-press runs** (commanded
  inputs overcount: 212 commanded vs 23 landed, 0822).

## Measurement companions

| What | How |
|---|---|
| Latency (any live run) | `EXPHIL_QUEUE_TRACE=1`, then `mix run scripts/analyze_qtrace.exs LOG` — expect sharp peak at frame-delay+2 |
| Chains, replay-free | same analyzer — CHAINS line (needs act= in log, 2026-08-24+) |
| Chains, from replays | ShineChain v3 (`scripts/analyze_break_phases.exs`, or score both ports when the bot's netplay port is unknown) |
