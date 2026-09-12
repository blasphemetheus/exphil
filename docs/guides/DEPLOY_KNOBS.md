# Deploy knobs — which flags in which context

Single source of truth for launch knobs (Bradley's ask, 2026-08-24,
after a netplay test silently ran at an untrained delay-id). **Rule:
copy the knob row for your context; when in doubt, copy the knob set
from the last record/crowned session's RESULTS.md — never reconstruct
from memory.** Update this table whenever a rung changes the recipe.

## Per-context knob table (ms_g15 / ms_g19 line — trained delay-ids {2,3})

| Context | Knobs | Why / source |
|---|---|---|
| **Netplay (Direct, vs human)** | `EXPHIL_NETPLAY_HOME=$HOME/.config/slippi-dolphin-bot` (bot account EXPH#288 — without it Dolphin copies YOUR login and boots to the Slippi LOGIN screen, 2026-08-28; the script now defaults to this path) + `--frame-delay 4 --delay-id-override 3 --deterministic` + `EXPHIL_QUEUE_TRACE=1` + `--replay-dir eval_runs/<name>/` | d4 = netplay latency regime; **id4 is UNTRAINED — the override to 3 is mandatory** (bare `--frame-delay 4` sets id4 and collapses chaining: 0824_stateful_netplay). Knobs of the chain-62 record (0822_netplay_crown). |
| **Local windowed play (vs human/CPU)** | `--frame-delay 3 --deterministic` (id3 implied = trained) | d3 local crown regime (ms_g15 card). 5090 harness note: laptop-trained policies need `--frame-delay 1` extra (project_5090_harness_delay_offset). |
| **Train-delay -> live flag (INVARIANTS item 12, 2026-09-12; supersedes the 09-09 row)** | `ExPhil.Eval.HarnessRung` is the law: both Dolphin runners apply a decision `--frame-delay` + 2 frames after the observed state (pinned on ep57 09-12: sync fd3/id2 chains 427/436, fd2 -> 2, fd4 -> 3/106), so live N plays reaction delay N+1 and a checkpoint trained at reaction k deploys at `--frame-delay k-1` (k >= 1). Delay-conditioned (drill) checkpoints: id = N + 1 - pipeline offset (2) = N - 1 — the deploy cards are unchanged. The Agent derives the id from the harness it is told it plays in (`harness:` sync/async/suite); `--delay-id-override` still wins but now WARNS when it disagrees. | Reaction-0 checkpoints (v16e, v3 default) have NO exact live rung (runner floor = reaction 1): `--frame-delay 0` is one slower, the 09-09 `--frame-delay 1` card was TWO slower (safe direction; one frame FAST is what breaks). Scenario suite: `--response-delay` = id + 2, auto-resolved. |
| **Any recorded/shared session (video, Discord clips)** | add `--nametag EXPH` | Bot is identified in footage and in the .slp by its in-game tag (Bradley's rule, 2026-09-02; flips memory_card to :folder so the tag applies). fox_gen v1 line plays local d0 (`--frame-delay 0`, the v1 recipe). |
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
