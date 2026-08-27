# 3-arm blind local decider — ms_g19_ep4 vs ms_g20_ep13 vs ms_g15

Run 2026-08-21 14:29–14:53, local d3, FD, human port 1, 6 games,
sealed order (prereg in `run_decider.sh`). Two invalidated prior
attempts documented in `invalid_session1/` (GPU contention, GOTCHA
#100) and a crashed game-1 (warmup-race, fixed: Agent :timeout +
async_runner 120s + live warmup line).

## Card (bot port 2; scored via analyze_shine_source)

| game | arm | shines/min | max chain | Bradley blind |
|---|---|---|---|---|
| 1 | ep4 | 83.6 | **96** | "great, very cool" |
| 2 | ep13 | 35.8 | 6 | "less good, some chaining" |
| 3 | g15 | 47.9 | 4 | "like 2 but better" |
| 4 | g15 | 44.7 | 3 | "not really chaining" |
| 5 | ep13 | 36.9 | 9 | "chained a bit, prob same as 1" |
| 6 | ep4 | 86.9 | **163** | "best yet" |

## Verdict: **WIN(ep4), categorical**

- **ms_g19_ep4 chains 96 & 163 vs a human — ALL-TIME HUMAN-SESSION
  RECORDS** (prior: 46 netplay / ~22 local). 2-for-2, 4-8x the other
  arms' chains, and the blind reads independently ranked both its
  games as the session's best. The g6-inversion fear is REFUTED for
  the gate-sweep peak: dummy-record and human-record coincide here.
- g15 played to its historical human range (c3-4) — session fair.
- **ep13's stand-dummy opponent-invariance did NOT transfer** (c6/c9):
  the mewtwo gate is not a human proxy; profile trades strike again.

## Behavior pass (analyze_behavior, bot port 2)

| game | arm | SD | KO'd | stocks taken | dmg dealt | offstage% |
|---|---|---|---|---|---|---|
| 1 | ep4 | 2 | 2 | 0 | 71.6 | 4.8 |
| 6 | ep4 | 1 | 3 | 2 | 90.2 | 4.9 |
| 2 | ep13 | 2 | 2 | 1 | 82.7 | 6.5 |
| 5 | ep13 | 0 | 4 | 1 | 108.2 | 3.3 |
| 3 | g15 | **4** | 0 | 1 | 98.2 | 8.2 |
| 4 | g15 | 2 | 2 | 3 | 192.9 | 3.2 |

INTERPRETATION CAVEAT (Bradley, post-session): he plays these
sessions as PRACTICE, not to compete — deaths to the bot are often
self-imposed positioning experiments, so stk_taken/dmg columns do NOT
measure the bot's fighting ability in this rung and must not be read
as fight-state evidence (that remains the async/scenario rungs' job).
The columns stay recorded for SD-vs-KO death classification only:
ep4's SD rate (1-2/game) is unremarkable for the lineage; g15's g3
(4 SDs, 0 KOs) was its bad game.

## Status

**ms_g19_ep4 = production candidate.** Full crown awaits the netplay
rung (d4 + id-3 override, offline 388.4 c389) on the next remote
Direct session — the same path g15 was crowned by. ms_g15 remains
champion-of-record until then. Deploy: `--frame-delay 3` local;
netplay `--frame-delay 4 --delay-id-override 3` (d2 excluded — mode
never formed).
