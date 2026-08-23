# 0824_crown_decider — ep4 CROWNED (blind stage-pinned A/B vs DBTD#411)

**Verdict: `ms_g19_ep4.bin` is CHAMPION** (Bradley's call on the
unblinded result). `ms_g15_oppmask_full.bin` becomes the fallback.
The stage-pinned decider owed since 08-22 (the 08-22 attempt was
stage-confounded; the decider law "PIN stages" comes from it).

## Protocol

Blind A/B: checkpoints copied to neutral `arm_red.bin`/`arm_blue.bin`
(random assignment; key file hidden until scoring), shuffled game
order, Bradley records per-game impressions by color only. Record
knobs both arms: windowed (deploy rung — NOT stateful), d4 +
`--delay-id-override 3`, deterministic, `--require-stage
final_destination`, EXPHIL_QUEUE_TRACE=1, bot-side replays kept.
Chains scored canonically (ShineChain v3 over .slp, both ports — bot
port flips by connect order; it landed p1 in the blue session, p2 in
red).

## Scores (canonical, bot's port)

| game | arm | checkpoint | shines/min | max chain | sustained >=2 | blind impression |
|---|---|---|---|---|---|---|
| 1 FD | blue | g15 | 40.9 | 3 | 5 | "not great" |
| 2 FD | blue | g15 | 53.0 | 3 | 4 | "not great" |
| 3 FD | blue | g15 | 80.0 | 13 | 13 | "alright, inconsistent chains" |
| 4 FD | red | **ep4** | 57.9 | **32** | 11 | "significantly better chains" |
| 5 FD | red | **ep4** | 60.3 | **12** | 11 | ↑ |

ep4's worse game matches g15's best; blind human judgment and replay
truth agree. Caveats (recorded, not crown-blocking): n=5 (3/2),
FD-only (the planned PS pair never ran), single day/opponent. Stacks
on ep4's prior record: all-time netplay chain 62 (0822), stand d3
437.4/min c438, human-session chain records 96/163.

## The session's other yield: five require_stage/LRAS bugs (all committed)

The `--require-stage` path had never been exercised in earnest; eight
launch attempts flushed five distinct bugs, three diagnosed live by
Bradley watching the screen:

1. `cca69fc` — stage RAM merge starved the netplay frame loop
   (per-frame bounded watcher snapshot) → local-only gate.
2. `de3611e` — LRAS chord churned by per-frame release_all: sub-frame
   pipe gaps let pad samples see PARTIAL chords (shield → pause).
3. `5602f56` — online watch set dropped stage lines (volatile-heap
   on-change churn; frame count froze at ~135 every incident) +
   memory_watch_set/force_quit_ops became pure, unit-tested fns.
4. `8cc8e8a` — pause FREEZES the spectator stream, killing any
   frame-driven pulse → LRAS moved to a wall-clock GenServer timer
   (300ms alternating Start edges, chord held).
5. `6dc60af` + `a8b5d76` — held chord leaked into the accepted game
   (ground on clear), and require_stage compared EXTERNAL ids against
   the now-INTERNAL gamestate.stage — FD was rejecting itself
   (GOTCHA #96 class; pinned against the events.ex conversion for all
   six legal stages).

Deploy card update: netplay = d4 + `--delay-id-override 3` (id4 is
untrained); local = d3. See docs/guides/DEPLOY_KNOBS.md.
