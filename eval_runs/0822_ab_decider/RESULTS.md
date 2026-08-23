# Blind A/B netplay decider: ms_g19_ep4 vs ms_g15_oppmask_full (2026-08-22 evening)

**VERDICT: INCONCLUSIVE — crown stays OPEN** (Bradley's final read
post-unblind; his initial "crown ep4 probably" was revised on the
stage analysis). ep4 holds the all-time netplay record (62, same
night) and edged chains here ({16,6,2} vs {11,4,3}), but n=3/arm
with an UNCONTROLLED stage draw (no FD appeared; YS went to ep4)
cannot seal it. ms_g15 remains champion-of-record. **Next decider:
STAGE-PINNED — each arm plays FD plus one agreed second stage,
balanced; no random Direct draw.**

## Protocol

6 games, 3 per arm, order sealed pre-session (sealed_order.txt,
shuffled 3/3), Bradley (DBTD#411) blind; one bot launch per game via
launch_game.sh (arm read from the sealed file — no arm info on
screen; GOTCHA #100 GPU guard re-checked every launch). Each arm at
its own validated config: g15 = crowned 0809 config (--frame-delay 4),
ep4 = 08-22 validated (--frame-delay 4 --delay-id-override 3). Blind
per-game notes collected before unsealing (blind_notes.txt). Two
Bradley-early-START stub games excluded (20:00:47 FD w/ g1's still-live
bot; 20:18:51 FD pre-g6). Scored from Bradley-side replays
(analyze_shine_source); qtrace all games.

## Results (replays = ground truth)

| game | arm | stage | frames | self shines/min | max chain | blind note |
|---|---|---|---|---|---|---|
| 1 19:57 | g15 | FoD | 7,489 | 63.9 | **11** | "pretty good, ~8 chain" |
| 2 20:01 | g15 | PS | 9,057 | 54.5 | 4 | "not as good as 1" |
| 3 20:06 | ep4 | YS | 6,147 | 29.9 | 2 | "same as 2" |
| 4 20:12 | ep4 | BF | 6,817 | 60.7 | **16** | "the good one! same as 1" |
| 5 20:16 | ep4 | DL | 6,497 | 42.1 | 6 | "same as 2/3" |
| 6 20:20 | g15 | FoD | 7,073 | 50.9 | 3 | "not that good; texture like 1/4" |

qtrace: ALL SIX sharp lag-6 (96.9-99.7%) — uniform regime, both arms,
no contention (idle-GPU guard held).

## Chain comparison (favors ep4, not decisively)

- Chains {16,6,2} vs {11,4,3}: decider max + 2-of-3 rank-paired, PLUS
  the same-night single-arm **chain 62 on FD = all-time netplay
  record** (0822_netplay_crown addendum; same rung, same opponent).
- Blind perception found NO g15 advantage: Bradley's two "good" games
  were the two high-chain games, one per arm — no dummy-inversion
  signature on ep4, no hidden human-facing flaw either direction.
- Rates: g15 steadier (51-64 vs 30-61) — consistent with the chains =
  transient-peaks law; the crown metric is chains.

## Stage analysis (Bradley's question)

Stage is real but partial: the FoD repeat (g1/g6, SAME arm) split
11 vs 3 — within-stage within-arm spread as large as between-stage;
temporal variance dominates. But the floor (chain 2) landed on YS —
the standing caveat stage (n>=8 bucket still owed; g3 = first ep4-YS
observation) — and FD (the showcase; tonight's 62) never appeared in
the decider. Draw disfavored ep4 if anything (it got YS; g15 got
FoD x2). **Protocol upgrade for future deciders: pin the stage to FD**
— the Direct stage draw was the uncontrolled variable in 0809 and
here.

## Caveats (recorded, do not gate the crown)

- n=3 per arm; Bradley's crown call was "probably".
- Blindness protected per-game impressions only: the chain-62 ep4 game
  earlier tonight was known to both parties before the decider (same
  epistemic shape as 0809, where g10b's numbers were known).
- g15's record config carries delay_id=4 (untrained-id deploy) per its
  0809 crowning; ep4 ran id3. Both are each arm's validated recipe.
