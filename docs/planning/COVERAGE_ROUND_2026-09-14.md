# Coverage round at rung 4: held-out starts, positions, facings, opponents (2026-09-14)

Bradley's ask after the delay-4 proof: "record and validate teacher clips at
rung four for held-out starts, positions, facings and opponents, then train
on the full pool and gate with the same chain." Root:
`eval_runs/0914_coverage_round/` (stages `collect.sh`, `gen2.sh`,
`teach.sh`, `train.sh`; `progress.log`; failed attempts kept as
`*_attempt*`).

## Candidate

`eval_runs/0914_coverage_round/round21/candidate.bin`
SHA256 `90d63f3560344ccee0bf348412c82fd993ed1f610aa26672d777e871f20621a4`.
Proof contract at delay 4 (windowed GRU F32, zero state, GRU64x2, AR head,
window 16, queue 5, prev action, dropout 0.0), fresh init, 21 epochs on
canonical + the nine original handoffs + 39 coverage handoffs, all cold+warm
(38,371 supervised targets; 147,235 draws/epoch with x64 early weighting);
final loss 2.35e-4.

## How the source games were made (two findings on the way)

1. Rollouts of the delay-4 candidate at reaction 4 vs level-1 CPU Marth,
   Falco, Peach, Samus, Fox (2 x 90 s each; 102-148 shines/min, chains
   14-31). Handoffs mined from these drifted 70/77 at the prefix replay.
2. **Suite bug:** port 2 was hard-coded to Fox, so every non-Fox prefix
   drifted from the countdown (frame -38). Fixed: the body comes from the
   source replay (`--opponent-character` overrides).
3. **CPU ports are unreplayable:** the CPU AI sets stick floats no
   controller byte reproduces (0.0039 recorded -> 0.0062 replayed); the
   error compounds until the first hit differs (frame ~112). The bot's own
   bucketed inputs replay exactly. Likely the unexplained "28/36 drifted"
   of RECOVERY_LABEL_CONFIRMATION.
4. **Workaround:** `gen2.sh` re-plays each rollout in the suite as an
   input-driven game (policy from frame 30 for 5000 f vs a GHOST of the
   recorded opponent inputs on the right body). Both ports are then
   bridge-driven; mined prefixes replay with zero drift. 13 games (10 CPU
   ghosts + Bradley's 09-13 session as a human ghost + 2 old Fox ghosts).

`scripts/mine_coverage_handoffs.exs` mined 54 handoffs (38 hit starts, 16
neutral starts, stratified by x bucket / facing / opponent side; Fox 18,
Peach 10, Samus 10, Marth 9, Falco 7). Teacher (neutral opponent after
handoff, 360 f, `--audit-teacher-labels`): 54/54 pass, chains 37-40, 1
prefix drift, 2 Dolphin errors -> 51 qualified. All 51 validate at delay 4
with 0 target mismatches -> 102 cold/warm clips. Split
(`scripts/split_coverage_clips.exs`, every 4th per class held out): 39
train, 12 held out, every opponent body in both.

Limitation: neutral starts cluster at the spawn x (the bot rarely walks);
position variety comes from knockback and opponent side.

## Frozen fit (delay 4)

- Train clips (96, cold+warm): gate PASS, 18/18 on every clip.
- Held-out clips (24, never trained): first-18 conditional argmax mean
  **94.7%**, worst clip 72.2% (teacher-forced; informative only).

## Closed loop at `--reaction-delay 4` (T=1.0, 2 runs/handoff)

| gate | result |
|---|---|
| original 9: cold / warm (120 f) | **12/12 and 12/12**, chains 13-14 |
| original 3 interruptions, neutral opp (360 f) | 6/6 chain >= 10 (40,40,36,30,32,29) |
| coverage TRAIN, 39 handoffs, warm, neutral opp | **75/78** chain >= 10 (misses: chains 4, 1, 1) |
| coverage HELD-OUT, 12 handoffs, warm, neutral opp | **24/24** chain >= 10 (21-40) |
| coverage HELD-OUT, replay opp (still attacking) | **22/24** chain >= 10; strict recovery on the 9 hit handoffs: 18/18 runs recover, ready-to-cycle 9-10 f typical |

All 156 runs: valid timing, zero prefix drift, zero errors.

Held-out detail: every hit start vs Falco/Marth/Peach/Samus recovers at 9-12
frames and chains 36-40 with the opponent still attacking. The weak cells
are Fox-ghost handoffs: hit 299 (chain 5/28 under replay, ready-to-cycle
65 f once), neutral 455 (chain 2 once; re-entry 175-226 f in three of four
runs) and Marth neutral 1993 (re-entry 96-200 f). Neutral starts far from
the loop re-enter slowly; hit starts re-enter fast.

## Verdict

The recipe generalizes across opponent bodies, facings, and post-hit
positions at rung 4: held-out handoffs it never trained on pass 24/24
(neutral) and 22/24 (under pressure), with fast strict recoveries. The
regression gates on the original nine still pass. Residuals: slow re-entry
from far-from-loop neutral starts, and the crouch/hop stall class seen
before. Nothing installed as a default; Bradley's live look decides.

## Costs

Rollouts 17 min; gen2 13 min; mine+teacher 5 min; validate+export 1 min;
train 6 min; frozen fit 1.5 min; live gates 48 min.
