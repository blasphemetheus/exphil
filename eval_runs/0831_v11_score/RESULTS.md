# v1.1 unfreeze score — AR vs IND (plan §6) — RESULTS

**2026-08-31 08:30.** Both arms: 3 epochs from ep10's trunk (trunk
transplant), seed 828, FIXED fox corpus (`--select-character-port`).
8×120 s vs CPU each + §6 instruments. Read
`AUTOREGRESSIVE_HEAD_PLAN.md` §6 before these numbers; per-instrument
files in this directory.

## Verdict: NO SIGNAL by the pre-registered §6 rule — the live look decides

The rule wanted AR recovery-route shares ≥2× IND with died-given-route
lower. **The route comparison INVERTED instead**: IND's first-route up-B
15.6% vs AR 9.4% (expert 16.3); AR airdodges more (37.5 vs 20.0). But
AR wins the *outcomes*: died-offstage 28.1% vs 35.6%, back 71.9% vs
62.2%, deaths/game 1.13 vs 2.13, val 5.480 vs 5.973. TV tied (0.54 vs
0.53). No collapse either arm (7/8 to cap, staleness ≤0.2%).

**Both arm-vs-arm live differences sit at or under the D1 floors at
n=8** (deaths 1.9× vs the 2.5–3× floor; routes are 3-vs-7 episode
counts). The one solid between-arm measurement is the val gap (0.49
nats, the conditioning term). Per the g6 rule the recipe decision was
always Bradley's live look; it now carries all of it.

## The headline that is NOT arm-vs-arm: the unfreeze worked

Both v1.1 arms crush ep10's CPU rung on recovery:

| | expert | v1.1-AR | v1.1-IND | ep10_cpu |
|---|---:|---:|---:|---:|
| recovery: died % | 12.7 | **28.1** | **35.6** | 60.7 |
| recovery: back % | 79.1 | **71.9** | **62.2** | 32.1 |
| deaths/game | 3.18* | 1.13 | 2.13 | — |
| edgeguarded deaths % | 39.3 | **0.0** | 23.5 | — |

(*expert plays humans, not a 120s-capped CPU — column is context, not a
bar.) Head + corpus are confounded in the v1→v1.1 delta **by design**
(pre-registered); the AR-vs-IND pair is the controlled comparison and it
is unresolved offline.

## Open question flagged (L1 before alarm): the coincidence inversion

8a (frozen trunk): AR offstage P(up|B) lift 2.20–2.65× vs IND 1.05×.
Post-unfreeze arm replays: **AR 1.28× vs IND 1.93×.** Candidate
mechanism, not yet tested: with the trunk unfrozen, joint training can
move the dependency INTO state-conditioning (P(B|s) and P(up|s) sharpen
together), relieving the AR head's conditional pathway — and the IND
arm can only express it that way. If real, that is itself interesting
("the trunk absorbs the coincidence when allowed to") and would predict
IND's val gap closing with more epochs — it did not (0.49 nats stable),
so small-n (32/45 episodes) remains the boring favorite. A dedicated
teacher-forced coincidence probe on both checkpoints would settle it
offline; queued behind the live look.

## Next actions

1. **Bradley: live look** at `fox_gen_v1.1_AR_20260831_080100_policy.bin`
   (and IND for contrast) — deploy knobs: local windowed, T=0.5/0.5,
   frame-delay 0. This gates the head recipe AND the `--head` default
   flip (pre-registered).
2. Blind B1-vs-B2 pair same session (C1 calibration).
3. If AR passes: rerun Leg S + the C2/C3 polish instruments on v1.1-AR
   (the parked critic line resumes on it; G3b spike gate already open).

Ops note: first launch of the night died at root-disk 100% — see GOTCHA
#106; stale port-1 caches moved to `/data/exphil/old_port1_embedding_cache`.
