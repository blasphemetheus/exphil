# Human-play findings, 2026-08-03/04 — and the work they imply

First real remote Direct sessions with a proper replay capture. Two
opponents (SNO#395 "Greg", ACAB#182 "Brandon"), four checkpoints, **14
replays + full qtrace logs** saved to `eval_runs/0803_direct_*` and
`eval_runs/0804_direct_*`. This is the corpus the P5 curation loop has been
blocked on since it was written.

The session changed the project's headline claim. Read this before trusting
any pre-08-04 statement about multishine capability.

---

## Findings

### F1. Real netplay latency is a sharp 5 — the delay campaign is CLOSED

Every game, both opponents, both connections: `analyze_qtrace` peak at lag
5, 99.3-100% agreement, one clean peak (no smear). Identical to local
loopback.

Consequences: `--frame-delay 3` is correct for real Direct play; the
network adds nothing measurable; the **18-21-frame "Phillip target" tier is
formally dead** (it was already deprioritized 08-03 on ladder evidence —
this is the empirical confirmation). Task #12's measurement is delivered.

### F2. NO checkpoint multishines against a human — chains are dummy-only

Measured from replays with `analyze_shine_source` (chain length), not from
qtrace press counts — see F5.

| policy | shines/game | self-shines | **max chain** | vs stand dummy |
|---|---|---|---|---|
| `ms_g6_sp1` | 0, 0 | 0, 0 | **0** | 434/min, chain 434 |
| `ms_g7_pressure` | 18 | 8 | **1** | 72/min |
| `ms_g2_mdq_ss` | 21-29 | 14-27 | **1** | 380/min, chain 367 |
| `ms_g4_d2mix` | 40 | 36 | **2** | 424/min, chain 423 |

Every one of these chains 400+ unbroken against a standing dummy. The best
result against a person was a chain of **two**.

"The multishine problem is solved" was true of a static environment only.
The technique does not survive contact with an opponent.

### F3. The promotion criterion is INVERTED, not merely uninformative

`ms_g6_sp1` was crowned production on 08-03 for winning the stand-dummy
comparison (434 vs 380). It scored **zero shines in two human games** while
the policy it displaced shone 20-27 times per game.

Better predictor on this evidence: **chain strength at the rung you deploy
at (d3)**. The two d3 whole-run chainers (`g4_d2mix` 424 c423, `g9_sp34`
428 c429) gave the best human results; `sp1`, whose strength is d2, gave
the worst. Rank by deploy-rung chain, not by headline number.

### F4. The absorber has TWO doors: pressure AND stage

The "crouching" both opponents saw is the hold-B fixed point (GOTCHAS/
INIT_FORENSICS: Melee registers button EDGES, so a held B is a no-op; runs
of 1013 and 1235 frames were traced). Two independent triggers:

- **Human pressure** — states no dummy ever produces
- **Stage** — "on DL it holds shine", FD plays fine. Every drill fixture is
  FD-only, so Dreamland is off-distribution

The absorber is therefore the generic response to *any* unfamiliar state
distribution, not something specific to opponents. **That means it can be
reproduced without humans** (task A).

### F5. Measurement lesson: B-press rate is NOT shines is NOT chains

Mid-session I reported "`mdq_ss` is 10x better vs humans" from qtrace
**B-press cycles/min** (150-292 vs 16-26). Bradley's eye contradicted it
("it shines once in a row"), and the replay-level chain metric agreed with
him: max chain 1 for those same games. The press counter includes presses
that never produce a reflector (held B, presses in hitstun, eaten inputs).

The directional claim survived (`mdq_ss` shines, `sp1` doesn't); the
magnitude was noise. **Score behavior from replays with the technique
metric; use qtrace only for latency and input-plumbing questions.** This is
GOTCHAS #79's lesson recurring in a new costume — the observer's eye beat
the instrument.

### F6. The human corpus exists now

14 replays survived because the `--replay-dir` guard landed hours earlier
(GOTCHA #84 — the 08-02 couch corpus was lost to exactly this).

---

## Work items

### A. Reproduce the absorber OFFLINE via stage shift  *(task #30 — DONE)*

**RAN 2026-08-04** (`eval_runs/0804_stage_absorber.{sh,log}`, ms_g4_d2mix,
stand dummy, d3, stage the only variable, 3 runs each):

| stage | run 1 | run 2 | run 3 |
|---|---|---|---|
| FD | 424/min c423 | 424 c423 | 424 c423 |
| Battlefield | 425 c423 | 425 c423 | 425 c423 |
| Dreamland | 406 c401 | 396 c209 | 399 c315 |
| **Yoshi's Story** | 286 c236 | **40 c1** | **44 c2** |

**A1. The DL hypothesis is REFUTED.** Dreamland chains 200-400 against a
stand dummy — nothing like the live "holds shine". So the live DL failure
needed the HUMAN too; stage alone was not sufficient. BF is pixel-identical
to FD, so "not-FD" is not the trigger either.

**A2. But we got the offline reproduction anyway — on Yoshi's Story.**
2 of 3 runs collapse to chain 1-2, and forensics confirm it is the ABSORBER,
not an SD or stage-edge artifact:

| run | squat occupancy | offstage | end stock |
|---|---|---|---|
| YS collapsed (r2) | **52.2%** | 0.0% | 4 |
| YS good (r1) | 2.5% | 0.0% | 4 |
| FD (r1) | 0.1% | — | 4 |

Half the game spent in Squat, all stocks intact. That is the hold-B fixed
point, reproduced with no human, in 60 seconds, on demand.

**A3. Bonus — it is STOCHASTIC on the same stage, which is better for
study than a deterministic failure.** Same policy, same dummy, same stage:
one run chains 236, two collapse. That gives a *contrastive pair* (good vs
absorbed trajectory, everything else held constant) — exactly the setup
that cracked the crouch zoo in July, and it feeds the CycleSim/BasinRollout
instruments directly.

**A4. Harness caveat discovered:** "3-run determinism" is a property of FD
and BF, NOT of the harness. DL and YS vary run to run. Any future stage
work needs more runs before trusting a number.

### B. Fix the promotion criterion  *(task #31)*
Revert the production designation from `sp1`; add a pre-crowning gate that
scores at the deploy rung AND against a **moving** opponent (`--p2-policy`
ladder path exists; cpu-9 measured flat 21-26 across policies and does not
discriminate). Record F3's rule in LATENCY_ARCHITECTURE.

### C. Curation cycle 3 on the human replays  *(task #32)*
`snippet_mine.exs` → `dagger_drill --snippet-frames`. Cycle 2 proved
snippet-dose mixing is SAFE for the core skill (stand held 385 c374) but
gainless on *synthetic* pressure; the missing ingredient was real human
states, which now exist. Gates: stand d3 >= 300, plus B's moving-opponent
gate or A's DL repro.

### D. Architecture bake-off, re-aimed  *(task #27)*
Already unblocked; now rank backbones by **d3 chain and off-distribution
(DL) robustness** rather than headline dummy score, per F3.

### E. Multi-stage training  *(task #33)*
Stage is already a network input (7-dim compact encoding) — it has simply
never seen variety. Record/synthesize fixtures on BF/DL/YS. Likely removes
an entire class of off-distribution failure independent of the pressure
problem. Note `Constants.fd_edge_x` is hardcoded in several places.

### F. More human games  *(opportunistic)*
`g4_d2mix`'s chain-of-2 is one game; `sp1`'s zero is two. The harness now
captures everything automatically, so any future session adds data for free.
