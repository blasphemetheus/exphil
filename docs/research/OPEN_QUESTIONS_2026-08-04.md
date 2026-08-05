# The 20 most interesting open questions (2026-08-04)

Compiled from: HANDOFF_2026-08-04, HUMAN_PLAY_FINDINGS_2026-08-04,
INTERP_ROADMAP.md (v1 experiment log + milestone verdicts),
INTERP_NEXT_RESEARCH_2026-07-20.md, INTERP_ROADMAP_V2.md,
ML_FIELDS_ROADMAP.md. Each question cites the finding that raised it.
Ordered roughly by how much an answer would change the program.

## The human gap (the headline mystery)

1. **What exactly does a human's presence perturb such that every
   policy chains 400+ on a dummy and at most 2 on a human?** Is it
   perception (OOD states), control (pressure-induced input noise), or
   objective (imitation can't exceed its demonstrator under pressure)?
   — *HUMAN_PLAY_FINDINGS_2026-08-04; the 08-04 headline correction.*

2. **Why is the dummy ranking INVERTED against humans** (g6 beats g4 by
   54 shines/min on the dummy, loses 40-0 against a person) — is static
   overfit literally opponent-input-blindness, or something subtler?
   Testable now via the W2 ablation on the g6/g4 labeled pair.
   — *HANDOFF_2026-08-04 correction table; INTERP_ROADMAP_V2 W2.*

3. **What is the absorber's entry mechanism?** Same checkpoint, same
   stage (YS), stand dummy: 2 of 3 runs collapse to 52% squat, one
   plays fine. What state difference in the frames before divergence
   decides which basin a run falls into — and why is Yoshi's Story the
   only stage that does this offline?
   — *HANDOFF_2026-08-04 task A / #34; the stochastic contrastive pair.*

4. **Is the failure-under-pressure a general law of BC policies, or
   specific to cycling skills?** The multishine is a metronome; does a
   less rhythm-locked skill (tech-chase, edgeguard) degrade as sharply
   against humans?
   — *implied by HUMAN_PLAY_FINDINGS; untested.*

## Representation & mechanism (from the interp program)

5. **Why do trained trunks COMPRESS AWAY information relative to their
   own raw inputs** (memory-probe accuracy: trained 0.65-0.73 < random
   init 0.77 < raw floor 0.84)? Is behavior-driven pruning universal
   across architectures and tasks, and can an auxiliary objective stop
   it selectively?
   — *INTERP_ROADMAP P1 v2 verdict, 2026-07-13.*

6. **What determines whether the heads CONSULT a feature the trunk
   knows?** The trunk enriches opp_behind (0.72 > raw 0.64) while the
   heads ignore it; the fix had to go through the teacher. Is there a
   trainable-time signal that predicts knowing-without-acting before
   deployment reveals it?
   — *P3 case #3, 2026-07-15; the layer-level verdict.*

7. **What channel in a SINGLE current frame predicts the dummy's
   "random" future tech choice at probe accuracy 1.000** — 30+ frames
   before the animation, RNG confirmed unseeded? Physics/DI leak,
   facing-direction coupling, or a label-construction artifact? Never
   resolved.
   — *P1 v1 open mystery, 2026-07-13.*

8. **Why is tech choice weakly decodable (0.36-0.40 vs 0.33 chance) 30
   frames BEFORE the episode even starts?** A static positional tell in
   the dummy's setup would be exploitable by training — and would
   contaminate any "reaction" claim.
   — *P4 verdict footnote, 2026-08-02.*

9. **Can BC ever learn to READ techs, or does gradient starvation
   structurally prune sub-1% events?** P4 says nobody reads techs;
   tech episodes are <1% of signal. Does episode upweighting produce a
   post-entry rise in the offset curve (the pre-registered acceptance
   test), or is there a floor imitation can't cross?
   — *P4 verdict + acceptance test, 2026-08-02; task #8.*

10. **Why do GRU and Mamba learn the SAME features with the SAME
    deficits** (own_offstage, own_shielding present; opp_knockdown
    ~0.07 in both)? If features are task-driven not architecture-driven
    at this scale, does the cross-arch crosscoder find ANY
    architecture-specific feature at all?
    — *P6 SAE cross-arch verdict, 2026-08-03; survey's crosscoder item.*

11. **Why is the shield-steering axis all-or-nothing** — α=1.0 causal
    and clean (9/9 card, breaks 215f→32f) but α=0.5 WORSE than
    baseline? What geometry makes an intervention direction
    non-monotonic in strength?
    — *P6 steering A/B, 2026-07-19.*

12. **What is the general theory of closed-loop compounding?** LEACE:
    surgically perfect offline, catastrophic live (policy immediately
    visits states outside the eraser's fit). The same offline/live gap
    shape recurs in evals and cycles. Can closed-loop failure be
    predicted from an offline quantity (e.g. F2's OOD score)?
    — *P6 LEACE entry, 2026-07-14; ML_FIELDS_ROADMAP F2.*

## Training dynamics

13. **How much training instability is COMPILER-dependent?** The NaN
    autopsy showed the stable BCE form was rewritten into an unstable
    exp variant by XLA fusion — stability was a property of the
    compiled program, not the math. What else in the loss/grad path is
    one fusion decision away from a cliff?
    — *P2 crime-scene autopsy, 2026-07-14; the ±60 clamp fix.*

14. **Do features form BEFORE behavior improves, and can formation
    curves pick export epochs?** The P2 open item (probe-eval every N
    epochs) was never run; loss demonstrably doesn't rank checkpoints.
    A yes would replace loss-based export selection everywhere.
    — *INTERP_ROADMAP P2 open item; motivating observation in v1 assets.*

15. **Why does human habit metastasize under BC?** ~7f L-cancel trigger
    taps became 215f shield-break holds via copy-loop amplification —
    no shield concept existed in the data at all. What OTHER
    demonstrator habits are currently being amplified into pathologies
    not yet noticed?
    — *#30 verdict, 2026-07-14; the metastasized-L-cancel finding.*

16. **What makes a delay-rung set COMPOSE?** Multi-delay {2,3} + SS
    transfers sharply; adding rungs or spacing them differently
    shattered the d2 pin (grind-5: 205.8 c73). No theory exists — only
    the empirical champion recipe.
    — *task #20; HANDOFF_2026-08-03 rung-spacing surprise.*

## Data & evaluation

17. **What does a replay frame actually CARRY?** Cycle 1's +43%
    pressure gain apparently came from redundant context frames
    (distribution matching), not the hit-recovery moments the miner
    keeps — cycle 2 kept the moments, lost the gain. What unit of data
    (frame, snippet, distribution shape) transfers a skill?
    — *P5 cycles 1-2 reading, 2026-08-03; the cycle-3 prereg hinges on it.*

18. **What is the minimal replay set that reproduces g4-level skill?**
    20,317 relabeled human frames vs 800k+ synthetic is cycle 3's
    implicit version of the question; a 10x data reduction is a 10x
    faster loop on this hardware.
    — *cycle-3 prereg, eval_runs/0804_cycle3_human.sh; ML_FIELDS F4.*

19. **How few Dolphin games can a promotion decision cost at a fixed
    error budget?** The g6 disaster fixed the ranking rule (deploy-rung
    chains) but the statistics are still fixed-count and uncalibrated
    (one human datapoint per policy). Sequential stopping + paired
    determinism (CRN on FD/BF) should compress it — by how much?
    — *promote_check design notes, HANDOFF_2026-08-04; ML_FIELDS F1.*

20. **Why is real netplay latency a SHARP 5 frames** — identical on two
    remote connections and loopback, with none of the anticipated
    jitter? Where did the variance go (Slippi's rollback absorbing it?),
    and does the answer hold at worse pings than tested?
    — *HANDOFF_2026-08-04 latency finding; delay campaign closure.*

---

Meta-note: questions 1-4 are one cluster (the static-environment
overfit story told four ways) and 5-6 another (what training prunes and
why); an answer to any member likely moves its siblings. The
highest-leverage single experiment on the board is the W1 absorber-entry
diagnosis, which touches questions 3, 12, and 17 at once.
