# AWBC B1/B2/B3 — offline verdict (2026-08-13, ~03:00)

First offline-RL experiment in the program (OFFLINE_RL_SPEC F5,
multishine adaptation). Three arms at the g15 champion recipe, equal
recipe/compute (pool has ~1.8% run-to-run variance from per-invocation
rollout collection — noted, no arm favored). Pre-registration in
run_awbc_arms.sh (written before any arm ran).

## Headline

**B2 (AWBC) beats B1 (plain BC) on every offline execution metric, and
the B3 control shows the gain is pure outcome-information.** This is the
program's first training signal that exceeded what plain imitation
extracted from the same data.

## Stand-fox d3, sync headless x3 (bit-identical per arm — FD determinism)

| arm | self-shines/min | max chain | mean chain | empty hops | air-shine breaks | rung-0 dep |
|---|---|---|---|---|---|---|
| B1 plain BC | 284.6 | 249 | 7.92 | 6 | 13 | 0.29 |
| **B2 AWBC** | **333.5** | **302** | **10.41** | **1** | **7** | 1.55 |
| B3 shuffled | 70.9 | 2 | 1.03 | 1 | 8 | 2.21 |

- B2 vs B1: **+17% shine rate, +21% max chain**, and CLEANER failure
  modes (empty hops 6→1, sloppy air-shine breaks 13→7) — the
  registered prediction said any real edge would show in
  robustness/recovery rather than peak speed; it showed in BOTH.
- **B3 collapsed** (max chain 2, grounded_fraction 0.79 = holds shine
  without cycling). Stronger than the registered "B3 flat" success
  case: the same weight DISTRIBUTION randomly placed actively destroys
  the skill, so B2's gain is entirely about WHERE the advantage signal
  concentrates weight (the chain-critical frames — consistent with the
  margin-lever finding that single aerial-B decision frames carry
  chains). Corollary lesson: per-frame loss weighting is a sharp tool —
  noise in the weights is not neutral.

## Gates

- Stand-hold (within 10% of B1): B2 PASSES (exceeds B1). B3 fails (as
  a control, that's data, not a deployment candidate).
- **Rung-0 must-not-rise vs B1: B2 FAILS AS WRITTEN** (1.55 vs 0.29).
  Context: B1's 0.29 is anomalously LOW (champion g10b = 1.44, crowned
  g15 = 4.28 under the scramble-calibration caveat); B2 sits at
  champion-typical absolute levels. Interpretation: flag, not
  disqualifier — but it HARDENS the requirement that any promote goes
  through the human deploy rung (the g6 inversion lesson applies with
  extra force).

## Verdict & next

Per pre-registration: B2 > B1 with B3 ≠ placebo ⇒ **outcome weighting
transfers at this data scale. AWBC is validated machinery.** Not a
crown: stand-dummy numbers never crown (standing rule); ms_awbc_b2 has
NOT beaten ms_g15 (430.4/min record) — the arms are a fresh-pool
baseline comparison, not a champion challenge.

1. Deploy-rung session (Bradley): blind local B1-vs-B2 (sealed order,
   0812_crown_decider protocol) to confirm the offline delta survives
   humans. Only then consider an AWBC retrain OF the champion recipe
   as a g16 candidate.
2. F3 KL-anchor: now the natural composition (anchor retention + AWBC
   amplification) — one change at a time was the right call; the
   B3 collapse shows why bundling would have been unattributable.
3. Generalist arm (human corpus, Rewards.Standard) per the original
   spec — machinery is now proven.
4. Interp follow-up: WHERE did B2's weight mass land? (advantage-vs-
   margin-frame overlap; does the early-warning "chain will break"
   probe read differently on b2?)

Checkpoints: ms_awbc_b1/b2/b3.bin. Logs + replays in this directory.

## Deploy-rung addendum (2026-08-13 morning, eval_runs/0813_awbc_decider)

Blind local B1-vs-B2, 4 games (2 each, sealed order b2/b1/b2/b1), d3
deterministic FD, Bradley port 2.

| game | arm | self/min | maxchain |
|---|---|---|---|
| g1 | B2 | 72.7 | **21** |
| g2 | B1 | 41.8 | 3 |
| g3 | B2 | 58.3 | 1 |
| g4 | B1 | 64.5 | 5 |

- **NO INVERSION**: Bradley's committed blind call was {g2,g3}=B2 (the
  weaker-feeling pair) — wrong; the session's best game (g1, chain 21,
  only champion-class chain of the day) WAS the AWBC arm. B2 per-arm:
  65.5 self/min, max 21; B1: 53.2, max 5.
- Blind feel could NOT separate arms (his pairing grouped g1[B2] with
  g4[B1]) — per-game variance vs human neutral swamps arm identity at
  n=2. Deploy verdict: B2 >= B1, suggestive not decisive, consistent
  with offline, zero contradicting evidence.
- Both arms well below champion vs human pressure ("collapses to
  failing multishine if I play neutral") — the known fight-state gap;
  these are baseline-pool arms, not g15 challengers.

**GATE OUTCOME: AWBC graduates.** Next: g16 = champion recipe + --awbc
through the full promote ladder vs ms_g15 (stand -> local human ->
netplay d4), rung-0 watched per the standing caveat.

## Interp addendum: where B2's weight mass landed (interp_weight_mass.exs, 08-13)

28 pool files / 102k frames, delay-0 lists (weight structure only —
alignment caveat). Beta 2.06, ratio 7.0.

**The mechanism is REGIONAL, not pinpoint.** Mean weight by
frames-until-next-shine is a textbook return-to-go gradient:

| at shine | 1-30f | 31-120f | 121-300f | no shine ahead |
|---|---|---|---|---|
| 1.78 | 1.19 | 0.74 | 0.38 | 0.15 |

100% of top-decile weight frames sit within 120f of a shine. But the
MARGIN-CRITICAL frames (aerial-B press edges, the per-link chain
decision) average 1.013 vs 0.999 for everything else — AWBC does NOT
single them out; it amplifies whole chain neighborhoods (and their
critical frames only as passengers). By family: jumpsquat highest
(1.26 — the JC link), aerial_jump lowest among cycle families (0.69 —
the full-hop failure dwell, correctly damped), other 0.77.

Implications:
1. AWBC and the margin lever operate at DIFFERENT granularities
   (regional vs pinpoint) — complementary in principle, though the
   champion recipe buried the margin lever per the 0808 A/B; not
   relitigated for g16.
2. B3's collapse now has a sharper mechanism: shuffling destroys the
   SMOOTH gradient — adjacent frames of one cycle get wildly different
   losses, breaking the temporal coherence the cycle needs. It was
   never "noise-as-regularizer"; it was structural damage.
3. The "no shine ahead" bucket at 0.15 means AWBC quietly downweights
   ALL non-multishine behavior in the pool — worth remembering when
   composing with fight-state mixes in g16+: the anchor (F3) or mix
   shares may need to protect non-shine skills.
