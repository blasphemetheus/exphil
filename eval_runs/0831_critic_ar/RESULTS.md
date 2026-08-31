# Critic ladder on coherent candidates (v1.1-AR) — RESULTS

**2026-08-31 15:05.** Plan (a) of the 08-31 critic sequence: the 08-29
ladder re-run with `sample_autoregressive_kn` candidates (k=16) on
v1.1-AR. Decision rule pre-registered in CRITIC_D2_DESIGN (08-29):
STRONG = ≥50% of the sampling→oracle gap recovered on held-out AND
fox_il_v1, **beating mode-of-N by ≥5 pts** → wire live; PARTIAL 20–50%
→ MLP head; NULL → V + rollouts.

## The ladder

| decode | in-dist (37,502 frames) | fresh fox_il_v1 (19,146) |
|---|---:|---:|
| sampling pass@1 | 3.4% | 1.3% |
| mode-of-16 (critic-free) | 11.0% | 5.1% |
| **selector Best-of-N** | **13.7%** | **6.5%** |
| oracle pass@16 | 21.4% | 11.1% |
| gap recovered | 57.4% | 52.5% |

Train-side: selector pass@1 16.5% vs shuffled-label control 11.5% —
the control recovers a lot on its own (generic action-frequency
preference, i.e. mode-seeking in the features), so the label-specific
signal is the 16.5-vs-11.5 margin, not 16.5-vs-4.3.

## Verdict: PARTIAL-plus — better than 08-29, under the wire-live bar

- **First time the learned selector beats the free majority vote**, on
  both corpora (08-29 it LOST to the vote). Coherent candidates changed
  the ordering.
- Gap-recovered clears 50% on both corpora ✓ — but the **margin over
  mode-of-N is +2.7 / +1.4 pts, under the declared ≥5** → NOT wire-live.
  Per the rule's PARTIAL branch the next escalation is an MLP selector
  head; the NULL-branch V+rollouts work (G3b) is already queued and
  remains the tail-keeping candidate (L9: an offline mode-beater still
  owes the live gate — mode-of-16 also won offline and froze live).

## Caveats that travel

1. **v1.1-AR's conditioning wire is atrophied** (L_cond 1.14, probe) —
   these "coherent" candidates are only mildly so. The natural rerun is
   this exact ladder on v1.2-ARrefit once the refit lands (~25 min).
2. Absolute match rates here are NOT comparable to interp_passk's
   (different match rule, denominators, stride): only within-table
   differences mean anything (standing law).
3. Offline master-match remains mode-seeking (L9); any wiring decision
   waits on the live gate (frozen-input ≤ 0.20, 7/8 to cap) plus the
   new live metrics (F1 airdodge-in-danger, F2 in-threat commitments).

Runner: `scripts/critic_ar_chain.sh`; data `cache/critic/v11ar_erickfm40_k16.nx`.
