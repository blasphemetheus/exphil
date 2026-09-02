# Decode-knob ladder on v1.3-ARrefit (clean trunk + clean extracts) — RESULTS

First ladder run where BOTH the trunk and the instrument are port-correct.
All prior ladder numbers (08-29, 0831_critic_ar, 0831_critic_refit) were
dirty-trunk and/or E1c-diluted.

| N=16, decision frames | v1.2 (dirty stack) | v1.3 in-dist | v1.3 fresh (fox_il_v1) |
|---|---:|---:|---:|
| sampling pass@1 | 1.3% | 5.2% | 2.1% |
| mode-of-N | 7.2% | 18.3% | 8.9% |
| **selector Best-of-N** | 10.8% | **24.5%** | 11.0% |
| oracle pass@16 | 13.9% | 32.8% | 16.0% |
| selector − mode margin | +3.6 | **+6.2** | +2.1 |

Critic internals (train.md): V rank 0.607 (shuffled 0.512); selector−
shuffled-label = +8.7 state-conditional points (v1.2 stack: +3.4).

## Verdict: PARTIAL-CLEAR — first bar-clearing margin ever, in-dist only

- **In-dist clears the pre-registered ≥5 wire-live bar for the first
  time** (+6.2). Fresh corpus stays under (+2.1).
- Everything roughly doubled on the clean stack — the corrupted corpus
  was suppressing sampling coherence, the free mode-of-N vote, AND the
  selector's state signal simultaneously.
- Mode-of-N alone is now a large free offline win (18.3 vs 5.2).
- Per the standing law: offline match-rate belief stops here — an
  in-dist-only clear justifies queueing the LIVE gate experiment
  (frozen-input ≤ 0.20, 7/8 to cap, F1 airdodge, F2 commitment, F3
  approach_delta) for selector and mode-of-N decodes on v1.3-ARrefit,
  not deploying anything.
