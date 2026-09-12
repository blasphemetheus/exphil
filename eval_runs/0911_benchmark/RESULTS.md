# Multishine benchmark pilot — 2026-09-11

Retrospective scoring only. No gameplay, training, or teacher interventions were
run. Inputs are frozen in `benchmarks/multishine/0911_pilot.json`; full records,
hashes, recovery events, and teacher examples are in `pilot.json` beside this file.
These previously inspected games are not held-out confirmation.

Metric: `multishine_reentry_v1`. Protocol is declared async d3/id2, T=1.0,
stand/CPU on FD; it is not independently verified from the recordings. The
manifest has one g23a stand game and two games in each other cell, so it is an
unbalanced pilot. No significance or promotion claim is justified.

| Policy/scenario | Self-initiated shines/min | Max chain | Re-entry completed / censored |
| --- | --- | --- | --- |
| g23a ep57, stand | 304.5 | 277 | 2 / 0 |
| g23a ep57, CPU run 1 | 43.2 | 3 | 8 / 40 |
| g23a ep57, CPU run 2 | 54.3 | 2 | 7 / 39 |
| g24a ep55, stand run 1 | 113.7 | 27 | 8 / 1 |
| g24a ep55, stand run 2 | 94.4 | 3 | 5 / 1 |
| g24a ep55, CPU run 1 | 62.2 | 9 | 6 / 44 |
| g24a ep55, CPU run 2 | 64.8 | 8 | 9 / 39 |

Rates differ from the older scorecards: this metric excludes pregame frames,
uses its explicit recent-hit rule, and breaks chains on reported hitstun.
Compare within this report rather than mixing metric versions.

## Interpretation

- g24 retains the observed moving-opponent rate/chain improvement and stationary
  regression. Improved CPU chaining does not establish reliable recovery.
- Many CPU recovery episodes end in another interruption, death, or replay end.
  Do not summarize only successful re-entry times. A large censored count also
  does not mean every episode was an avoidable policy mistake.
- The two g23 CPU games use fallback teacher labels for 4,668/5,378 and
  4,593/5,258 audited pairs. g24 uses 4,544/5,378 and 4,373/5,260. Most CPU
  supervision in this audit therefore comes from handwritten recovery rules,
  not the demonstrated fixture table.
- Those games include 105–138 fallback labels during reported hitstun. These
  are review candidates, not proven invalid labels: an input can be intended
  for a later opportunity. This audit does not model queued execution.

The next useful validation is whether the teacher's corrections actually return
the character to a sustainable loop from matched failure states, followed by
fresh balanced reference games. Do not infer teacher correctness from imitation
loss, agreement with recorded actions, or fallback coverage alone.

See `docs/guides/MULTISHINE_BENCHMARK.md` for timing definitions, censoring,
manifest usage, and the controlled experiment checklist.
