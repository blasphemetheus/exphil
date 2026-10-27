# Trained Mamba parity and resync measurements

**AI-generated report.**

Two input rounding inconsistencies are fixed. Single and batched inference now
receive identical embeddings in all 1,280 decisions checked per backend/build.
Every deterministic controller also matches. Strict carried-state parity still
fails for some convolution-buffer values; it is not marked complete.

## Numerical checks

The final runs use the supported staggered schedule, eight different replay
sections, the same trained Fox Mamba checkpoint, and 160 decisions per row.
The checkpoint was trained at delay zero with an 80-frame window. Both runtime
stacks are tested on CPU and CUDA. No tolerance was increased.

| Build / backend | Embedding differences | Controller differences | Conv-buffer comparisons failing `1e-4` | Largest conv-buffer difference |
|---|---:|---:|---:|---:|
| Installed / CPU | 0 / 1280 | 0 / 1280 | 252 / 1280 | 0.001076 |
| Installed / CUDA | 0 / 1280 | 0 / 1280 | 21 / 1280 | 0.000343 |
| Candidate / CPU | 0 / 1280 | 0 / 1280 | 209 / 1280 | 0.001023 |
| Candidate / CUDA | 0 / 1280 | 0 / 1280 | 194 / 1280 | 0.001001 |

Every SSM hidden-state comparison passes the same elementwise tolerance. In
sampled one-step comparisons with identical incoming state and input, all
exposed stages and complete-step outputs pass. The first numerical difference
is in the input projection: at most `3.81e-6` in f32, versus approximately
`3.6e-15` to `7.1e-15` when the same projection uses f64. This is evidence for
batch-shape-dependent floating-point summation, rather than different input
values or different recurrent equations. It does not prove every later error
comes from that first projection alone.

Returning intermediate tensors can change compiler fusion. The stage probe
and the normal complete step are therefore separate checks. Repeated steps
amplify small differences; the strict convolution-buffer failures remain in the
reports. Exact controllers on this finite deterministic replay do not establish
identical stochastic actions or every possible gameplay decision.

Before the embedding fixes, an exploratory aligned run had differing embeddings
on 1,018 of 1,280 decisions on each backend. Fixing timing features reduced that
to 413; fixing distance eliminated the rest. Both fixes match existing training
preprocessing. Their exact-equality regressions fail before the changes, and
all 152 focused embedding tests pass afterward.

## Framework compatibility

The installed process loads Axon 0.8 and XLA 0.10. The isolated candidate loads
Axon 0.9 and XLA 0.11. Both advertise Nx/EXLA 1.0, so the reports also retain
actual build paths, BEAM hashes, and EXLA native-library hashes. Main's dependency
declarations and shared native libraries are unchanged.

Both CPU and CUDA comparisons between builds pass: all 160 batches of eight
controllers match exactly, and all four sampled output-feature comparisons
meet `atol=rtol=1e-4`. Source/input hashes and schedules match. This validates
trained deterministic inference for these observations. It does not merge the
framework branch or substitute for its training-update/resume checks and
longer opponent-quality evaluation.

## Dolphin scheduling results

Two independent agents control both ports of a real headless Dolphin match.
Each arm measures 600 game frames; the first 160 are excluded. Two rounds
reverse schedule order. Requests use an absolute 60 Hz clock. No competing
GPU compute process was detected. The executable, ISO, checkpoint, and sources
are hashed in `dolphin_paced/summary.json`.

| Backend / schedule | Calls over 16.67 ms, rounds 1 / 2 | Pair inference p99, rounds 1 / 2 |
|---|---:|---:|
| CPU aligned | 14 / 14 | 22.06 / 21.51 ms |
| CPU staggered | 0 / 0 | 12.23 / 11.88 ms |
| GPU aligned | 0 / 0 | 9.39 / 11.62 ms |
| GPU staggered | 0 / 0 | 6.67 / 6.41 ms |

There are 440 measured calls per arm, each producing two controllers. CPU
staggering removes the periodic inference-budget overruns in this test. It
raises p95 from roughly 3 ms to 11 ms because twice as many frames now contain
a replay, while the worst frames improve. GPU p95 similarly rises from
2.9–3.6 ms to 5.7–5.8 ms, but both schedules already meet the inference budget.
Keep staggering opt-in; it is useful for CPU independent agents and may help
GPU tails, but it is not a universal latency improvement.

Actual frame-interval medians are about 17.1 ms; p95 spans 17.2–18.1 ms. These
include console reads and scheduler timing. Blocking input means this is an
inference-budget screen, not an asynchronous deadline guarantee or evidence of
better win rate. The original relative-sleep pacing experiment had about 20 ms
frame intervals and is excluded from this table.

Longer GPU runs used the staggered schedule for 3,600 frames on each build.
Both completed without crashes and with **0 / 3,440** warm-frame inference
budget overruns. Pair p99 was **7.18 ms installed** and **7.03 ms candidate**.
These were separate native matches, not identical seeded trajectories, so the
small timing difference is not an upgrade-speedup claim. Both used the policy's
trained delay of zero. Netplay delay regimes and opponent-quality promotion
remain separate work.

Run the longer check with `--phase 15 --frames 3600`; run the candidate through
`scripts/run_framework_validation.exs` with the same isolated stack used by
the numerical checks. The two `dolphin_long_*_paced.json.gz` files retain all
per-frame samples.

## Reproduction and artifacts

[The guide](../../docs/guides/MAMBA_BATCH_PARITY.md) describes the runners and
commands. `*_final_summary.json` and `*_candidate_summary.json` summarize the
four final runs. The two `*_framework_comparison.json` files retain comparisons
across builds. Full numerical records are compressed as `*.json.gz`; use
`gzip -dk FILE.json.gz` before passing a record to the Elixir tools. Compression
does not alter the JSON. Early exploratory runs and the first relative-pacing
Dolphin runs are not the final timing evidence.

**AI-generated poem**

> The inputs meet, the sums diverge;
> Small rounded ripples cross and merge.
> We mark the gap and keep the test;
> The measured clocks can tell the rest.
