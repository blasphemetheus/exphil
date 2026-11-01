# Mamba learning comparison

**AI-generated.** This is a small training pilot, not a policy promotion.

## Result

Canonical Mamba-2/state 16 is the best next candidate for a larger training
comparison. It learns this held-out sample faster than Mamba-1 on both seeds.
State 128 has nearly the same loss at 300 updates, but loses the equal-time
comparison. Means across two seeds:

| Model | Loss at 300 updates | Loss at two minutes | Updates in two minutes |
| --- | ---: | ---: | ---: |
| Mamba-1/state 16 | 5.649 | 4.743 | 1,797 |
| Canonical Mamba-2/state 16 | 4.867 | 3.951 | 2,598.5 |
| Canonical Mamba-2/state 128 | 4.847 | 4.223 | 2,301 |

The small-state Mamba-2 completed about 45% more updates in the same budget.
At the final snapshots, process-resident GPU memory was approximately 9.3 GiB
for Mamba-1 and 9.6–9.7 GiB for Mamba-2. These allocator-residency samples do
not establish a peak-memory advantage for any model.

All twelve exports loaded and played. The 96 time-limited trajectories and one
completed match did **not** demonstrate stronger gameplay for Mamba-2. The
completed match was a pilot loss; the frozen opponent usually led at the time
limit. Mamba-1's two-minute snapshots led 4/16 time-limited trajectories versus
1/16 for small-state Mamba-2 and 2/16 for large-state Mamba-2. The sample is too
small, and the policies too briefly trained, to use this as a final ranking.
Keep the deployed policy unchanged. Expand the training data and held-out game
count, then use longer matches before choosing a replacement.

Full per-run measurements and gameplay outcomes: [pilot/RESULTS.md](pilot/RESULTS.md).

## Method

Compare ExPhil's existing Mamba-1 with canonical `mamba2_full`, with state sizes
16 and 128. All start from fresh weights. Hidden width 512, two layers, window
80, batch 128, f32, learning rate 0.0002, and a constant learning-rate schedule
are shared. The remaining input and loss settings follow the master Fox
coherence recipe in `eval_runs/1024_master/recipe.sh`.

Use 32 local master-corpus games, selected by the existing sorted file loader.
Hold out three whole games, chosen by path hash. The other 29 provide training
examples. This small file sample is not representative evidence about the full
corpus. Validation loss is the existing held-out objective, without training's
offstage/onset weighting. Do not compare it numerically with the training loss.

The runner saves two points along each model's training trajectory: 300 updates
and 120 seconds of training. The clock includes batch loading and the first
training compilation. It excludes initial data preparation, initial validation,
and snapshot evaluation/export. It therefore measures a training budget, not
total process runtime. Both are recorded. First-step time is also recorded.
There is no warmup update discarded from learning. The time threshold is checked
after each completed update, so it can overshoot by one update.

Seeds 905 and 906 are repeated across all three models. Model order reverses for
the second seed. Model sizes differ; parameter counts are recorded. This is an
architecture comparison at equal width, not a parameter-matched comparison.
Native Mamba scan dispatch is enabled. Mamba-2 uses its chunked SSD path.

The streaming batch seed now reaches the loader. Previously it fell back to
the system clock. Before training, the runner checks that repeating a seed
repeats the first two batches and that another seed changes them. The sweep
also checks that those batch fingerprints match across architectures for each
seed. The supplied seed fixes the shuffle order on repeated passes too.

The manifest records source/BEAM hashes, replay hashes, arguments, losses,
parameter counts, and GPU contention. GPU memory snapshots are process-resident
memory from `nvidia-smi`, not peak live tensor memory. Policy files stay local.
The independent eight-game smoke directory predates the seed fix and is not
part of the comparison.

Run from ExPhil's own environment:

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/sweep_mamba_learning.exs --out eval_runs/mamba_learning_rerun
```

Progress goes to each arm's `.log`. Completed arms are skipped only when source
hashes and settings match. An interrupted arm starts fresh; it does not resume
optimizer state. Use a new output directory for a changed experiment.

The checked-in `pilot/` directory contains the original measurements. Its
`source/pipeline.ex` preserves the exact pipeline source hashed during training;
the working source was subsequently formatted. The seed fix is the only
behavioral pipeline change. The archived source is evidence, not a second
implementation to maintain.

Held-out loss is only the first gate. A better bot still needs longer training,
more held-out games, and matches against fixed opponents with repeated seeds.
No trained pilot is deployed or promoted by these scripts.

After training, `scripts/readout_mamba_learning.exs <pilot-directory>` plays
each snapshot against frozen `coh_fox_mamba_master_ep3`. Each pairing has four
parallel environments, 1,800 frames, both ports, Final Destination, zero delay,
and simulator seeds 71/72. Agent sampling remains stochastic and is not seeded
by those simulator seeds. These are eight short trajectories per snapshot,
not eight guaranteed completed matches. The existing league records stocks
and percent at the time limit; report those separately from completed games.
This checks exported-policy compatibility and gives a preliminary gameplay
readout. It cannot certify competitive strength or simulator-to-Dolphin parity.

The focused `StreamingSeedTest` passes against the fixed pipeline on EXLA's
host backend. Removing the new seed line in a separate VM makes the repeated-
batch assertion fail. The experiment also checks the real replay batches and
matches their fingerprints between architectures, beyond the synthetic test.

**AI-generated poem**

The same small field, three seeds of thought;
we count what grows, not what we hoped.
