# Larger Mamba learning comparison

**AI-generated.** Follow-up to the November 1 pilot. No policy promotion is
automatic. Scripts, settings and results belong to this experiment; the earlier
pilot remains unchanged.

Compare existing Mamba-1/state 16 with canonical Mamba-2/state 16, fresh weights,
seeds 905 and 906, 256 existing master-corpus replay files. Hold out 24 whole
games by the existing path-hash split; train on the other 232. Reject overlapping
file contents between training and validation. Both models receive the same
examples in the same order for each seed. Reverse model order on the second seed.

Keep the first pilot's Fox controller recipe, width 512, two layers, window 80,
batch 128, f32, constant learning rate 0.0002. Save snapshots at 3,000 updates and
600 seconds of training. The timer includes batch loading and the first training
compilation, but excludes preparation, validation and exports. Record first-step
time separately. GPU memory readings are allocator-resident snapshots, not peak
live tensor memory. Desktop GPU use can remain present.

Disk space at start: 3.3 GB. Reuse replays without copying them. Prepared training
chunks and validation batches are cached in a private directory on `/dev/shm`
(RAM-backed storage), keyed by replay contents, data settings and source hashes.
Tensors are on BinaryBackend before serialization. Only policy exports, JSON
and logs go on disk. The sweep refuses to start another arm below 1 GiB free.
The cache can be deleted after the experiment; source replays are never deleted.

Run from ExPhil's own devenv, with no competing GPU compute job:

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/sweep_mamba_learning.exs \
  --out eval_runs/mamba_learning_20261105/comparison \
  --models mamba:16,mamba2_full:16 --seeds 905,906 \
  --files 256 --holdout 24 --steps 3000 --seconds 600 \
  --prepared-cache /dev/shm/exphil-mamba-learning-20261105
```

The source-hash guard requires a new output directory if code changes. Finished
arms can be resumed without repeating them; an interrupted arm restarts fresh.
No optimizer checkpoints are saved in this bounded comparison.

Gameplay uses the equal-time snapshots: frozen master-ep3 opponent, plus direct
Mamba-1/Mamba-2 matches paired by training seed. Each pairing uses four parallel
environments, 10,800 frames (three minutes) per port, both ports, Final Destination,
zero delay, simulator seeds 71/72. Policy sampling is stochastic and is not
seeded by simulator seeds. This tests the existing execution contract, using
windowed inference for these non-BPTT policies. The Rust parity simulator is not
a substitute for final Dolphin testing. Completed games and unfinished scores
are reported separately.

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/readout_mamba_learning.exs eval_runs/mamba_learning_20261105/comparison \
  --frames 10800 --envs 4 --budgets equal_time --direct
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/summarize_mamba_learning.exs eval_runs/mamba_learning_20261105/comparison
```

**AI-generated poem**

Give each learner broader ground;
let longer games test what was found.
