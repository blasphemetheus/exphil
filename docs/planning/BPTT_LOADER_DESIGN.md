# Contiguous-BPTT Loader — design (v2 plank 1)

Status: 2026-09-03 evening — design complete; planks A+B CODE DRAFTED
with tests, **UNCOMPILED/UNTESTED** (w180 beam live, no mix allowed).
First action when GPU frees: `mix test test/edifice/recurrent/carry_backbone_test.exs`
(in ../edifice) and `mix test test/exphil/training/trajectory_cursors_test.exs`.
Owner: Claude session 09-03, direction approved by Bradley
(HANDOFF_2026-09-03 task 2).

Drafted files:
- `../edifice/lib/edifice/recurrent/recurrent.ex` →
  `build_backbone_with_carry/2` (plank A) +
  `../edifice/test/edifice/recurrent/carry_backbone_test.exs`
  (incl. the chunked==long-unroll equivalence law + param-name
  transplant compatibility)
- `lib/exphil/training/trajectory_cursors.ex` (plank B) +
  `test/exphil/training/trajectory_cursors_test.exs`
- `lib/exphil/training/data.ex` → public `frame_action/1` seam
- Substrate decision taken (least-build default): integrate with the
  existing streaming layer using stream chunks >= batch_size files;
  mmap-corpus remains the upgrade path.
- Unroll default 80 (slippi-ai parity, per handoff wording).

Remaining: plank C (trainer carry state + per-row zero-reset), plank D
(per-timestep AR-head loss), pipeline wiring (--bptt flag), val
protocol. Planks C+D are the deep integration (~1-2 days).

Plank C/D implementation notes (from 09-03 evening reading):
- The AR heads (networks/policy/heads.ex) are entirely `Axon.dense` +
  `Axon.embedding` — BOTH broadcast over leading dims, so applying them
  to the full `{B,T,H}` trunk output with `{B,T}` teacher-forced
  targets yields `{B,T,K}` logits with NO structural change to the
  head graph. Plank D reduces to: (a) a Policy build variant that
  keeps the sequence (skip the last-timestep extraction) on top of
  `build_backbone_with_carry`, (b) a flatten adapter at the loss
  boundary (`{B,T,*} -> {B*T,*}` for logits/targets/frame_weights)
  feeding the existing `Policy.imitation_loss` unchanged.
- Plank C seam: `train_loop.ex` `train_step_standard` destructures the
  batch; thread `carry` as a third element through
  `compute_policy_loss_and_grad` -> `loss_and_grad_fn.(params, states,
  actions, frame_weights, carry)`; the predict container returns
  `%{output, hidden}` — return `hidden` (stop-gradded by construction:
  it's a plain jit arg) alongside grads. Per-row zero-reset happens
  OUTSIDE the jit: `carry = carry * (1 - is_resetting)[:, None, None]`
  before the step (keeps the model graph reset-free).
- `loss_and_grad_fn` arity is fixed per head type at build time — the
  bptt variant is a NEW builder (build_bptt_loss_and_grad_fn), not a
  patch of the existing arms.

## Why

The neutral diagnosis (HANDOFF_2026-09-03 §4.7): temporally-extended
intention (2-5 s arcs) is structurally unrepresentable when every
training sequence is an independent random 60-frame window with a
zero-initialized hidden state. slippi-ai's BC — the existence proof of
human-looking play at this task — trains with forward-carried recurrent
state over whole games. This is one of the four load-bearing
divergences from the parity audit (contiguous-BPTT, data scale, name
conditioning, T=1.0 decode).

The window-180 arm (running tonight as `w180`) is the cheap
discriminator: if 3 s of context moves neutral instruments, this build
is confirmed before it lands.

## The reference design (slippi-ai, verified in source 09-03)

`~/git/slippi-ai/slippi_ai/data.py` (TrajectoryManager, ~:520-600):

- One TrajectoryManager per **batch row**. Each holds a cursor
  (`self.frame`) into one flattened game.
- `grab_chunk()`: `needs_reset = needs_game or frame + unroll_length >
  game_len`. On reset it draws the next game from the source iterator
  (skipping games shorter than `unroll_length`). Slice
  `[frame : frame+unroll_length]`, then `self.frame = end -
  self.overlap` — consecutive chunks **overlap by `overlap` frames**
  (delay + 1 in their setup) so the next chunk's targets line up.
- Emits per-frame `is_resetting` (True only at index 0 of a chunk that
  started a new game) and a per-chunk `name` code (identity
  conditioning rides the same path).
- `BatchAccumulator` stacks B rows into fixed buffers (batch dim =
  B independent game-cursors, time dim = unroll_length).

`~/git/slippi-ai/slippi_ai/jax/train_lib.py` (:68-72, :143-144):

- The Learner owns `self.hidden_state`, initialized once as
  `learner.initial_state(batch_size)`.
- Every step: `learner_stats, self.hidden_state =
  learner.step(frames, self.hidden_state)` — the **final hidden state
  of chunk k is the initial state of chunk k+1**, for the same batch
  rows. Gradients truncate at the chunk edge (state passed between
  steps is data, not graph).
- `fetch_batch` asserts `is_resetting` is False everywhere except
  t=0 ("Unexpected mid-episode reset") — resets happen only at chunk
  starts, i.e. game boundaries land exactly on chunk boundaries by
  construction.

`~/git/slippi-ai/slippi_ai/jax/policies.py` (:80-201):

- `network.unroll(inputs, is_resetting[:-1], initial_state)` — the
  network zeroes each row's carry where `is_resetting[t=0]` is True
  (per-row select between carried state and initial_state), then
  unrolls.
- Delay handling inside `imitation_loss`: chunk length is
  `unroll + delay + 1`; states `[0, U-1]` predict actions
  `[D+1, U+D]`; overlap = D+1 keeps targets contiguous across chunks.

## Design requirements for ExPhil

1. **Loader**: N=batch_size cursors, each walking one replay's frames
   in order; per-chunk emit {embedded window, controller targets,
   is_resetting flag per row}; on game end, cursor draws the next
   replay (shuffled queue, reshuffled per epoch); short replays
   skipped or padded-and-reset.
2. **State carry**: trainer holds `{B, hidden}` GRU carry per layer;
   train step takes it as input, returns final carry; rows with
   is_resetting get zero (or learned initial) state before unroll.
   Gradients stop at the chunk edge (`Nx.Defn.Kernel.stop_grad` on the
   incoming carry, or simply pass it as a plain input — it's already
   detached if it never enters value_and_grad as a differentiated arg).
3. **Boundary law**: state resets ONLY at real game boundaries — never
   at chunk edges, never at streaming-chunk (file-group) edges. If the
   file-chunk streaming layer forces cursor pools to drain per chunk,
   the carry for a row must survive across... [OPEN: interacts with
   --stream-chunk-size; see mapping]
4. **One knob**: first run on the EXISTING corpus, architecture
   unchanged (GRU), same hyperparams as v1.3/v1.4 — isolate the
   context lever.
5. **Chunk length**: slippi-ai uses unroll 80 (per handoff). Overlap
   must cover frame_delay + 1 for target alignment (ExPhil currently
   trains d0 local — overlap 1).
6. **Shuffle semantics change**: batch rows are no longer i.i.d.
   windows; consecutive batches are correlated in time. slippi-ai
   accepts this (it's the point). Val loss comparisons vs windowed
   baselines must note the different sampling distribution.

## ExPhil mapping (Explore agent, 09-03 — file:line verified)

**Streaming loop**: `Pipeline.setup!` → `Trainer.fit` calls
`Pipeline.batch_stream` fresh each epoch (trainer.ex:233);
`batch_stream_streaming` (pipeline.ex:975-1113) flat-maps file chunks
through `Streaming.parse_chunk` → `create_dataset` →
`Data.batched_sequences`. Trainer params/optimizer thread through the
flat stream — chunks are invisible to the trainer. ChunkPipeline
(chunk_pipeline.ex:87-226) is the async-prefetch variant of the same
(file-I/O chunks, nothing temporal).

**Batching** (data.ex): lazy path `batched_sequences_lazy` (:2368)
slices windows at `i*stride` off one flat `{num_frames, embed_dim}`
tensor. Batch = `%{states: {B,W,D}, actions: last-frame targets,
frame_weights: {B}}`.

**Three load-bearing discoveries:**

1. **Supervision is ONE LABEL PER WINDOW** — targets come from only
   the last frame (`frame_idx = i*stride + window_size - 1`,
   data.ex:2579). slippi-ai supervises EVERY timestep. Under
   contiguous cursors with unroll 80, last-frame-only supervision
   would be 1 label / 80 frames / row — massively less signal per
   compute than today's stride-5 windows. **Per-timestep supervision
   is effectively part of this build**, not optional: heads must be
   applied at every timestep ({B*T, features}) and loss reduced over
   T with per-frame weights (the neutral/transition weighting logic
   at data.ex:2562-2574 generalizes to per-timestep directly).
2. **The GRU unroll can neither accept an initial carry nor return a
   final one** — Edifice.Recurrent's Axon path does
   `Axon.gru(...) |> {output_seq, _hidden}` discarding the carry
   (edifice recurrent.ex:348-370); the fused CUDA path
   (`FusedScan.gru_scan`) takes no h0 at all (fused_scan.ex:318-332).
   A stateful step API exists but is inference-only
   (`Edifice.Recurrent.init_state/step`, used by agent.ex:1507).
   **Plank A is an Edifice change**: a build mode with an
   `"initial_hidden"` input `{B, num_layers, hidden}` and a container
   output `{output_seq, final_hidden}` (Axon.gru accepts an initial
   carry node; the fused kernel's fallback signature already has an
   h0 slot, passed nil today).
3. **Windows already cross replay boundaries** — nothing in the
   windowing path consults game boundaries (no segment info survives
   parsing; per-file counts are computed then discarded,
   streaming.ex:124→:135). Today's training sees teleporting states
   inside windows near boundaries. See GOTCHA #109. Boundary recovery
   exists as a pattern: frame-number reset (`cur <= prev_frame`,
   advantage_weighting.ex:214).

**No hidden-state seam in the train step**: train_loop.ex
destructures only `%{states, actions}`; loss fns call
`predict_fn.(params, states)` — fixed arity, single model input
`"state_sequence"`. Plank B threads `carry` in/out of
`Imitation.train_step` and the loss/grad build.

## Implementation plan (planks, in order)

- **A. Edifice**: `Recurrent.build(carry: true)` → model inputs
  `{"state_sequence", "initial_hidden"}`, output
  `%{output: seq, hidden: final}`. Axon path first (correctness);
  fused-scan h0 second (speed). Unit: final hidden of chunk k fed to
  chunk k+1 == one long unroll (the equivalence test).
- **B. Loader**: `ExPhil.Training.TrajectoryCursors` — takes an
  embedded chunk + segment boundaries (recovered by frame-number
  reset at parse time and THREADED THROUGH — stop discarding per-file
  counts), holds B cursors, emits `{states {B,U,D}, per-timestep
  targets, per-timestep weights, is_resetting {B}}`. Overlap =
  frame_delay + 1 (d0 ⇒ 1).
- **C. Trainer**: carry lives in TrainingState; per-row zero-reset via
  `is_resetting` select before the step; carry is a plain (stop-grad)
  input — gradient truncation falls out for free.
- **D. Loss**: per-timestep AR-head loss ({B*T} flattening), reuse
  focal/neutral/transition weighting per frame.

## Open decisions (for Bradley)

- **Batch size vs concurrent games**: B=256 cursors need >=256
  concurrent games, but a 100-file stream chunk holds ~100. Options:
  (a) bigger chunks (~300 files ≈ 5 GB embedded — fits), (b) smaller
  B for BPTT runs (slippi-ai uses modest batches), (c) the mmap
  corpus substrate (mmap_corpus.ex exists, 08-08) so cursors walk the
  whole corpus with no chunk constraint — cleanest, most build.
- Unroll length (80 = slippi-ai parity; 60/180 = comparability with
  our arms).
- Zero vs learned initial state (Edifice Axon layout currently
  glorot-samples initial hidden from an RNG key param — a quirk to
  kill in carry mode).
- Chunk-edge truncation of in-flight games (acceptable first version)
  vs cursors that span stream chunks.
- Val protocol under carried state (held-out cursor pool).
