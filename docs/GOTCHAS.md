# Technical Gotchas

Hard-won knowledge from debugging ExPhil. Each section documents a specific issue, symptoms, and the fix.

## Table of Contents

1. [Polaris.Updates.apply_updates nil issue](#1-polarisupdatesapply_updates-nil-issue)
2. [Axon training vs inference mode](#2-axon-training-vs-inference-mode)
3. [Nx.to_number with {1} shaped tensors](#3-nxto_number-with-1-shaped-tensors)
4. [Axon.ModelState deprecation](#4-axonmodelstate-deprecation)
5. [JIT compilation time](#5-jit-compilation-time)
6. [EXLA Backend Configuration](#6-exla-backend-configuration)
7. [EXLA/Defn.Expr tensor mismatch in closures](#7-exladefnexpr-tensor-mismatch-in-closures)
8. [Elixir struct pattern matching order](#8-elixir-struct-pattern-matching-order)
9. [EXLA tensor serialization in checkpoints](#9-exla-tensor-serialization-in-checkpoints)
10. [Policy architecture mismatch on load](#10-policy-architecture-mismatch-on-load)
11. [Dynamic sequence length causes infinite JIT](#11-dynamic-sequence-length-causes-infinite-jit)
12. [Pre-computed tensors in Axon.nx closures](#12-pre-computed-tensors-in-axonnx-closures)
13. [Mix stale builds](#13-mix-stale-builds)
14. [BinaryBackend timeouts in tests](#14-binarybackend-timeouts-in-tests)
15. [RTX 5090 (Blackwell) not supported by EXLA](#15-rtx-5090-blackwell-not-supported-by-exla)
16. [Noisy XLA/CUDA output during training](#16-noisy-xlacuda-output-during-training)
17. [Polaris.Schedules incompatibility with Nx 0.10](#17-polarisschedules-incompatibility-with-nx-010)
18. [Nx.to_number in training loop blocks GPU](#18-nxto_number-in-training-loop-blocks-gpu)
19. [deep_backend_copy copies all model params every batch](#19-deep_backend_copy-copies-all-model-params-every-batch)
20. [Embedding config() vs default_config() dimension mismatch](#20-embedding-config-vs-default_config-dimension-mismatch)
21. [embed_states_fast missing features vs embedding_size](#21-embed_states_fast-missing-features-vs-embedding_size)
22. [Precomputed embeddings require copying BOTH states AND actions](#22-precomputed-embeddings-require-copying-both-states-and-actions)
23. [Registry sanitize_config expects map but receives keyword list](#23-registry-sanitize_config-expects-map-but-receives-keyword-list)
24. [Peppi uses external character IDs, not internal Melee IDs](#24-peppi-uses-external-character-ids-not-internal-melee-ids)
25. [GPU OOM during embedding precomputation](#25-gpu-oom-during-embedding-precomputation)
26. [Streaming mode requires --no-prefetch](#26-streaming-mode-requires---no-prefetch)
27. [Streaming mode auto-disables precompute](#27-streaming-mode-auto-disables-precompute)
28. [--train-character requires metadata collection](#28---train-character-requires-metadata-collection)
29. [Jamba/Temporal model architecture mismatch on load](#29-jambatemporal-model-architecture-mismatch-on-load)
30. [Model outputs nonsensical actions](#30-model-outputs-nonsensical-actions-eg-always-rolls-right)
31. [Polaris composed optimizer state structure](#31-polaris-composed-optimizer-state-structure)
32. [Prefetcher deadlock with EXLA tensors in spawned processes](#32-prefetcher-deadlock-with-exla-tensors-in-spawned-processes)
33. [LSTM/GRU training is inherently slow (not a bug)](#33-lstmgru-training-is-inherently-slow-not-a-bug)
34. [Imitation.new uses window_size, not seq_len](#34-imitationnew-uses-window_size-not-seq_len)
35. [Prefetcher materializes all batches into memory](#35-prefetcher-materializes-all-batches-into-memory)
36. [Embedding pre-computation exhausts RAM on large datasets](#36-embedding-pre-computation-exhausts-ram-on-large-datasets)
37. [0% GPU utilization with pre-computed embeddings](#37-0-gpu-utilization-with-pre-computed-embeddings)
38. [Streaming prefetcher timeout causes silent batch drops](#38-streaming-prefetcher-timeout-causes-silent-batch-drops)
39. [XLA preallocates 90% GPU memory and never releases it](#39-xla-preallocates-90-gpu-memory-and-never-releases-it)
40. [--augment flag bypasses precomputed embeddings (100x slower)](#40---augment-flag-bypasses-precomputed-embeddings-100x-slower)
41. [--cache-embeddings flag parsed but not used](#41---cache-embeddings-flag-parsed-but-not-used-train_from_replaysexs)
42. [Nx.to_number returns atoms for special float values](#42-nxto_number-returns-atoms-for-special-float-values-nan-infinity)
43. [Precomputed embeddings on CPU cause slow Nx.take (~17s/batch)](#43-precomputed-embeddings-on-cpu-cause-slow-nxtake-17sbatch)
44. [Optimizer not JIT compiled with EXLA causes 0% GPU utilization](#44-optimizer-not-jit-compiled-with-exla-causes-0-gpu-utilization)
45. [O(n) list traversal for action collection causes ~28s/batch](#45-on-list-traversal-for-action-collection-causes-28sbatch)
46. [LSTM/GRU gradient explosion causes NaN loss mid-training](#46-lstmgru-gradient-explosion-causes-nan-loss-mid-training)

---

## 1. Polaris.Updates.apply_updates nil issue

`Polaris.Updates.apply_updates/2` is a `defn` with a default nil parameter (`state \\ nil`).
In Nx 0.10.0, calling it directly fails during lazy container traversal.

**Fix:** Wrap with `Nx.Defn.jit/1`:
```elixir
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)
new_params = apply_updates_fn.(params, updates)
```

---

## 2. Axon training vs inference mode

- `mode: :train` returns `%{prediction: ..., state: ...}` for stateful layers (dropout)
- `mode: :inference` returns predictions directly
- For gradient computation, use `mode: :inference` to avoid pattern matching complexity

---

## 3. Nx.to_number with {1} shaped tensors

`Nx.to_number/1` only works on scalar tensors (shape `{}`). When `Nx.slice` returns
a single element, it has shape `{1}` which must be squeezed first:

```elixir
tensor |> Nx.squeeze() |> Nx.to_number()
```

---

## 4. Axon.ModelState deprecation

Pass full `%Axon.ModelState{}` to predict functions, not just the `.data` map:

```elixir
# Good
predict_fn.(model_state, input)

# Deprecated (triggers warning)
predict_fn.(model_state.data, input)
```

---

## 5. JIT compilation time

First batch takes 2-5 minutes on CPU for large models (1991 input dims).
This is normal - subsequent batches are fast after compilation.

---

## 6. EXLA Backend Configuration

EXLA provides ~2000x speedup over Nx.BinaryBackend. Configured in `config/config.exs`:

```elixir
config :nx, default_backend: EXLA.Backend
config :exla, default_client: :host  # CPU with XLA optimizations
```

For CUDA GPU support (if available):
```elixir
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

---

## 7. EXLA/Defn.Expr tensor mismatch in closures

When using `Nx.Defn.value_and_grad` with closures that capture tensors, you get:
```
cannot invoke Nx function because it relies on two incompatible tensor implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Copy ALL captured tensors before using them in gradient computation:
```elixir
# Copy batch data
states = Nx.backend_copy(states)
actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

# Also copy model parameters to avoid closure capture issues
model_state = deep_backend_copy(trainer.policy_params)
```

---

## 8. Elixir struct pattern matching order

Structs are maps, so `is_map(%Nx.Tensor{})` returns `true`. When using guards with
struct patterns, the struct clause MUST come before the `is_map` guard:

```elixir
# CORRECT order - struct patterns first
defp deep_backend_copy(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor)
defp deep_backend_copy(%Axon.ModelState{data: data} = state) do
  %{state | data: deep_backend_copy(data)}
end
defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
end
defp deep_backend_copy(other), do: other
```

---

## 9. EXLA tensor serialization in checkpoints

EXLA tensors contain device buffer references that are process/session-specific.
When serializing checkpoints with `:erlang.term_to_binary/1`, these buffers become
invalid after the training process ends.

**Fix:** Always convert tensors to `Nx.BinaryBackend` before saving:
```elixir
def save_checkpoint(trainer, path) do
  checkpoint = %{
    policy_params: to_binary_backend(trainer.policy_params),
    optimizer_state: to_binary_backend(trainer.optimizer_state),
    # ... other fields
  }
  File.write(path, :erlang.term_to_binary(checkpoint))
end

defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
defp to_binary_backend(%Axon.ModelState{data: data, state: state} = ms) do
  %{ms | data: to_binary_backend(data), state: to_binary_backend(state)}
end
defp to_binary_backend(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
end
defp to_binary_backend(other), do: other
```

**Symptoms of stale EXLA buffers:**
- `ArgumentError: unable to get buffer. It may belong to another node`
- `ArgumentError: decode failed, none of the variant types could be decoded`

---

## 10. Policy architecture mismatch on load

When loading a policy, the model architecture must match exactly. If training used
`hidden_sizes: [64, 64]` but loading uses default `[512, 512]`, you get shape errors.

**Fix:** Always save and restore `hidden_sizes`, `embed_size`, `dropout` in policy config:
```elixir
# In export_policy
config = %{
  embed_size: trainer.config[:embed_size],
  hidden_sizes: trainer.config[:hidden_sizes],
  dropout: trainer.config[:dropout],
  # ... other config
}
```

---

## 11. Dynamic sequence length causes infinite JIT

When building attention models with `shape: {nil, nil, embed_size}` (dynamic batch AND
sequence length), XLA tries to compile kernels that handle arbitrary sequence lengths.
This can take 30+ minutes or never complete.

**Symptoms:**
- Training process stuck at "Step 4: Training..." with high CPU, no progress
- JIT compilation takes >10 minutes for attention models

**Fix:** Use concrete sequence length (`:seq_len` option) in attention model builders:
```elixir
# GOOD - concrete seq_len (fast JIT, ~2-3 minutes)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: 30  # Defaults to window_size, which is correct for training
)

# BAD - dynamic seq_len (can hang indefinitely)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: nil  # Forces dynamic shape - avoid for training
)
```

The attention modules now default `seq_len` to `window_size` for training efficiency.
For inference with variable-length sequences, pass `seq_len: nil` explicitly and
expect longer JIT compilation on first run.

---

## 12. Pre-computed tensors in Axon.nx closures

When pre-computing tensors (like attention masks) that are captured in `Axon.nx`
closures, they must be in `Nx.BinaryBackend` to avoid EXLA/Defn.Expr mismatch.

**Symptom:**
```
cannot invoke Nx function because it relies on two incompatible tensor
implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Convert pre-computed tensors to BinaryBackend before capture:
```elixir
# GOOD - explicit BinaryBackend
mask = window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
Axon.nx(input, fn tensor ->
  # mask is now safely inlined into the defn expression
  scaled_dot_product_attention(query, key, value, mask: mask)
end)

# BAD - EXLA tensor captured in closure
mask = window_mask(seq_len, window_size)  # Uses default EXLA backend
Axon.nx(input, fn tensor ->
  # CRASH: mask is EXLA but tensor is Defn.Expr during tracing
  scaled_dot_product_attention(query, key, value, mask: mask)
end)
```

---

## 13. Mix stale builds

Mix uses file timestamps to decide what to recompile. Sometimes edits don't trigger
recompilation (editor buffering, clock drift, etc.), causing old bytecode to run.

**Symptoms:**
- "I fixed the warning but it still appears"
- Code changes have no effect
- Training runs old version of the code

**Fix:** Before important runs, force full recompilation:
```bash
# Option 1: Force recompile (faster)
mix compile --force

# Option 2: Clean build (slower but guaranteed clean)
rm -rf _build && mix compile
```

**Prevention:** Add to your training workflow:
```bash
# Recommended: Always compile before training
mix compile --force && mix run scripts/train_from_replays.exs --epochs 10
```

---

## 14. BinaryBackend timeouts in tests

**Symptom:** Test times out after 60000ms with stack trace showing `Nx.BinaryBackend.bin_dot`

```
** (ExUnit.TimeoutError) test timed out after 60000ms
stacktrace:
  (nx 0.10.0) lib/nx/binary_backend.ex:551: anonymous fn/5 in Nx.BinaryBackend.bin_dot/5
```

**Cause:** Tests using BinaryBackend (the pure Elixir backend) for matrix operations are extremely slow compared to EXLA. Large model tests can easily exceed the 60s default timeout.

**Fix options:**

1. Add timeout tag to slow tests:
```elixir
@tag timeout: 120_000  # 2 minutes
test "builds large model" do
  # ...
end
```

2. Run with `--trace` (sets infinite timeout):
```bash
mix test test/slow_test.exs --trace
```

3. Use EXLA in tests when available:
```elixir
setup do
  Nx.default_backend(EXLA.Backend)
  :ok
end
```

**Note:** These timeouts are flaky - they depend on system load. The test itself is correct; it just needs more time on BinaryBackend.

---

## 15. RTX 5090 (Blackwell) not supported by EXLA

**As of January 2025**, the RTX 5090 (Blackwell architecture, Compute Capability 12.0) is not supported by EXLA/XLA.

**Symptoms:**
```
RuntimeError: /usr/local/cuda/bin/ptxas ptxas too old. Falling back to the driver to compile.

Falling back to the CUDA driver for PTX compilation; ptxas does not support CC 12.0
```

Or:
```
RuntimeError: No PTX compilation provider is available. Neither ptxas/nvlink nor nvjtlink is available.
```

**Root cause:** The RTX 5090 shipped January 2025 with a new Compute Capability (12.0). NVIDIA's CUDA toolkit versions lag behind new GPU architectures:
- CUDA 12.2: Does not support CC 12.0
- CUDA 12.6: Partial support, ptxas cannot compile for CC 12.0
- The XLA flag `--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found` does not help because EXLA throws the error before the fallback takes effect

**What we tried that didn't work:**
1. Upgrading to CUDA 12.6.3 base image
2. Using `devel` image (includes ptxas) instead of `runtime`
3. XLA flags: `XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"`

**Solution:** Use RTX 4090 or older GPUs that are fully supported:
- RTX 4090 (Ada Lovelace, CC 8.9) - **Recommended**, works perfectly with CUDA 12.6
- RTX 3090 (Ampere, CC 8.6) - Budget option, fully supported
- A100/H100 - Overkill for small models but fully supported

**Future:** Blackwell support will likely come with:
- CUDA 13.x release (expected mid-2025)
- Updated EXLA precompiled binaries
- XLA updates for CC 12.0 PTX generation

**Cost comparison (RunPod):**
| GPU | $/hr | Status |
|-----|------|--------|
| RTX 4090 | $0.34 | ✅ Works |
| RTX 5090 | $0.70 | ❌ Not supported |

The RTX 4090 is actually better value for ExPhil's model size (24GB VRAM is plenty).

---

## 16. Noisy XLA/CUDA output during training

Training produces lots of informational messages from XLA, CUDA, and cuDNN that clutter output.

**Symptoms:**
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1768939388.742377 se_gpu_pjrt_client.cc:1101] Using BFC allocator.
All configs were filtered out because none of them sufficiently match the hints...
ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_3'...
XLA service 0x797951898c70 initialized for platform CUDA...
```

**These are harmless:**
- **absl::InitializeLog warning**: Google logging library init message
- **BFC allocator**: XLA memory allocator initialization (normal)
- **"configs filtered out"**: EXLA autotuning falling back to full search (normal)
- **ptxas register spill**: GPU compiler couldn't fit all vars in registers (minor perf impact)
- **XLA service initialized**: Confirmation that GPU is working

**Suppression options:**

1. **XLA/TensorFlow C++ logs** (`I0000`, `WARNING: All log messages before absl`):
   ```bash
   # Suppress all TF/XLA C++ logs (0=all, 1=info, 2=warning, 3=error)
   TF_CPP_MIN_LOG_LEVEL=3 mix run scripts/train_from_replays.exs
   ```

2. **Elixir Logger info messages** (`[info] XLA service initialized`):
   ```elixir
   # In config/runtime.exs or at script start:
   Logger.configure(level: :warning)
   ```
   Or via environment:
   ```bash
   ELIXIR_LOG_LEVEL=warning mix run scripts/train_from_replays.exs
   ```

3. **Combined suppression (recommended for clean output):**
   ```bash
   TF_CPP_MIN_LOG_LEVEL=2 mix run scripts/train_from_replays.exs 2>&1 | \
     grep -v "ptxas warning\|configs were filtered\|BFC allocator\|XLA backend"
   ```

4. **Add to your shell profile for permanent suppression:**
   ```bash
   # In ~/.bashrc or ~/.zshrc
   export TF_CPP_MIN_LOG_LEVEL=2
   ```

5. **Docker/RunPod - add to run command:**
   ```bash
   docker run -e TF_CPP_MIN_LOG_LEVEL=2 ...
   ```

**What each level suppresses:**
| TF_CPP_MIN_LOG_LEVEL | Suppresses |
|---------------------|------------|
| 0 | Nothing (default) |
| 1 | INFO messages |
| 2 | INFO + WARNING |
| 3 | INFO + WARNING + ERROR |

**Note:** Level 2 is usually best - keeps errors visible while hiding noise.

**Why these appear:** XLA (the compiler backend EXLA uses) comes from TensorFlow/JAX and uses Google's logging infrastructure. These are initialization confirmations, not problems.

**Best practice:** The messages don't affect training. Use `TF_CPP_MIN_LOG_LEVEL=2` for cleaner output while keeping actual errors visible.

---

## 17. Polaris.Schedules incompatibility with Nx 0.10

Polaris 0.1.0's schedule functions don't work with Nx 0.10 when `warmup_steps` is 0.

**Symptoms:**
```elixir
** (FunctionClauseError) no function clause matching in Nx.apply_vectorized/2
    The following arguments were given to Nx.apply_vectorized/2:
        # 1
        [init_value: 0.0001]
```

**Root cause:** Polaris schedules return a keyword list `[init_value: lr]` instead of a proper schedule function when certain parameters are 0.

**Fix in ExPhil:** We replaced Polaris.Schedules with direct Nx implementations in `lib/exphil/training/imitation.ex`. The default `warmup_steps` is set to 1 (not 0) to avoid triggering the bug.

**If you see this error:** Ensure you're using the latest code with `--warmup-steps 1` or rebuild the Docker image.

---

## 18. Nx.to_number in training loop blocks GPU

Calling `Nx.to_number(loss)` after every training batch causes severe GPU underutilization (0% GPU compute, 90% VRAM).

**Symptoms:**
- GPU utilization 0% despite 90% VRAM usage
- Training takes 100x longer than expected (hours instead of minutes)
- ~80s per batch instead of ~0.2s per batch
- CPU utilization also low (6-10%)

**Root cause:** `Nx.to_number/1` forces a synchronous GPU→CPU transfer. When called after every batch:

```
GPU computes batch → GPU IDLE (waiting) → CPU reads loss → Next batch
                     ↑
              This idle time dominates training
```

The GPU finishes computing quickly but sits idle while waiting for the CPU to fetch the scalar loss value.

**Common sources of this bug:**
1. Progress display converting loss every batch for display
2. Logging/metrics collection converting every batch
3. Accumulating numbers instead of tensors

**Fix:** Keep losses as tensors during training, convert only at epoch end:

```elixir
# BAD - blocks after every batch
{new_trainer, losses} = Enum.reduce(batches, {trainer, []}, fn batch, {t, ls} ->
  {new_t, m} = Imitation.train_step(t, batch, nil)
  {new_t, [Nx.to_number(m.loss) | ls]}  # ← BLOCKS HERE
end)

# GOOD - single GPU→CPU transfer at epoch end
{new_trainer, losses} = Enum.reduce(batches, {trainer, []}, fn batch, {t, ls} ->
  {new_t, m} = Imitation.train_step(t, batch, nil)
  {new_t, [m.loss | ls]}  # loss stays as tensor
end)
avg_loss = losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()  # One transfer
```

**Progress display fix (January 2025):** The training script's progress bar was calling `Nx.to_number(metrics.loss)` every batch to display the current loss. Fixed by:
1. Only converting to number every 50 batches for display
2. Storing `{display_loss, tensor}` tuples in accumulator
3. Computing epoch mean from tensors at epoch end

```elixir
# For progress display: convert only periodically
display_loss = if rem(batch_idx, 50) == 0 do
  Nx.to_number(metrics.loss)
else
  previous_display_loss  # Reuse cached value
end

# Accumulate tensor, convert at epoch end
{new_trainer, [{display_loss, metrics.loss} | losses], ...}
```

**Related functions affected:**
- `Imitation.train_step/3` - now returns `loss` as tensor
- `Imitation.evaluate_batch/2` - returns tensor for accumulation
- `Imitation.evaluate/2` - handles conversion internally
- `scripts/train_from_replays.exs` - progress display fixed

**Regression tests:** See `ImitationTest` for tests tagged `:regression` that verify tensor accumulation works correctly.

---

## 19. deep_backend_copy copies all model params every batch

**Status:** FIXED

Previously, `train_step/3` called `deep_backend_copy(trainer.policy_params)` every batch, copying all model parameters (~2M for typical models). This was a workaround for EXLA/Defn.Expr tensor mismatch (see Gotcha #7).

**The problem (old code):**
```elixir
# In train_step - COPIES ALL PARAMS EVERY BATCH
model_state = deep_backend_copy(trainer.policy_params)
loss_fn = fn params ->
  predict_fn.(params, states)  # states is captured in closure!
  # ...
end
{loss, grads} = Nx.Defn.value_and_grad(loss_fn).(model_state)
```

**The fix:** Build the loss+grad function ONCE in `new/1` with `predict_fn` and config captured, then call it with explicit arguments in `train_step`:

```elixir
# In new/1 - build once, capture predict_fn and config
loss_and_grad_fn = build_loss_and_grad_fn(predict_fn, config)

# In train_step - NO COPYING, just call with explicit args
{loss, grads} = trainer.loss_and_grad_fn.(trainer.policy_params, states, actions)
```

**Impact:** ~20-30% speedup by eliminating per-batch parameter copies and closure creation.

---

## 20. Embedding config() vs default_config() dimension mismatch

**Status:** FIXED

`Embeddings.config()` and `Embeddings.default_config()` had different default values, causing a dimension mismatch where the model expected one size but received embedded tensors of a different size.

**Symptoms:**
```
** (Axon.CompileError) exception found when compiling layer Axon.Layers.dense/4:
    ** (ArgumentError) dot/zip expects shapes to be compatible,
    dimension 1 of left-side (1175) does not equal dimension 0 of right-side (1204)
```

**The problem:**
```elixir
# Player struct defaults
defstruct [
  with_speeds: true,        # struct default
  # ...
]

# But Embeddings.config() had different hardcoded defaults!
def config(opts \\ []) do
  player_config = %Player{
    with_speeds: Keyword.get(opts, :with_speeds, false),  # DIFFERENT!
    # ...
  }
end

# This caused:
Embeddings.embedding_size()                    # => 1204 (uses struct defaults)
Embeddings.embedding_size(Embeddings.config()) # => 1175 (uses config defaults)
```

**The fix:** Make `config()` start from `default_config()` and merge overrides:

```elixir
def config(opts \\ []) do
  base = default_config()

  player_opts = Keyword.take(opts, [:with_speeds, :with_nana, ...])
  player_config = struct(base.player, player_opts)

  # ... merge other options
  struct(base, [{:player, player_config} | game_opts])
end
```

**Prevention tests:** Added tests that verify:
- `embedding_size()` matches `embedding_size(config())`
- `config()` fields match `default_config()` fields
- Actual embedded tensor size matches `embedding_size(config)`

**Lesson:** When you have multiple ways to create a default config, ensure they produce identical results. Add tests that catch dimension mismatches by comparing sizes from different code paths.

---

## 21. embed_states_fast missing features vs embedding_size

**Status:** FIXED

The batch embedding function `embed_states_fast/3` was missing projectile embedding, causing a dimension mismatch when `with_projectiles: true`.

**Symptoms:**
```
** (Axon.CompileError) dot/zip expects shapes to be compatible,
dimension 1 of left-side (1169) does not equal dimension 0 of right-side (1204)
```

The difference (35 dims) = 5 projectiles × 7 dims/projectile.

**How this happened:**
1. `embedding_size()` correctly counted projectile dimensions
2. Single-frame `embed/4` correctly embedded projectiles
3. Batch `embed_states_fast/3` was added later for training performance
4. When adding `embed_states_fast`, projectile embedding was forgotten
5. No test compared batch vs single embedding dimensions

**Root cause:** When implementing an optimized batch version of an embedding function, it's easy to forget optional features (projectiles, items, etc.) that are conditionally included.

**The fix:** Added `embed_batch_projectiles/2` to handle batch projectile embedding:

```elixir
# In embed_states_fast, after frame_count:
embs_with_projectiles = if config.with_projectiles do
  proj_emb = embed_batch_projectiles(game_states, config)
  embs_with_frame ++ [proj_emb]
else
  embs_with_frame
end
```

**Prevention tests:** Added tests in `game_test.exs`:
- `embed_states_fast produces embedding matching embedding_size()`
- `embed_states_fast matches single embed() output dimensions`
- `embed_states_fast handles projectiles correctly`

**Lesson:** When implementing batch/optimized versions of functions:
1. Always test that output dimensions match the original
2. Check all conditional features (`if config.with_X`) are present in both
3. Compare against `embedding_size(config)` which is the source of truth

---

## 22. Precomputed embeddings require copying BOTH states AND actions

**Status:** FIXED

When using precomputed embeddings for training (via `embed_states_fast`), both `states` AND `actions` tensors are EXLA tensors that get captured in the loss function closure passed to `value_and_grad`. Both need `Nx.backend_copy`.

**Symptoms:**
```
** (RuntimeError) cannot invoke Nx function because it relies on two incompatible
tensor implementations: Nx.Defn.Expr and EXLA.Backend
    (exphil 0.1.0) lib/exphil/networks/policy.ex:981: ExPhil.Networks.Policy.binary_cross_entropy/3
    (exphil 0.1.0) lib/exphil/networks/policy.ex:930: ExPhil.Networks.Policy.imitation_loss/3
```

**How this happened:**
1. Training script precomputes embeddings as EXLA tensors for efficiency
2. `build_loss_and_grad_fn` creates a closure that captures `states` and `actions`
3. Initial fix added `Nx.backend_copy(states)` to avoid the mismatch
4. But `actions` is a map of multiple EXLA tensors (buttons, main_x, etc.)
5. The mismatch moved from states to actions, appearing in `binary_cross_entropy`

**The fix:** Copy both states AND actions in `build_loss_and_grad_fn`:

```elixir
fn params, states, actions ->
  states = Nx.as_type(states, precision)

  # CRITICAL: Copy BOTH states AND actions
  states = Nx.backend_copy(states)
  actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

  loss_fn = fn p ->
    # states and actions are now safe to use in defn context
    {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(p, states)
    Policy.imitation_loss(logits, actions, ...)
  end

  Nx.Defn.value_and_grad(loss_fn).(params)
end
```

**Why this is subtle:** The error message points to `binary_cross_entropy` which is deep in the loss calculation, not at the closure capture site. The stack trace makes it look like a Policy.ex bug when it's actually a data preparation issue.

**Lesson:** When using `Nx.Defn.value_and_grad`, ALL non-parameter tensors captured in the closure must be copied:
1. Look at what variables the loss_fn closure captures
2. Identify which are precomputed EXLA tensors
3. Copy ALL of them, not just the obvious one (`states`)

---

## 23. Registry sanitize_config expects map but receives keyword list

**Status:** FIXED

The training script passes parsed CLI options as a keyword list to the registry, but `sanitize_config/1` only had a clause for maps.

**Symptoms:**
```
** (FunctionClauseError) no function clause matching in ExPhil.Training.Registry.sanitize_config/1
    The following arguments were given to ExPhil.Training.Registry.sanitize_config/1:
        # 1
        [name: "moonwalking_fthrow", checkpoint: "...", ...]
    Attempted function clauses (showing 1 out of 1):
        defp sanitize_config(config) when is_map(config)
```

**Root cause:** Elixir keyword lists and maps are different types. `OptionParser.parse/2` returns keyword lists, but the registry expected a map.

**The fix:** Add a clause to handle keyword lists by converting to map first:

```elixir
defp sanitize_config(config) when is_list(config) do
  # Handle keyword lists by converting to map first
  sanitize_config(Map.new(config))
end

defp sanitize_config(config) when is_map(config) do
  # Remove function values that can't be serialized
  config
  |> Enum.reject(fn {_k, v} -> is_function(v) end)
  |> Enum.reject(fn {_k, v} -> is_pid(v) end)
  |> Enum.into(%{})
end
```

**Lesson:** When writing functions that accept "config" or "options", consider accepting both keyword lists and maps for flexibility. Use `is_list/1` guard for keyword lists and `is_map/1` for maps.

---

## 24. Peppi uses external character IDs, not internal Melee IDs

**Status:** FIXED

The peppi library uses **external character IDs** (Character Select Screen order), not Melee's internal game IDs. This caused character detection to fail for most replays.

**Symptoms:**
- Only ~17% of "Mewtwo replays" detected as Mewtwo
- Other characters showing up in stats for single-character replay sets
- `--train-character` filtering out most replays

**The two ID schemes:**

| Character | External ID (peppi) | Internal ID (game) |
|-----------|--------------------|--------------------|
| Captain Falcon | 0x00 | 0x02 |
| Fox | 0x02 | 0x01 |
| Mewtwo | **0x0A** | **0x10** |
| Ice Climbers | 0x0E | 0x0A |
| Ganondorf | 0x19 | 0x19 |

**How this happened:**
1. Initial character mapping assumed peppi used internal Melee IDs
2. Most characters have different IDs between the two schemes
3. Mewtwo (0x0A external) was being matched against 0x10, returning "Unknown"

**The fix:** Updated `native/exphil_peppi/src/lib.rs` to use external (CSS) character IDs:

```rust
fn character_name(char_id: u8) -> String {
    // Peppi uses "external" character IDs (CSS order)
    match char_id {
        0x00 => "Captain Falcon",
        0x02 => "Fox",
        0x0A => "Mewtwo",      // Was incorrectly 0x10
        0x0E => "Ice Climbers",
        0x19 => "Ganondorf",
        // ... etc
    }
}
```

**How to verify:** After fixing, run with `--train-character mewtwo` and check that all Mewtwo replays are detected.

**Lesson:** When working with Melee data:
1. Always verify which ID scheme a library uses (external vs internal)
2. External IDs = Character Select Screen order
3. Internal IDs = In-game memory order
4. Test with known single-character replay sets to verify mappings

---

## 25. GPU OOM during embedding precomputation

**Status:** FIXED

When precomputing embeddings for large datasets, the GPU runs out of memory because all embeddings accumulate on the GPU before being used.

**Symptoms:**
```
Allocator (GPU_0_bfc) ran out of memory trying to allocate 141.1KiB
If the cause is memory fragmentation maybe the environment variable
'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation.

** (RuntimeError) Out of memory while trying to allocate 144480 bytes.
    (exla 0.10.0) EXLA.NIF.binary_to_device_mem(...)
```

**The math:**
```
175,600 sequences × 30 frames × 1204 dims × 4 bytes = ~25GB
RTX 4090 VRAM = 24GB
```

**Root cause:**
1. `precompute_embeddings` and `precompute_frame_embeddings` process in batches
2. Each batch creates EXLA tensors on GPU
3. Embeddings accumulate in a list, all staying on GPU
4. Eventually exceeds VRAM capacity

**The fix:** Copy each batch to CPU after embedding:

```elixir
# In precompute_embeddings (sequences)
Enum.map(chunk, fn frame ->
  game_states = Enum.map(frame.sequence, & &1.game_state)
  Embeddings.Game.embed_states_fast(game_states, 1, config: embed_config)
  |> Nx.backend_copy(Nx.BinaryBackend)  # CRITICAL: Move to CPU
end)

# In precompute_frame_embeddings (frames)
batch_embedded
|> Nx.to_batched(1)
|> Enum.map(fn t ->
  t |> Nx.squeeze(axes: [0]) |> Nx.backend_copy(Nx.BinaryBackend)
end)
```

**Why `TF_GPU_ALLOCATOR=cuda_malloc_async` didn't help:**
- That setting helps with fragmentation (many small allocations)
- This issue was total memory exhaustion, not fragmentation
- 25GB simply doesn't fit in 24GB regardless of allocation strategy

**Lesson:** When accumulating large amounts of data:
1. Calculate total memory requirement upfront (count × dims × bytes)
2. Compare against available VRAM
3. Use `Nx.backend_copy(Nx.BinaryBackend)` to move data to CPU
4. Data transfers back to GPU per-batch during training (this is the intended flow)

---

## 26. Streaming mode requires --no-prefetch (FIXED)

**Symptom:** With `--stream-chunk-size`, training shows "No batches processed this epoch" even though replays are found.

**Root cause:** The prefetcher used `Stream.each` with a producer process that waited for batch requests. With lazy streaming chunks, the producer blocked waiting for the stream to yield batches, while the main process sent batch requests. This created a deadlock.

**Fix:** The prefetcher was rewritten to eagerly buffer batches from the stream iterator. The producer now spawns a separate process that iterates the stream and queues batches, which the consumer pulls from.

Streaming mode now works with prefetching enabled (default):
```bash
mix run scripts/train_from_replays.exs \
  --stream-chunk-size 30
  # No --no-prefetch needed anymore
```

**Code location:** `lib/exphil/training/prefetcher.ex` - `stream_producer/2` and `stream_producer_loop/4`

---

## 27. Streaming mode auto-disables precompute

**Symptom:** With `--stream-chunk-size`, parsing takes 3-5+ minutes per chunk instead of 30-60 seconds.

**Root cause:** In non-streaming mode, precompute is efficient:
- Compute all embeddings once upfront
- Reuse across all epochs

In streaming mode, precompute is wasteful:
- Compute embeddings for chunk 1, train, **discard**
- Compute embeddings for chunk 2, train, **discard**
- Repeat every epoch

With 5 chunks × 20 epochs = 100 precompute passes instead of 1!

**Fix:** Streaming mode now auto-disables precompute in `Streaming.create_dataset/2`. Embeddings are computed on-the-fly during training instead.

**Code location:** `lib/exphil/training/streaming.ex` - `create_dataset/2` ignores the `:precompute` option.

---

## 28. --train-character requires metadata collection

**Symptom:** With `--train-character mewtwo` and `--stream-chunk-size`, training shows "Found 0 replay files" or "No batches processed" even though replays exist.

**Root cause:** The `--train-character` flag needs to scan replay metadata to find which port has the target character. The metadata collection code was only triggered when:
- `--character` filter was set, OR
- `--stage` filter was set, OR
- File count ≤ 1000

With >1000 files and only `--train-character` (no other filters), metadata collection was skipped, so port selection didn't work.

**Fix:** Added `train_character != nil` to the metadata collection condition:
```elixir
# Before (broken)
if character_filter != [] or stage_filter != [] or initial_count <= 1000 do

# After (fixed)
if character_filter != [] or stage_filter != [] or train_character != nil or initial_count <= 1000 do
```

**Code location:** `scripts/train_from_replays.exs` around line 430.

---

## 29. Jamba/Temporal model architecture mismatch on load

**Symptom:** When evaluating or playing a Jamba/temporal model, you get shape errors like:
```
** (ArgumentError) dot/zip expects shapes to be compatible,
dimension 2 of left-side (128) does not equal dimension 0 of right-side (64)
```

Or the model produces nonsensical output (e.g., always rolling right).

**Root cause:** Jamba models have several architecture-defining parameters that must match between training and inference:
- `head_dim` - dimension per attention head (default 64)
- `num_heads` - number of attention heads (default 4)
- `window_size` - context window length (default 60)
- `num_layers` - number of Mamba/attention layers (default 2)
- `attention_every` - how often attention layers appear (default 3)

If training used non-default values (e.g., `head_dim: 32`, `num_heads: 2`), but evaluation uses defaults, the model architecture won't match the saved weights.

**Example:** Training with `num_heads=2, head_dim=32` creates attention layers with dim=64. If evaluation defaults to `num_heads=4, head_dim=64`, it builds dim=256. The saved weights (expecting 64) crash when fed 256-dim input.

**Fix:** The policy.bin and checkpoint.axon files now save all architecture parameters. The Agent loads them correctly:
```elixir
# Agent.ex loads from config:
num_heads: Map.get(config, :num_heads, 4),
head_dim: Map.get(config, :head_dim, 64),
```

For eval_model.exs, these are now also read from the checkpoint config. If using custom parameters, verify the config was saved correctly:
```bash
mix run -e '
  {:ok, p} = ExPhil.Training.Checkpoint.load_policy("checkpoints/model_policy.bin")
  IO.inspect(p.config)
'
```

**Prevention:**
1. All architecture params now saved in `build_config_json/1`
2. Evaluation scripts read params from checkpoint config
3. Agent.ex reads all params from policy config

---

## 30. Model outputs nonsensical actions (e.g., always rolls right)

**Symptom:** When playing against a model in Dolphin, it performs repetitive useless actions like continuously rolling in one direction, holding shield, or doing nothing.

**Possible causes:**

1. **Undertrained model** - The most common cause. Models need significant training (10+ epochs) to produce coherent behavior. Early checkpoints may show degenerate policies.

2. **Architecture mismatch** - See gotcha #29. If the architecture doesn't match, the model may still "work" but produce garbage outputs.

3. **Embedding mismatch** - The game state embedding must match between training and inference. Check `embed_size` in the config matches what the model expects.

4. **Wrong player port** - If training data used port 1 but you're controlling port 2, the inputs/outputs are reversed.

**Debugging steps:**

1. Run evaluation to check accuracy:
   ```bash
   mix run scripts/eval_model.exs --checkpoint model.axon --temporal --backbone jamba --window-size 30
   ```

2. Check if the saved config looks reasonable:
   ```bash
   mix run -e '
     {:ok, p} = ExPhil.Training.Checkpoint.load_policy("model_policy.bin")
     IO.inspect(p.config)
   '
   ```

3. Try a known-good model (e.g., MLP model that trained successfully)

4. Enable verbose logging in the Agent to see what actions it's selecting

**Expected eval metrics for a functional model:**
- Button accuracy: >80%
- Stick accuracy: >50% (top-3: >70%)
- Overall weighted: >65%

If metrics are much lower, the model needs more training.

---

## 31. Polaris composed optimizer state structure

**Symptom:** Pattern matching on optimizer state fails with confusing errors after using `Polaris.Updates.compose`.

**Root cause:** When you compose multiple optimizers (e.g., gradient clipping + AdamW):

```elixir
clip = Polaris.Updates.clip_by_global_norm(max_norm: 1.0)
optimizer = Polaris.Updates.adamw(learning_rate: 1.0e-4)
{init_fn, update_fn} = Polaris.Updates.compose(clip, optimizer)
```

The resulting state is wrapped in an extra tuple:

```elixir
# Expected (direct optimizer)
%{count: ..., mu: ..., nu: ...}

# Actual (composed optimizer)
{{clip_state, optimizer_state}}

# Where:
# clip_state = %{count: ...}
# optimizer_state = %{count: ..., mu: ..., nu: ...}
```

This double-nesting affects:
- Checkpoint save/load (must recursively convert tuples)
- Extracting step count for validation
- Any code that inspects optimizer state

**Fix:** Account for the nested structure when accessing optimizer state:

```elixir
# Extracting optimizer step count
def get_optimizer_step(optimizer_state) do
  case optimizer_state do
    # Composed optimizer (gradient clipping + base optimizer)
    {{_clip_state, inner_state}} when is_map(inner_state) ->
      Nx.to_number(inner_state[:count])

    # Direct optimizer (no composition)
    %{count: count} ->
      Nx.to_number(count)
  end
end
```

The `to_binary_backend/1` helper in `Imitation` correctly handles this by recursively processing tuples.

**Code location:** `lib/exphil/training/imitation.ex` - `to_binary_backend/1`, `get_optimizer_step/1`

---

## 32. Prefetcher deadlock with EXLA tensors in spawned processes

**Symptom:** Training shows "No batches processed this epoch" with loss=0.0. GPU utilization is 0% despite VRAM being allocated. The `--no-prefetch` flag makes training work correctly.

**Root cause:** The streaming prefetcher (`Prefetcher.reduce_stream_indexed`) spawns an iterator process that calls `Stream.run()` to iterate batches. When batches are created (via `Nx.stack` on pre-computed embeddings), the Nx operations happen in this spawned process.

EXLA tensors have process-local state and NIF resources that may not work correctly in raw `spawn_link`ed processes. The `Nx.stack` operation blocks or fails silently, causing no batches to be produced.

```elixir
# Inside prefetcher's stream_producer:
iterator = spawn_link(fn ->
  stream
  |> Stream.each(fn batch ->       # batch creation (Nx.stack) happens here
    send(parent, {:batch_ready, ref, batch})
  end)
  |> Stream.run()
end)
```

**The key safety property:** All Nx operations must happen in the main process, not spawned processes.

**Functions and their safety:**
- `reduce_stream_indexed`: Spawns iterator process, Nx operations happen there → **deadlock with EXLA**
- `reduce_indexed`: All operations in main process (no spawn) → **safe for EXLA**

**Fix:** Use `reduce_indexed` for non-streaming (precomputed embeddings) mode:

```elixir
{updated_trainer, epoch_losses, _, updated_global_batch_idx} = cond do
  opts[:prefetch] and streaming_mode ->
    # Streaming: use lazy stream-based prefetcher
    Prefetcher.reduce_stream_indexed(batch_stream, initial_acc, process_batch, ...)

  opts[:prefetch] ->
    # Non-streaming: use list-based prefetcher (safe for EXLA)
    Prefetcher.reduce_indexed(batch_stream, initial_acc, process_batch)

  true ->
    # No prefetching
    batch_stream |> Stream.with_index() |> Enum.reduce(...)
end
```

**Why `reduce_indexed` is safe:** It uses lazy iteration with `Stream.with_index() |> Enum.reduce()` which runs synchronously in the main process. No spawned processes, no deadlock risk.

**Important history (January 2025):** The original `reduce_indexed` called `Enum.to_list(batches)` upfront to materialize all batches before prefetching. This was safe but caused a severe startup delay (~25+ minutes) when training on large datasets (e.g., 3567 batches), because `Nx.stack` was called 3567 times upfront before any training started.

The fix was to simplify `reduce_indexed` to iterate lazily:
```elixir
# OLD (slow - materialized all batches upfront):
def reduce_indexed(batches, initial_acc, fun) do
  batch_list = Enum.to_list(batches)  # Nx.stack on ALL 3567 batches!
  # ... then prefetch from the list
end

# NEW (fast - lazy iteration in main process):
def reduce_indexed(batches, initial_acc, fun) do
  batches
  |> Stream.with_index()
  |> Enum.reduce(initial_acc, fn {batch, idx}, acc ->
    fun.(batch, idx, acc)  # Nx.stack happens here, one batch at a time
  end)
end
```

This is still safe because `Enum.reduce` runs synchronously in the main process - no spawned processes are involved. The batch stream is consumed lazily, and each `Nx.stack` happens in the main process.

**Regression tests:** `test/exphil/training/prefetcher_test.exs` has "Nx tensor compatibility" tests that verify `reduce_indexed` works correctly with EXLA tensors.

**Code location:** `scripts/train_from_replays.exs` around line 1485, `lib/exphil/training/prefetcher.ex`

---

## 33. LSTM/GRU training is inherently slow (not a bug)

**Symptom:** LSTM or GRU training takes 10-15 seconds per batch, while MLP/Mamba take <1s.

**This is expected behavior, not a bug.** Recurrent networks (LSTM, GRU) cannot parallelize across the sequence dimension because each timestep depends on the previous hidden state.

**Why recurrent networks are slow:**

```
Sequence processing:
  MLP/Mamba:  [frame1, frame2, frame3, ...] → All computed in parallel
  LSTM/GRU:   frame1 → frame2 → frame3 → ... → Sequential (each depends on prev)

With gradients (BPTT - Backpropagation Through Time):
  Forward:  30 sequential steps
  Backward: 30 sequential steps (gradients flow backward through time)
  Total:    60 sequential operations that can't be parallelized
```

**Expected training speeds per batch:**

| Architecture | Expected Speed | Parallelizable |
|--------------|----------------|----------------|
| MLP | <500ms | Yes (no sequence) |
| Mamba | <2000ms | Yes (selective scan) |
| LSTM | <15000ms | **No** (sequential) |
| GRU | <15000ms | **No** (sequential) |
| Jamba | <3000ms | Partially |

**Don't confuse with Gotcha #18:**
- Gotcha #18 (Nx.to_number blocking): ~80s/batch, 0% GPU util, affects ALL architectures
- This gotcha: ~10-15s/batch for LSTM/GRU only, GPU util is normal

**Recommendation:** For real-time inference, use Mamba instead of LSTM/GRU. Mamba achieves similar temporal modeling with parallelizable operations (8.9ms inference vs 220ms for LSTM).

**Code location:** `test/exphil/benchmarks/training_speed_test.exs` - thresholds adjusted for recurrent architectures

---

## 34. Imitation.new uses window_size, not seq_len

**Symptom:** Jamba model fails to compile with shape mismatch error:
```
(Axon.CompileError) cannot compile graph because input state_sequence,
inferred as shape {32, 30, 408}, is incompatible with the output of
layer self_attn_compute ({1, 60, 60})
```

**Cause:** `Imitation.new()` only recognizes `window_size` in its config, not `seq_len`. If you pass `seq_len: 30`, it's silently ignored and the default `window_size: 60` is used instead.

**The problem:**
```elixir
# This DOES NOT WORK - seq_len is ignored!
trainer = Imitation.new(
  backbone: :jamba,
  seq_len: 30,  # WRONG: This key is not in @default_config, so it's dropped
  # ...
)

# Imitation uses Keyword.take(opts, Map.keys(@default_config))
# Since seq_len is not in @default_config, it's not included
# The model is built with window_size: 60 (default)
```

**The fix:** Use `window_size` instead of `seq_len`:
```elixir
# This WORKS - window_size is a recognized config key
trainer = Imitation.new(
  backbone: :jamba,
  window_size: 30,  # CORRECT: Use window_size
  # ...
)
```

**Key insight:** The internal machinery does support `seq_len` at lower levels (`Policy.build_temporal`, `Hybrid.build`), but `Imitation.new()` filters opts to only recognized keys. If you need a non-default sequence length, use `window_size`.

**Code location:**
- `lib/exphil/training/imitation.ex:83` - `@default_config` defines valid keys
- `lib/exphil/training/imitation.ex:157` - `Keyword.take(opts, Map.keys(@default_config))` filters unknown keys

---

## 35. Prefetcher materializes all batches into memory

**Status:** FIXED in v6911c16

**Symptom:** System RAM exhausted during training, SSH freezes, can't connect to pod. Training appears to run but system becomes unresponsive.

**Cause:** The `Prefetcher.reduce/3` and `Prefetcher.wrap/2` functions called `Enum.to_list(batches)` which materialized ALL batches into RAM before training even started.

**Why it was written this way (and why it didn't help):**
The intent was to enable "prefetching" - loading the next batch while GPU trains on current. But the implementation was flawed:
1. `Enum.to_list(batches)` loaded ALL batches upfront (defeating the purpose)
2. The async tasks then just returned data already in memory
3. No actual prefetching happened - it was pure overhead + memory waste

The proper prefetching is in `reduce_stream_indexed/4` which pulls from a lazy stream in background tasks. The simple `reduce/3` doesn't need prefetching at all - just iterate lazily.

**The problem:**
```elixir
# OLD CODE (memory leak, no actual benefit):
def reduce(batches, initial_acc, fun) do
  # This loads EVERY batch into memory at once!
  batch_list = Enum.to_list(batches)  # Memory spike happens HERE
  # ... then iterates through already-loaded data
end

def wrap(enumerable, opts \\ []) do
  # Also materializes everything!
  batches = Enum.to_list(enumerable)
  # ...
end
```

For a training run with 10,000 batches of size 64 with 408-dim embeddings:
- Per batch: 64 × 408 × 4 bytes = ~100KB
- All batches: 10,000 × 100KB = **~1GB just for states**
- Plus actions, gradients, etc. → easily 5-10GB

**The fix:** Use lazy iteration instead of materializing:
```elixir
# NEW CODE (memory efficient):
def reduce(batches, initial_acc, fun) do
  # Lazy - only current batch in memory
  batches
  |> Enum.reduce(initial_acc, fn batch, acc ->
    fun.(batch, acc)
  end)
end

def wrap(enumerable, _opts \\ []) do
  # Return lazy stream - don't materialize
  Stream.map(enumerable, & &1)
end
```

**Additional mitigation:** Use `--gc-every N` flag to run garbage collection periodically:
```bash
mix run scripts/train_from_replays.exs --gc-every 50
```

**How to detect this issue:**
1. Monitor RAM with `watch -n 1 'free -h'`
2. If RAM spikes immediately when training starts (before any batches process), this is likely the cause
3. SSH becoming unresponsive is a strong indicator

**Prevention:** The fix is now in the codebase. For older versions:
- Use `--stream-chunk-size` to limit data loaded
- Use `--gc-every 50` for more aggressive garbage collection
- Reduce `--max-files` if training large datasets

**Code location:**
- `lib/exphil/training/prefetcher.ex:70-77` - `reduce/3` now uses lazy Enum.reduce
- `lib/exphil/training/prefetcher.ex:233-238` - `wrap/2` now returns lazy stream

## 36. Embedding pre-computation exhausts RAM on large datasets

**Problem:** Pre-computing embeddings for large datasets (1M+ sequences) accumulates all embeddings in memory before training starts, causing OOM.

**Symptoms:**
- Training crashes during "Pre-computing embeddings for X sequences..."
- Progress shows low percentage (e.g., "Embedding: 4%") then pod becomes unresponsive
- RAM hits 100% before training loop begins

**Root cause:** `Data.precompute_embeddings/2` uses `Enum.flat_map` to collect all embeddings into a list. For 1.8M sequences:
- Each sequence: 60 frames × 400 dims × 4 bytes = ~96KB
- Total: 1.8M × 96KB = **~173GB** (far exceeds available RAM)

**The fix:** Added periodic garbage collection during embedding:
```elixir
# In Data.precompute_embeddings/2
if gc_every > 0 and rem(chunk_idx, gc_every) == 0 and chunk_idx > 0 do
  :erlang.garbage_collect()
end
```

Also added warning for large datasets:
```
⚠ Large dataset (1826706 sequences) - consider using --stream-chunk-size
  Pre-computing will use ~173.1GB RAM
```

**Better solution:** For datasets >500K sequences, use streaming mode:
```bash
mix run scripts/train_from_replays.exs --stream-chunk-size 30
```

This processes files in chunks, computing embeddings on-the-fly instead of pre-computing all.

**Code location:**
- `lib/exphil/training/data.ex:696` - `precompute_embeddings/2` now has GC
- `lib/exphil/training/data.ex:768` - `precompute_frame_embeddings/2` now has GC

## 37. 0% GPU utilization with pre-computed embeddings

**Status:** FIXED

**Symptom:** Training runs extremely slowly (70+ seconds per batch) with 0% GPU utilization despite GPU memory being 90%+ utilized. Model parameters are on GPU, but compute happens on CPU.

**How to identify:**
```
# nvidia-smi shows:
# - GPU Mem: 90%
# - GPU Util: 0%
# Training log shows: "71.3s/batch" (should be <1s)
```

**Root cause:** Pre-computed embeddings are stored on CPU (BinaryBackend) to save GPU memory. During training, `Nx.backend_copy(tensor)` was called without an explicit backend argument:

```elixir
# BAD - This does nothing for CPU tensors!
states = Nx.backend_copy(states)
actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)
```

`Nx.backend_copy/1` copies within the *same* backend. For CPU tensors, it just copies CPU → CPU.
The result: model params on GPU, data on CPU = every tensor operation transfers to GPU, computes, transfers back. Massive overhead.

**The fix:** Explicitly transfer to GPU using `Nx.backend_transfer/2`:

```elixir
# GOOD - Explicitly transfer to GPU
states = Nx.backend_transfer(states, EXLA.Backend)
actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
```

**Key Nx backend functions:**
- `Nx.backend_copy(tensor)` - copies within current backend (no-op for CPU → CPU)
- `Nx.backend_copy(tensor, backend)` - copies to specified backend
- `Nx.backend_transfer(tensor, backend)` - transfers to backend (more efficient, may move instead of copy)
- `Nx.backend(tensor)` - returns current backend module

**What we learned:**
1. `Nx.backend_transfer(tensor, EXLA.Backend)` → Creates EXLA tensor → **breaks defn closures**
2. `Nx.backend_copy(tensor, Nx.BinaryBackend)` → Safe for defn → **works but slow (4-5s/batch)**
3. `Nx.Defn.jit(outer_fn)` → **FAST (~200ms/batch)** - tensors become Defn.Expr during tracing

**Why closures are the problem:**
The `value_and_grad(loss_fn)` traces `loss_fn`, which captures `states` and `actions` in its closure.
EXLA tensors in closures conflict with Defn.Expr tracing.

**The solution - JIT compile the outer function:**
```elixir
# WRONG - EXLA tensors captured in inner closure cause conflict
fn params, states, actions ->
  loss_fn = fn p -> compute(p, states, actions) end  # states/actions are EXLA here!
  Nx.Defn.value_and_grad(loss_fn).(params)
end

# RIGHT - Wrap with Nx.Defn.jit so tensors become Defn.Expr during tracing
inner_fn = fn params, states, actions ->
  loss_fn = fn p -> compute(p, states, actions) end  # states/actions are Defn.Expr!
  Nx.Defn.value_and_grad(loss_fn).(params)
end
Nx.Defn.jit(inner_fn, compiler: EXLA)
```

When the outer function is JIT compiled, all arguments (including states/actions) are traced as
Defn.Expr, not as live EXLA tensors. This avoids the conflict while keeping everything on GPU.

**Performance comparison:**
| Approach | Time/batch | Notes |
|----------|------------|-------|
| No fix (original bug) | 71s | 0% GPU util, all on CPU |
| BinaryBackend copy | 4-5s | Works but CPU→GPU transfer each batch |
| **JIT outer function** | **~200ms** | Tensors stay on GPU, optimal |

**Why multi-arg value_and_grad doesn't work:**
```elixir
# This DOESN'T work - value_and_grad expects arity-1 function
loss_fn = fn params, states, actions -> ... end
Nx.Defn.value_and_grad(loss_fn, [0]).(params, states, actions)  # FunctionClauseError!
```

**Implications for other architectures:**
- This pattern applies to ALL training code using value_and_grad with batch data
- Temporal models (LSTM, Mamba, Attention) benefit equally from this fix
- Any code that calls value_and_grad should wrap the outer function with Nx.Defn.jit

**Code location:**
- `lib/exphil/training/imitation.ex:875-910` - JIT-wrapped loss+grad function

**Regression test:**
- `test/exphil/benchmarks/gpu_integration_test.exs` - "CPU to GPU backend transfer" tests
- `scripts/test_gpu_speed.exs` - Quick benchmark script

---

## 38. Streaming prefetcher timeout causes silent batch drops

**Status:** FIXED

**Symptom:** Training with `--stream-chunk-size` shows "No batches processed this epoch" after exactly 60 seconds, despite having valid replay data.

**Output looked like:**
```
[07:26:04]   📦 Processing chunk 1/1 (108 files)...
[07:26:04]     Parsing 108 files...
[07:27:04] ⚠️  No batches processed this epoch - check replay data
[07:27:04]   ✓ Epoch 1 complete: train_loss=0.0 (60s)
```

**Root cause:** The streaming prefetcher had a hardcoded 60-second timeout:

```elixir
receive do
  {:batch, ^stream_ref, batch} -> {:ok, batch}
  {:done, ^stream_ref} -> :done
after
  60_000 -> :timeout  # Too short for large chunk parsing!
end
```

When parsing a large chunk of files took longer than 60 seconds, the prefetcher silently timed out and skipped all batches.

**The fix:**
1. Increased timeout from 60s to 5 minutes (300s)
2. Added warning log when timeout occurs

```elixir
@prefetch_timeout_ms 300_000

# ... and in timeout handler:
:timeout ->
  Logger.warning("[Prefetcher] Timeout waiting for batch #{idx}")
  process_prefetched_batches(...)
```

**Code location:** `lib/exphil/training/prefetcher.ex:365-377`

**Workaround (if timeout still occurs):** Use smaller `--stream-chunk-size` values (e.g., 5000 instead of 20000) to reduce per-chunk parsing time.

---

## 39. XLA preallocates 90% GPU memory and never releases it

**Status:** DOCUMENTED (workaround available)

**Symptom:** GPU memory shows 90% usage immediately after first EXLA operation, even before loading significant data. Subsequent operations OOM despite seemingly having enough memory, especially when running multiple models/architectures sequentially.

**Example output:**
```
[08:11:26] GPU: 21.57 GB/23.99 GB (90%) | Util: 3%  # 90% used before training starts!
...
[08:25:14] ❌ Architecture Mamba SSM failed: Out of memory while trying to allocate 3932160 bytes.
```

**Root cause:** XLA's BFC (Best-Fit with Coalescing) allocator preallocates 90% of GPU VRAM by default for performance optimization. This memory is **never released** during the BEAM process lifetime - not by garbage collection, not by setting tensors to nil, not by any API call.

This mirrors TensorFlow and JAX behavior (same underlying XLA client).

**What DOESN'T work:**
```elixir
# These do NOT free GPU memory:
:erlang.garbage_collect()           # Only frees BEAM memory, not XLA/GPU
tensor = nil                         # Reference gone, but XLA keeps the memory
Nx.backend_transfer(t, Nx.BinaryBackend)  # Frees that tensor's GPU memory, but BFC keeps the pool
```

**The fix - disable preallocation:**
```elixir
# Add to script or config/runtime.exs BEFORE any EXLA operations:
Application.put_env(:exla, :clients, cuda: [platform: :cuda, preallocate: false])
```

**Alternative - limit preallocation:**
```elixir
# Preallocate only 50% instead of 90%:
Application.put_env(:exla, :clients, cuda: [platform: :cuda, memory_fraction: 0.5])
```

**Trade-offs:**

| Setting | Memory Usage | Speed | Use Case |
|---------|--------------|-------|----------|
| Default (preallocate: true, 90%) | 90% immediately | Fastest | Single model training |
| `preallocate: false` | On-demand | ~5-10% slower | Multiple models, benchmarks |
| `memory_fraction: 0.5` | 50% immediately | ~5% slower | Shared GPU, multiple processes |

**Nuclear option - subprocess isolation:**

If memory issues persist, run each model in a separate BEAM process:
```bash
# GPU memory fully released when process exits
elixir -e "System.cmd(\"mix\", [\"run\", \"script.exs\", \"--arch\", \"mamba\"])"
```

**Key Nx/EXLA memory functions:**

| Function | Effect |
|----------|--------|
| `Nx.backend_copy(t, Nx.BinaryBackend)` | Copies to CPU (GPU tensor remains) |
| `Nx.backend_transfer(t, Nx.BinaryBackend)` | Moves to CPU (frees that tensor's GPU memory) |
| `:erlang.garbage_collect()` | Frees BEAM memory only |
| Process termination | **Only way to fully release XLA memory pool** |

**References:**
- [Elixir Forum: GPU RAM deallocation issue](https://elixirforum.com/t/possible-graphic-ram-deallocation-issue-noticed-when-using-nx-with-exla/47629)
- [Elixir Forum: Axon memory usage high](https://elixirforum.com/t/axon-memory-usage-high/63304)
- [EXLA docs](https://hexdocs.pm/exla/EXLA.html)

**Code location:** `scripts/benchmark_architectures.exs:50-52`

**Recommended setting for benchmarks:** `memory_fraction: 0.7` balances preallocation benefits with headroom for spikes. Use `preallocate: false` only if OOM persists.

---

## 40. --augment flag bypasses precomputed embeddings (100x slower)

**Status:** DOCUMENTED (workaround: don't use --augment)

**Symptom:** Training with `--augment` is extremely slow (~100s/batch instead of <1s/batch). GPU shows 90% memory usage but 0% utilization - training runs entirely on CPU.

**Example output:**
```
[08:11:26] GPU: 21.57 GB/23.99 GB (90%) | Util: 0%
[08:11:26] Starting epoch 1/30
[08:13:06] Batch 1/3227 | Loss: 7.1918 | ~100s/it   # WAY TOO SLOW
```

**Root cause:** Augmentation modifies raw game states (X positions, velocities, facing directions) which must happen BEFORE embedding. When `--augment` is enabled, precomputed embeddings are bypassed entirely.

**The culprit (data.ex:329):**
```elixir
use_precomputed = dataset.embedded_frames != nil and augment_fn == nil
```

With augmentation enabled, every batch:
1. Retrieves 512 raw game states
2. Applies augmentation (mirror flip, noise injection)
3. Re-embeds all 512 states from scratch
4. This embedding runs on CPU (EXLA not involved), hence 0% GPU util

**Workaround options:**

1. **Don't use `--augment`** (recommended for now)
   ```bash
   # Remove --augment flag from command
   mix run scripts/train_from_replays.exs --epochs 30 ...
   ```

2. **Use frame delay augmentation instead** (operates at temporal level, no re-embedding)
   ```bash
   mix run scripts/train_from_replays.exs --online-robust --epochs 30 ...
   ```

3. **Train longer without augmentation** - More epochs may compensate for reduced diversity

**Why mirror augmentation matters:**
- Most Melee stages are horizontally symmetric
- Mirroring effectively doubles training data
- Prevents overfitting to one side of the stage

**Planned fix:** Augmented embedding caching - pre-compute both original and mirrored embeddings, then randomly select during training. See [GOALS.md](GOALS.md#2-data-pipeline) for details.

**Code locations:**
- `lib/exphil/training/data.ex:329` - The bypass condition
- `lib/exphil/training/augmentation.ex` - Mirror and noise functions

---

## 16. Precomputed Embeddings: Array vs Stacked Tensor Performance

**Status:** FIXED (Jan 2026)

**Symptom:** MLP training with precomputed embeddings was taking ~50 seconds per batch instead of ~50 milliseconds. GPU showed 90% memory usage but 0% utilization.

**Example output:**
```
[08:11:26] GPU: 21.57 GB/23.99 GB (90%) | Util: 0%
[08:11:26] Starting epoch 1/30
[08:12:16] Batch 1/3227 | Loss: 2.1234 | ~50s/it   # ~1000x slower than expected
```

**Root cause:** Precomputed embeddings were stored as individual tensors in an Erlang array (1.3M tensors of shape `{287}`). Each batch of 512 frames required:
1. 512 individual `:array.get()` calls
2. `Nx.stack()` to combine 512 scattered tensors into `{512, 287}`
3. `Nx.backend_transfer()` to move to GPU

`Nx.stack()` on 512 individual tensors is extremely slow because it must allocate memory and copy from 512 scattered locations.

**The fix:** Store embeddings as a single stacked tensor `{num_frames, embed_size}` and use `Nx.take()` for batch extraction:

```elixir
# OLD (slow): Individual tensors in Erlang array
embedded_array = :array.from_list(individual_tensors)
# Batching: O(batch_size) stack operations
embeddings = Enum.map(indices, fn idx -> :array.get(idx, array) end)
states = Nx.stack(embeddings)  # SLOW: 512 individual copies

# NEW (fast): Single stacked tensor
stacked_embeddings = Nx.concatenate(batch_tensors, axis: 0)
# Batching: O(1) gather operation
indices_tensor = Nx.tensor(indices, type: :s64)
states = Nx.take(stacked_embeddings, indices_tensor, axis: 0)  # FAST: single gather
```

**Performance improvement:**
- OLD: ~50 seconds/batch (CPU-bound copying)
- NEW: ~50 milliseconds/batch (GPU gather)
- **1000x speedup**

**Code locations:**
- `lib/exphil/training/data.ex` - `precompute_frame_embeddings/2` and `create_batch_precomputed/3`
- `lib/exphil/training/embedding_cache.ex` - Save/load handles both formats

**Cache compatibility:** Old cached files (array format) are auto-converted to stacked tensor on load.

---

## 41. --cache-embeddings flag parsed but not used (train_from_replays.exs)

**Status:** FIXED (Jan 2026)

**Symptom:** Training with `--cache-embeddings` shows precomputation completing, but cache directory remains empty. Every run recomputes embeddings from scratch (~14 minutes for 1.4M frames), and training speed is ~50s/batch instead of ~0.04s/batch.

**Example output:**
```bash
ls -la /workspace/cache/embeddings/
# Shows: empty directory despite multiple training runs

# Training shows:
# Embedding: 100% (1454622/1454622) - done!
# But no "[EmbeddingCache] Saved ..." message appears
```

**Root cause:** The `--cache-embeddings` flag was defined in `Config.ex` and parsed, but `train_from_replays.exs` called the non-cached functions:

```elixir
# BUG: Always called non-cached version regardless of flag
Data.precompute_frame_embeddings(dataset)  # No caching!
Data.precompute_embeddings(seq_dataset)     # No caching!

# Should have called:
Data.precompute_frame_embeddings_cached(dataset, cache_opts)
Data.precompute_embeddings_cached(seq_dataset, cache_opts)
```

The benchmark script (`benchmark_architectures.exs`) used the cached versions correctly, which is why caching worked there but not in training.

**The fix:** Updated `train_from_replays.exs` to use cached functions when `--cache-embeddings` is set:

```elixir
if opts[:cache_embeddings] do
  Output.puts("  Using embedding cache (#{opts[:cache_dir]})...")

  Data.precompute_frame_embeddings_cached(dataset,
    cache: true,
    cache_dir: opts[:cache_dir],
    force_recompute: opts[:no_cache],
    replay_files: replay_files,
    show_progress: true
  )
else
  Data.precompute_frame_embeddings(dataset)
end
```

**Also added:** Warning when using `--cache-embeddings` with `--stream-chunk-size` since streaming mode doesn't benefit from caching (chunks processed on-the-fly).

**Performance impact:**
- Without fix: Every run = 14 min precompute + ~50s/batch training
- With fix: First run = 14 min, subsequent runs = instant load + ~0.04s/batch training
- **Speedup after first run: ~1000x+**

**Code locations:**
- `scripts/train_from_replays.exs:1128-1175` - Updated precomputation logic
- `scripts/train_from_replays.exs:159` - Added `EmbeddingCache` to alias list
- `scripts/train_from_replays.exs:524-530` - Added streaming mode warning

**RunPod note:** Ensure `source /app/scripts/runpod_entrypoint.sh` is run to create `/app/cache -> /workspace/cache` symlink. Without this, cache may be written to ephemeral storage and lost on pod restart.

---

## 42. Nx.to_number returns atoms for special float values (NaN, Infinity)

**Symptoms:**
```
** (ArithmeticError) bad argument in arithmetic expression
    scripts/train_from_replays.exs:1799: anonymous fn/10 in ...
```

Training crashes mid-epoch with `ArithmeticError` during loss EMA calculation.

**Root cause:** `Nx.to_number/1` returns **atoms** (`:nan`, `:infinity`, `:neg_infinity`) for special floating-point values, not floats. When you then try to do arithmetic like:

```elixir
raw = Nx.to_number(metrics.loss)  # Returns :nan atom
smoothed = 0.1 * raw + 0.9 * prev  # ArithmeticError! Can't multiply atoms
```

**The fix:** Check for special atoms before doing arithmetic:

```elixir
is_special_atom? = fn
  :nan -> true
  :infinity -> true
  :neg_infinity -> true
  _ -> false
end

smoothed =
  cond do
    is_special_atom?.(raw) -> raw  # Propagate atom, skip arithmetic
    smoothed_loss == nil -> raw
    is_special_atom?.(smoothed_loss) -> raw
    true -> 0.1 * raw + 0.9 * smoothed_loss
  end
```

**Why this happens:** Loss becomes NaN when training diverges (usually due to learning rate too high, gradient explosion, or numerical instability). The NaN propagates to `Nx.to_number`, which returns `:nan` atom.

**Prevention:** Use `--max-grad-norm 1.0` and conservative learning rates (`--lr 1e-5`) for large models like Jamba. The NaN check downstream will catch it and provide a helpful error message.

**Code locations:**
- `scripts/train_from_replays.exs:226-231` - `is_special_atom?` helper
- `scripts/train_from_replays.exs:1847-1852` - Fixed EMA calculation with cond guard

---

## 43. Precomputed embeddings on CPU cause slow Nx.take (~17s/batch)

**Status:** FIXED

**Symptom:** Training with `--cache-embeddings` is extremely slow (~17s/batch instead of ~0.2s/batch) despite JIT compilation working correctly. GPU shows 90% memory usage but 0% utilization.

**Example output:**
```
[22:42:49] GPU: 21.57 GB/23.99 GB (90%) | Util: 0%
Epoch 1: █░░░░░░░░░░░░░░░░░░░   5% | 133/2556 | loss: 14.5466 | 17.24s/it | ETA: 696m 9s
```

**Root cause:** Precomputed embeddings loaded from cache are stored on CPU (BinaryBackend). Every batch does:

```elixir
# In create_batch (data.ex:392-393):
indices_tensor = Nx.tensor(indices, type: :s64)
Nx.take(tensor, indices_tensor, axis: 0)  # SLOW! tensor is on CPU
|> Nx.backend_transfer(EXLA.Backend)
```

`Nx.take` runs on whatever backend the source tensor is on. With 1.45M frames × 287 dims on CPU, extracting 512 scattered rows is slow (random memory access pattern).

**The fix:** Transfer embeddings to GPU once after loading, before training starts:

```elixir
# In train_from_replays.exs, after precomputation:
base_dataset =
  if base_dataset.embedded_frames != nil do
    Output.puts("  Transferring embeddings to GPU...")
    gpu_embeddings = Nx.backend_transfer(base_dataset.embedded_frames, EXLA.Backend)
    %{base_dataset | embedded_frames: gpu_embeddings}
  else
    base_dataset
  end
```

Now `Nx.take` runs on GPU - O(1) gather operation instead of scattered CPU reads.

**Performance impact:**
| Approach | Time/batch | Notes |
|----------|------------|-------|
| CPU embeddings (bug) | ~17s | Random CPU memory access |
| **GPU embeddings (fixed)** | **~0.2s** | Fast GPU gather |

**Memory requirements:**
- 1.45M frames × 287 dims × 4 bytes (f32) = ~1.67 GB
- RTX 4090 with 24GB VRAM easily accommodates this
- For larger datasets, may need to use streaming mode instead

**Code location:** `scripts/train_from_replays.exs:1231-1254` - GPU transfer after precomputation

**Fallback safeguard:** If for some reason the GPU transfer in the training script fails (e.g., stale code), `create_batch_precomputed` in `data.ex` has a fallback that:
1. Detects CPU embeddings on first batch
2. Logs a warning
3. Transfers to GPU once (cached in process dictionary)
4. Subsequent batches use the cached GPU tensor

This ensures fast training even if the primary GPU transfer fails. Look for "Embeddings on CPU - Nx.take will be slow!" in logs to detect this condition.

**Debugging:** The training script now shows diagnostic output before GPU transfer:
```
GPU transfer check: has_embeddings=true, streaming_mode=false
Embedding tensor: shape={1234567, 287}, backend=Nx.BinaryBackend
Transferring embeddings to GPU (1416.2 MB)...
```
If you see `BinaryBackend` in the logs followed by "Transferring embeddings to GPU", the fix is working. If you don't see the transfer message, check the diagnostic output to understand why.

## 44. Optimizer not JIT compiled with EXLA causes 0% GPU utilization

**Status:** FIXED

**Symptom:** Training runs but GPU utilization shows 0% in `nvidia-smi` despite GPU memory being allocated (90%). Training speed is ~35s/batch instead of expected <1s/batch.

**Example output:**
```
[08:32:30]   GPU: 21.57 GB/23.99 GB (90%) | Util: 0%
Epoch 1: ... | 28.94s/it
```

**Root cause:** The optimizer and `apply_updates_fn` were JIT compiled without specifying `compiler: EXLA`:

```elixir
# BAD - defaults to CPU Evaluator
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)

# GOOD - uses GPU
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2, compiler: EXLA)
```

Without `compiler: EXLA`, `Nx.Defn.jit` uses the default Evaluator which runs on CPU. The forward/backward pass used GPU (via `build_loss_and_grad_fn` which did specify EXLA), but the optimizer step ran on CPU.

**The fix:** Add `compiler: EXLA` to all JIT calls in `imitation.ex`:

```elixir
optimizer_fn = Nx.Defn.jit(optimizer_update, compiler: EXLA)
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2, compiler: EXLA)
```

**Code location:** `lib/exphil/training/imitation.ex:272-275`

**Verification:** Run `nvidia-smi dmon -s u -d 1` while training. You should see GPU utilization spikes (37-100%) instead of constant 0%.

## 45. O(n) list traversal for action collection causes ~28s/batch

**Status:** FIXED

**Symptom:** Training is slow (~28s/batch) even with embeddings on GPU and optimizer using EXLA. GPU utilization shows brief spikes but overall throughput is poor.

**Root cause:** The `create_batch_precomputed` function used `Enum.at` to access frames:

```elixir
# BAD - O(n) per access, 238K frames × 1024 batch = 244M traversals/batch!
action_frame = Enum.at(dataset.frames, idx + frame_delay)
```

Elixir lists are linked lists. `Enum.at(list, n)` traverses from head to index n, which is O(n). With 238K frames and batch_size 1024, each batch requires ~244 million list node traversals.

**The fix:** Use Erlang `:array` for O(1) random access:

```elixir
# Convert list to array once (cached in process dictionary)
frames_array = get_or_create_frames_array(dataset)

# O(1) access
action_frame = :array.get(idx + frame_delay, frames_array)
```

**Performance impact:**
| Approach | Time/batch | Complexity |
|----------|------------|------------|
| `Enum.at` (bug) | ~28s | O(batch_size × dataset_size) |
| **`:array.get` (fixed)** | **<1s** | O(batch_size) |

**Code location:** `lib/exphil/training/data.ex` - `create_batch_precomputed`, `create_batch_augmented`, `create_batch_standard`

**Key insight:** Always use `:array` or maps for random access in hot paths. Elixir lists are great for sequential access and prepending, but terrible for random access.

## 46. LSTM/GRU gradient explosion causes NaN loss mid-training

**Status:** FIXED

**Symptom:** LSTM or GRU training starts well, loss decreases steadily, then suddenly goes NaN partway through the first epoch:

```
loss: 3.1073 → NaN at batch 5951
```

**Root cause:** Multiple factors contribute to RNN gradient explosion:

1. **No layer normalization:** Unlike Mamba and Transformer which have built-in normalization, vanilla LSTM/GRU hidden states can grow unbounded.

2. **Input scale:** Raw embeddings with varying magnitudes compound through timesteps.

3. **Learning rate too high:** RNNs need lower LR (1e-5) compared to feedforward networks (1e-4).

**Note on orthogonal initialization:** While literature recommends orthogonal init for RNNs, Axon's `orthogonal` initializer has shape constraints that fail with certain hidden sizes. We use `glorot_uniform` with layer normalization instead, which provides equivalent stability.

**The fix:** Layer normalization in `lib/exphil/networks/recurrent.ex`:

```elixir
# 1. Input layer normalization
normalized_input = Axon.layer_norm(input, name: "input_ln")

# 2. Layer norm after each RNN layer
Axon.layer_norm(output_seq, name: "#{name}_ln")
```

Plus architecture-specific learning rates in `scripts/train_all_architectures.sh`:

```bash
LR_LSTM="1e-5"      # 10x lower than MLP
LR_GRU="1e-5"       # 10x lower than MLP
RNN_GRAD_CLIP="0.5" # Aggressive gradient clipping
```

**Comparison of stability:**

| Architecture | Gradient Flow | Built-in Normalization | Recommended LR |
|--------------|---------------|------------------------|----------------|
| MLP | Direct | Optional | 1e-4 |
| LSTM/GRU | Through gates | Now added | 1e-5 |
| Mamba | Selective | Yes (via gates) | 5e-5 |
| Attention | Skip connections | Layer norm | 1e-4 |

**References:**
- Ba et al. (2016) - "Layer Normalization"
