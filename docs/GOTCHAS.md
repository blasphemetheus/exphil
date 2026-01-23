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
- CPU utilization also low (6-10%)

**Root cause:** `Nx.to_number/1` forces a synchronous GPU→CPU transfer. When called after every batch:

```
GPU computes batch → GPU IDLE (waiting) → CPU reads loss → Next batch
                     ↑
              This idle time dominates training
```

The GPU finishes computing quickly but sits idle while waiting for the CPU to fetch the scalar loss value.

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

**Related functions affected:**
- `Imitation.train_step/3` - now returns `loss` as tensor
- `Imitation.evaluate_batch/2` - returns tensor for accumulation
- `Imitation.evaluate/2` - handles conversion internally

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

## 26. Streaming mode requires --no-prefetch

**Symptom:** With `--stream-chunk-size`, training shows "No batches processed this epoch" even though replays are found.

**Root cause:** The prefetcher uses `Stream.each` with a producer process that waits for batch requests. With lazy streaming chunks, the producer blocks waiting for the stream to yield batches, while the main process sends batch requests. This creates a deadlock-like situation where batches are never consumed.

**Fix:** Use `--no-prefetch` when using streaming mode:
```bash
mix run scripts/train_from_replays.exs \
  --stream-chunk-size 30 \
  --no-prefetch
```

**Performance impact:** ~10-20% slower without prefetch, but streaming mode is necessary for large datasets that don't fit in RAM.

**Future fix:** Rewrite prefetcher to handle lazy chunk streams correctly.

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
