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

**Status:** Known performance issue, not yet fixed.

`train_step/3` calls `deep_backend_copy(trainer.policy_params)` every batch, copying all model parameters (~2M for typical models). This is expensive.

**Why it exists:** Workaround for EXLA/Defn.Expr tensor mismatch (see Gotcha #7). When `Nx.Defn.value_and_grad` traces the loss function, it creates Expr tensors that can't mix with concrete EXLA tensors captured in closures.

**Current code:**
```elixir
# In train_step - COPIES ALL PARAMS EVERY BATCH
model_state = deep_backend_copy(trainer.policy_params)
loss_fn = fn params ->
  predict_fn.(params, states)  # states is captured!
  # ...
end
{loss, grads} = Nx.Defn.value_and_grad(loss_fn).(model_state)
```

**Proper fix:** Refactor to use `defn` functions that take all inputs as parameters:

```elixir
# Define at module level with defn
defn train_step_impl(params, states, actions, predict_fn) do
  # No closures, all inputs are arguments
  logits = predict_fn.(params, states)
  Policy.imitation_loss(logits, actions)
end

# In train_step - NO COPYING
{loss, grads} = Nx.Defn.value_and_grad(&train_step_impl/4).(
  trainer.policy_params, states, actions, trainer.predict_fn
)
```

**Estimated impact:** 20-30% speedup by eliminating per-batch parameter copies.

**Workaround for now:** The fix is complex and risks breaking the training loop. The `Nx.to_number` fix (Gotcha #18) addresses the larger blocking issue. Parameter copying adds overhead but doesn't cause 0% GPU utilization.
