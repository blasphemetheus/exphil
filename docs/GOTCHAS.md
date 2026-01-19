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
