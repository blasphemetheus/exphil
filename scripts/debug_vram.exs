#!/usr/bin/env elixir
# VRAM diagnostic: wraps the exact training flow with memory snapshots
# Run: mix run scripts/debug_vram.exs 2>&1 | tee logs/debug_vram.log

alias ExPhil.Training.{Imitation, GPUUtils}
alias ExPhil.Training.Output

defmodule VRAMDebug do
  def gpu_mb do
    case GPUUtils.get_memory_info() do
      {:ok, info} -> info.used_mb
      _ -> -1
    end
  end

  def snap(label) do
    :erlang.garbage_collect()
    Process.sleep(200)
    mb = gpu_mb()
    IO.puts("[VRAM] #{label}: #{mb} MB")
    mb
  end

  def snap_no_gc(label) do
    mb = gpu_mb()
    IO.puts("[VRAM] #{label} (no GC): #{mb} MB")
    mb
  end
end

# Use the training script's exact setup, then diverge for VRAM tracking
# We'll piggyback on precomputed data from a previous run via embedding cache

VRAMDebug.snap("startup")

# Build trainer with same config as mamba_test_3
Output.puts("Building trainer...")
trainer = Imitation.new(
  embed_size: 288,
  hidden_sizes: [512, 512, 256],
  hidden_size: 512,
  learning_rate: 1.0e-4,
  batch_size: 32,
  precision: :bf16,
  temporal: true,
  backbone: :mamba,
  window_size: 60,
  num_layers: 2,
  state_size: 16,
  expand_factor: 2,
  conv_size: 4,
  focal_loss: false,
  label_smoothing: 0.0,
  button_weight: 1.0,
  button_pos_weight: nil,
  stick_edge_weight: nil,
  policy_type: :autoregressive
)

VRAMDebug.snap("after trainer init")

# JIT compile validation function (same as real script's warmup)
Output.puts("JIT compiling validation function (warmup)...")
dummy_states = Nx.broadcast(0.0, {32, 60, 288}) |> Nx.as_type(:bf16)
dummy_actions = %{
  buttons: Nx.broadcast(0.0, {32, 8}),
  main_x: Nx.broadcast(0, {32}) |> Nx.as_type(:s64),
  main_y: Nx.broadcast(0, {32}) |> Nx.as_type(:s64),
  c_x: Nx.broadcast(0, {32}) |> Nx.as_type(:s64),
  c_y: Nx.broadcast(0, {32}) |> Nx.as_type(:s64),
  shoulder: Nx.broadcast(0, {32}) |> Nx.as_type(:s64)
}
# Warmup eval loss (validation path)
_ = trainer.eval_loss_fn.(trainer.policy_params, dummy_states, dummy_actions) |> Nx.to_number()
VRAMDebug.snap("after validation JIT warmup")

# Warmup predict_fn (diagnostics path)
Output.puts("JIT compiling predict function...")
{_b, _mx, _my, _cx, _cy, _sh} = trainer.predict_fn.(trainer.policy_params, dummy_states)
VRAMDebug.snap("after predict JIT warmup")

# Create synthetic batches that match the real shape
# This avoids the replay parsing complexity while testing the exact same GPU path
Output.puts("Creating synthetic batches (same shapes as real training)...")
key = Nx.Random.key(42)

make_batch = fn key ->
  {states, key} = Nx.Random.normal(key, shape: {32, 60, 288}, type: :bf16)
  {buttons, key} = Nx.Random.uniform(key, shape: {32, 8}, type: :f32)
  buttons = Nx.round(buttons)
  {mx, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {my, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {cx, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {cy, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {sh, key} = Nx.Random.randint(key, 0, 5, shape: {32}, type: :s64)

  batch = %{
    states: states,
    actions: %{buttons: buttons, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh}
  }
  {batch, key}
end

# Pre-generate CPU tensors that simulate what create_sequence_batch_fast does
# (CPU embeddings → Nx.stack → backend_transfer to GPU)
make_cpu_batch = fn key ->
  # 32 individual CPU sequence tensors (like from embedded_sequences array)
  {cpu_seqs, key} = Enum.map_reduce(1..32, key, fn _, k ->
    {t, k} = Nx.Random.normal(k, shape: {60, 288}, type: :bf16)
    {Nx.backend_copy(t, Nx.BinaryBackend), k}
  end)
  # Stack and transfer to GPU (exactly what create_sequence_batch_fast does)
  states = cpu_seqs |> Nx.stack() |> Nx.backend_transfer(EXLA.Backend)
  # Actions on GPU
  {buttons, key} = Nx.Random.uniform(key, shape: {32, 8}, type: :f32)
  buttons = Nx.round(buttons)
  {mx, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {my, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {cx, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {cy, key} = Nx.Random.randint(key, 0, 17, shape: {32}, type: :s64)
  {sh, key} = Nx.Random.randint(key, 0, 5, shape: {32}, type: :s64)
  batch = %{
    states: states,
    actions: %{buttons: buttons, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh}
  }
  {batch, key}
end

{batch1, key} = make_cpu_batch.(key)
{batch2, key} = make_cpu_batch.(key)
{batch3, _key} = make_cpu_batch.(key)

VRAMDebug.snap("after synthetic batch creation")

# JIT warmup
Output.puts("JIT warmup...")
{trainer, _} = Imitation.train_step(trainer, batch1, nil)
VRAMDebug.snap("after JIT warmup (step 1)")

# Run 500 steps with freshly created batches each time (like real training)
Output.puts("\n=== Training 500 steps (fresh batches) ===")
{trainer, key} = Enum.reduce(1..500, {trainer, key}, fn i, {t, k} ->
  {batch, k} = make_cpu_batch.(k)
  {new_t, metrics} = Imitation.train_step(t, batch, nil)
  _loss = Nx.to_number(metrics.loss)

  if rem(i, 100) == 0 do
    VRAMDebug.snap_no_gc("  step #{i}")
  end

  {new_t, k}
end)

VRAMDebug.snap("after 500 training steps")

# Simulate validation (10 forward passes, no grad)
Output.puts("\n=== Validation (10 eval batches) ===")
VRAMDebug.snap("before validation")

Enum.each(1..10, fn _ ->
  loss = trainer.eval_loss_fn.(trainer.policy_params, batch1.states, batch1.actions)
  Nx.to_number(loss)
end)

VRAMDebug.snap_no_gc("after validation (no GC)")
VRAMDebug.snap("after validation (with GC)")

# Simulate diagnostics (10 predict calls)
Output.puts("\n=== Diagnostics (10 predict calls) ===")
VRAMDebug.snap("before diagnostics")

Enum.each(1..10, fn _ ->
  {_b, _mx, _my, _cx, _cy, _sh} = trainer.predict_fn.(trainer.policy_params, batch1.states)
end)

VRAMDebug.snap_no_gc("after diagnostics (no GC)")
VRAMDebug.snap("after diagnostics (with GC)")

# THE CRITICAL TEST: can we do another training step?
Output.puts("\n=== Epoch 2 first step ===")
VRAMDebug.snap("before epoch 2 step 1")

{_trainer, metrics} = Imitation.train_step(trainer, batch2, nil)
_loss = Nx.to_number(metrics.loss)

VRAMDebug.snap("after epoch 2 step 1 — SUCCESS!")

Output.puts("\n✓ No OOM — the model fits with these settings.")
