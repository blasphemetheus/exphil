#!/usr/bin/env elixir
# Profile the overhead of Trainer.fit's callback system
# Compares: raw train_step loop vs Trainer.fit with callbacks
# Run: mix run scripts/profile_callback_overhead.exs

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation, Callback, TrainingState}
alias ExPhil.Training.Callbacks.{ProgressBar, Validation, Diagnostics, EpochSummary, GracefulShutdown, Checkpoint}

opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "10", "--batch-size", "16", "--seed", "42"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

pipeline = Pipeline.setup!(opts)
trainer = Trainer.new(pipeline, opts)
{stream, _} = Pipeline.batch_stream(pipeline, [])

# JIT warmup
batch = Enum.take(stream, 1) |> hd()
{trainer, _} = Imitation.train_step(trainer, batch, nil)

Output.puts("\n=== Callback Overhead Profiling ===\n")

batches = stream |> Stream.take(200) |> Enum.to_list()

# 1. Raw loop — no callbacks, no state management
Output.puts("1. Raw loop (train_step only):")
{raw_us, _} = :timer.tc(fn ->
  Enum.reduce(batches, trainer, fn batch, t ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    _ = Nx.to_number(metrics.loss)
    new_t
  end)
end)
raw_ms = raw_us / 200_000
Output.puts("   #{Float.round(raw_ms, 2)}ms/iter")

# 2. With state updates (like Trainer.fit but no callbacks)
Output.puts("2. With TrainingState management:")
state = %TrainingState{trainer: trainer, pipeline: pipeline, opts: opts, epoch: 1, epochs: 1}
{state_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.with_index(batches), state, fn {batch, idx}, st ->
    {new_t, metrics} = Imitation.train_step(st.trainer, batch, nil)
    loss = Nx.to_number(metrics.loss)
    %{st | trainer: new_t, step: st.step + 1, batch_idx: idx,
      batch_metrics: %{loss: loss}, epoch_losses: [loss | st.epoch_losses]}
  end)
end)
state_ms = state_us / 200_000
Output.puts("   #{Float.round(state_ms, 2)}ms/iter (+#{Float.round(state_ms - raw_ms, 2)}ms overhead)")

# 3. With empty callback dispatch (measures dispatch cost)
Output.puts("3. With empty callback dispatch:")
empty_callbacks = Callback.init_all([])
{dispatch_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.with_index(batches), {state, empty_callbacks}, fn {batch, idx}, {st, cbs} ->
    {new_t, metrics} = Imitation.train_step(st.trainer, batch, nil)
    loss = Nx.to_number(metrics.loss)
    st = %{st | trainer: new_t, step: st.step + 1, batch_idx: idx,
      batch_metrics: %{loss: loss}, epoch_losses: [loss | st.epoch_losses]}
    st = Callback.increment_event_count(st, :on_batch_end)
    {_, st, cbs} = Callback.run(cbs, :on_batch_end, st)
    if rem(idx, 100) == 0, do: :erlang.garbage_collect()
    {st, cbs}
  end)
end)
dispatch_ms = dispatch_us / 200_000
Output.puts("   #{Float.round(dispatch_ms, 2)}ms/iter (+#{Float.round(dispatch_ms - raw_ms, 2)}ms overhead)")

# 4. With ProgressBar callback only
Output.puts("4. With ProgressBar callback:")
pb_callbacks = Callback.init_all([{ProgressBar, [log_interval: 10]}])
pb_state = %{state | pipeline: pipeline}
{_, pb_state, pb_callbacks} = Callback.run(pb_callbacks, :on_epoch_begin, pb_state)

{pb_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.with_index(batches), {pb_state, pb_callbacks}, fn {batch, idx}, {st, cbs} ->
    {new_t, metrics} = Imitation.train_step(st.trainer, batch, nil)
    loss = Nx.to_number(metrics.loss)
    st = %{st | trainer: new_t, step: st.step + 1, batch_idx: idx,
      batch_metrics: %{loss: loss}, epoch_losses: [loss | st.epoch_losses]}
    st = Callback.increment_event_count(st, :on_batch_end)
    {_, st, cbs} = Callback.run(cbs, :on_batch_end, st)
    if rem(idx, 100) == 0, do: :erlang.garbage_collect()
    {st, cbs}
  end)
end)
IO.write(:stderr, "\n")
pb_ms = pb_us / 200_000
Output.puts("   #{Float.round(pb_ms, 2)}ms/iter (+#{Float.round(pb_ms - raw_ms, 2)}ms overhead)")

# 5. With all training callbacks
Output.puts("5. With all callbacks (ProgressBar + GracefulShutdown + Checkpoint):")
all_callbacks = Callback.init_all([
  {GracefulShutdown, [checkpoint_path: opts[:checkpoint]]},
  {ProgressBar, [log_interval: 10]},
  {Checkpoint, [save_best: false, checkpoint_path: nil]}
])
all_state = %{state | pipeline: pipeline}
{_, all_state, all_callbacks} = Callback.run(all_callbacks, :on_train_begin, all_state)
{_, all_state, all_callbacks} = Callback.run(all_callbacks, :on_epoch_begin, all_state)

{all_us, _} = :timer.tc(fn ->
  Enum.reduce(Enum.with_index(batches), {all_state, all_callbacks}, fn {batch, idx}, {st, cbs} ->
    {new_t, metrics} = Imitation.train_step(st.trainer, batch, nil)
    loss = Nx.to_number(metrics.loss)
    st = %{st | trainer: new_t, step: st.step + 1, batch_idx: idx,
      batch_metrics: %{loss: loss}, epoch_losses: [loss | st.epoch_losses]}
    st = Callback.increment_event_count(st, :on_batch_end)
    {_, st, cbs} = Callback.run(cbs, :on_batch_end, st)
    if rem(idx, 100) == 0, do: :erlang.garbage_collect()
    {st, cbs}
  end)
end)
IO.write(:stderr, "\n")
all_ms = all_us / 200_000
Output.puts("   #{Float.round(all_ms, 2)}ms/iter (+#{Float.round(all_ms - raw_ms, 2)}ms overhead)")

Output.puts("\n=== Summary ===")
Output.puts("  Raw train_step:     #{Float.round(raw_ms, 2)}ms")
Output.puts("  + State mgmt:      +#{Float.round(state_ms - raw_ms, 2)}ms")
Output.puts("  + Callback dispatch:+#{Float.round(dispatch_ms - state_ms, 2)}ms")
Output.puts("  + ProgressBar:      +#{Float.round(pb_ms - dispatch_ms, 2)}ms")
Output.puts("  + All callbacks:    +#{Float.round(all_ms - dispatch_ms, 2)}ms")
Output.puts("  TOTAL:              #{Float.round(all_ms, 2)}ms")
