# Small real-input concurrency reproducer; no 64-game chunk parsing.
# Args: capture directory, output directory, cpu|gpu, number of training steps.
alias ExPhil.Training.{Imitation, Output}
alias ExPhil.Data.Peppi
[capture, out, mode, steps_s] = System.argv()
unless mode in ["cpu", "gpu"], do: raise("mode must be cpu or gpu")
steps = String.to_integer(steps_s)
File.mkdir_p!(out)
saved = capture |> Path.join("before_step.axon") |> File.read!() |> :erlang.binary_to_term()
cfg = Map.put(saved.config, :button_pos_weight,
  Nx.tensor(List.duplicate(1.0, 8), backend: Nx.BinaryBackend))
trainer = Imitation.new(Map.to_list(cfg))
{:ok, restored} = Imitation.load_checkpoint(trainer, Path.join(capture, "before_step.axon"))
trainer = %{restored | config: cfg}
batch = capture |> Path.join("batch.bin") |> File.read!() |> :erlang.binary_to_term()
paths = "checkpoints/fox_mamba_v1_20260925/split.json" |> File.read!() |> Jason.decode!()
path = hd(paths["train"])
{:ok, replay} = Peppi.parse(path)
states = replay |> Peppi.to_training_frames() |> Enum.take(1000) |> Enum.map(& &1.game_state)
embed_cfg = ExPhil.Embeddings.config_for_source([stage_internals: true], Peppi.provides())
Output.puts("Warmup full training step before starting concurrent embedding")
{trainer, metrics} = Imitation.train_step(trainer, batch, nil)
Nx.to_number(metrics.loss)
:ok = Imitation.save_checkpoint(trainer, Path.join(out, "before_concurrency.axon"))

defmodule MambaConcurrentEmbedding do
  def run(states, cfg, count) do
    receive do
      :stop -> count
    after
      0 ->
        tensor = ExPhil.Embeddings.Game.embed_states_fast(states, 1, config: cfg)
        # Synchronize and release each result, as replay precomputation does.
        Nx.to_binary(tensor)
        if rem(count, 100) == 0, do: :erlang.garbage_collect()
        run(states, cfg, count + 1)
    end
  end
end

backend = if mode == "cpu", do: Nx.BinaryBackend, else: EXLA.Backend
task = Task.async(fn ->
  Nx.with_default_backend(backend, fn -> MambaConcurrentEmbedding.run(states, embed_cfg, 0) end)
end)
Output.puts("Starting #{steps} training steps with concurrent #{mode} embedding")
trainer = Enum.reduce(1..steps, trainer, fn step, tr ->
  File.write!(Path.join(out, "last_step.json"), Jason.encode!(%{step: step, mode: mode}))
  {next, metrics} = Imitation.train_step(tr, batch, nil)
  loss = Nx.to_number(metrics.loss)
  unless is_number(loss), do: raise("nonfinite loss")
  if rem(step, 100) == 0, do: Output.puts("step #{step}: loss #{loss}")
  next
end)
send(task.pid, :stop)
iterations = Task.await(task, :infinity)
:ok = Imitation.save_checkpoint(trainer, Path.join(out, "completed.axon"))
File.write!(Path.join(out, "completed.json"), Jason.encode!(%{
  steps: steps, embedding_iterations: iterations, mode: mode, replay: path}))
Output.success("Completed #{steps} training steps and #{iterations} concurrent embedding batches")
