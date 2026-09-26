# Replay saved real inputs without reparsing. Mode shapes varies partial-batch
# sizes; mode fixed repeatedly trains the identical full-size batch.
alias ExPhil.Training.{Imitation, Output}
[capture, out, mode | args] = System.argv()
steps = case args do [n] -> String.to_integer(n); [] -> 30 end
File.mkdir_p!(out)
saved = File.read!(Path.join(capture, "before_step.axon")) |> :erlang.binary_to_term()
cfg = Map.put(saved.config, :button_pos_weight, Nx.tensor(List.duplicate(1.0, 8), backend: Nx.BinaryBackend))
trainer = Imitation.new(Map.to_list(cfg))
{:ok, restored} = Imitation.load_checkpoint(trainer, Path.join(capture, "before_step.axon"))
trainer = %{restored | config: cfg}
batch = File.read!(Path.join(capture, "batch.bin")) |> :erlang.binary_to_term()
defmodule MambaCrashProbe do
  def slice(%Nx.Tensor{} = t, n), do: Nx.slice_along_axis(t, 0, n, axis: 0)
  def slice(m, n) when is_map(m), do: Map.new(m, fn {k, v} -> {k, slice(v, n)} end)
  def slice(v, _), do: v
end
Output.banner("Mamba crash probe: #{mode}")
trainer = Enum.reduce(1..steps, trainer, fn step, tr ->
  n = if mode == "shapes", do: 129 - step, else: 128
  if n < 1, do: raise("shape sweep limited to 128 steps")
  b = MambaCrashProbe.slice(batch, n)
  # Last input and periodic pre-update state survive a native abort.
  File.write!(Path.join(out, "last_batch.bin"), :erlang.term_to_binary(b))
  File.write!(Path.join(out, "last_step.json"), Jason.encode!(%{step: step, batch_size: n, mode: mode}))
  if step == 1 or rem(step, 1000) == 0 do
    :ok = Imitation.save_checkpoint(tr, Path.join(out, "before_step.axon"))
  end
  Output.puts("starting step #{step}, batch #{n}")
  {t, metrics} = Imitation.train_step(tr, b, nil)
  loss = Nx.to_number(metrics.loss)
  unless is_number(loss), do: raise("nonfinite loss #{inspect(loss)}")
  Output.puts("completed step #{step}, loss #{loss}")
  t
end)
:ok = Imitation.save_checkpoint(trainer, Path.join(out, "completed.axon"))
Output.success("Completed #{steps} steps")
