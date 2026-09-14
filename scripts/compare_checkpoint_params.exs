# Recursive parameter-tree equality of two policy checkpoints (BinaryBackend):
#   mix run --no-start scripts/compare_checkpoint_params.exs A.bin B.bin
# Prints tensor counts, every tensor with a nonzero max |a-b| (or a shape/key
# mismatch), and the overall max. Serialized file hashes differ across exports
# of identical weights (metadata), so compare the trees, not the files.

alias ExPhil.Training.Checkpoint
[pa, pb] = System.argv()
load = fn p -> {:ok, e} = Nx.with_default_backend(Nx.BinaryBackend, fn -> Checkpoint.load_policy(p) end); e.params end
defmodule Flat do
  def flat(%Nx.Tensor{} = t, prefix), do: [{prefix, t}]
  def flat(m, prefix) when is_map(m) and not is_struct(m),
    do: Enum.flat_map(m, fn {k, v} -> flat(v, "#{prefix}.#{k}") end)
  def flat(%{data: data}, prefix), do: flat(data, prefix)
  def flat(_, _), do: []
end
a = Map.new(Flat.flat(ExPhil.Training.Utils.ensure_model_state(load.(pa)), "")); b = Map.new(Flat.flat(ExPhil.Training.Utils.ensure_model_state(load.(pb)), ""))
IO.puts("tensors: #{map_size(a)} vs #{map_size(b)}; same keys: #{Enum.sort(Map.keys(a)) == Enum.sort(Map.keys(b))}")
diffs = for {k, t} <- a do
  u = b[k]
  if u != nil and Nx.shape(t) == Nx.shape(u),
    do: {k, Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(Nx.as_type(t, :f32), Nx.as_type(u, :f32)))))},
    else: {k, {:shape_or_missing, Nx.shape(t)}}
end
IO.inspect(Enum.reject(diffs, fn {_, d} -> d == 0 or d == 0.0 end), label: "nonzero diffs")
IO.puts("max |a-b|: #{diffs |> Enum.map(fn {_, d} -> if is_number(d), do: d, else: :infinity end) |> Enum.max()}")
