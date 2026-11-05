# AI-generated. Authorship and method: eval_runs/mamba_learning_20261105/README.md.
defmodule ExPhil.LearningDataCache do
  alias ExPhil.Training.Output

  # Use only a private cache made by this runner. Prepared tensors must already
  # be on BinaryBackend; serializing device references is not portable.
  def fetch(nil, _signature, build), do: build.()

  def fetch(directory, signature, build) do
    Application.ensure_all_started(:crypto)
    key = :crypto.hash(:sha256, :erlang.term_to_binary(signature)) |> Base.encode16(case: :lower)
    File.mkdir_p!(directory)
    File.chmod!(directory, 0o700)
    path = Path.join(directory, key <> ".term")

    if File.exists?(path) do
      Output.puts("Loading prepared data from #{path}")
      {^key, value} = path |> File.read!() |> :erlang.binary_to_term([:safe])
      value
    else
      value = build.() |> to_host()
      Output.puts("Saving prepared data to #{path}")
      tmp = path <> ".tmp"
      File.write!(tmp, :erlang.term_to_binary({key, value}))
      File.rename!(tmp, path)
      value
    end
  end

  defp to_host(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

  defp to_host(map) when is_map(map),
    do: Map.new(:maps.to_list(map), fn {k, v} -> {k, to_host(v)} end)

  defp to_host(list) when is_list(list), do: Enum.map(list, &to_host/1)

  defp to_host(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_host/1) |> List.to_tuple()

  defp to_host(value) when is_reference(value) or is_pid(value) or is_function(value),
    do: raise(ArgumentError, "Prepared cache contains a process-local value")

  defp to_host(value), do: value
end
