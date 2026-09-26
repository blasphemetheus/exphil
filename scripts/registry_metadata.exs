# Read-only checkpoint inspection. Run with existing BEAM modules, not Mix.
Application.put_env(:nx, :default_backend, Nx.BinaryBackend)
Application.ensure_all_started(:nx)
Logger.configure(level: :error)

defmodule RegistryMetadata do
  def clean(%Nx.Tensor{}), do: "[tensor omitted]"
  def clean(m) when is_map(m), do: Map.new(Map.drop(m, [:__struct__, :port_map, :player_tag_map, :replay_files, :embed_canary]), fn {k, v} -> {to_string(k), clean(v)} end)
  def clean(l) when is_list(l), do: Enum.map(l, &clean/1)
  def clean(t) when is_tuple(t), do: t |> Tuple.to_list() |> clean()
  def clean(x) when is_binary(x) or is_number(x) or is_boolean(x) or is_nil(x), do: x
  def clean(x) when is_atom(x), do: Atom.to_string(x)
  def clean(_), do: "[non-JSON value omitted]"

  def inspect_file(path) do
    try do
      cond do
        String.ends_with?(path, ".axon") ->
          # A training-only ancestor is identified by its exact file/hash.
          # Its sidecar config is handled separately; never label it playable.
          %{path: path, kind: "training_checkpoint", config: %{}}
        String.match?(Path.basename(path), ~r/^head_iter\d+\.bin$/) ->
          head = path |> File.read!() |> :erlang.binary_to_term()
          true = is_map(head.ar)
          %{path: path, kind: "ppo_head", config: %{}, parent_path: head.policy, iteration: head.iter}
        true ->
          {:ok, export} = ExPhil.Training.Checkpoint.load_policy(path)
          true = is_map(export.params)
          %{path: path, kind: "policy", config: clean(export.config)}
      end
    rescue
      e -> %{path: path, error: Exception.message(e)}
    end
  end
end

[input, output] = System.argv()
paths = input |> File.read!() |> Jason.decode!()
File.open!(output, [:write], fn io ->
  paths |> Enum.with_index(1) |> Enum.each(fn {path, i} ->
    IO.write(io, Jason.encode!(RegistryMetadata.inspect_file(path)) <> "\n")
    if rem(i, 100) == 0, do: IO.puts("Inspected #{i}/#{length(paths)} artifacts")
    :erlang.garbage_collect()
  end)
end)
