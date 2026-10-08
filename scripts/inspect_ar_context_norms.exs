Application.ensure_all_started(:nx)
Application.ensure_all_started(:axon)
Nx.default_backend(Nx.BinaryBackend)
for arm <- System.argv() do
  path = "checkpoints/coh_#{arm}/model_policy.bin"
  {params, _meta} = Edifice.Checkpoint.load(path, return_metadata: true)
  data = case params do %Axon.ModelState{data: d} -> d; %{data: d} -> d; m -> m end
  IO.puts("== #{arm}")
  for name <- Enum.sort(Map.keys(data)), String.starts_with?(name, "ar_") and (String.contains?(name, "embed") or String.contains?(name, "danger") or String.contains?(name, "context")) do
    for {k, v} <- data[name] do
      v = Nx.backend_transfer(v, Nx.BinaryBackend) |> Nx.as_type(:f32)
      IO.puts("  #{name}.#{k} #{inspect(Nx.shape(v))} mean|x|=#{Float.round(Nx.to_number(Nx.mean(Nx.abs(v))), 5)} max=#{Float.round(Nx.to_number(Nx.reduce_max(Nx.abs(v))), 4)}")
    end
  end
  if Map.has_key?(data, "ar_danger_hidden") do
    # per-input-column weight mass of the hidden layer: which danger feature does the readout lean on?
    kh = Nx.backend_transfer(data["ar_danger_hidden"]["kernel"], Nx.BinaryBackend) |> Nx.as_type(:f32)
    IO.puts("  hidden kernel row |.|-sum per column [y, jumps, on_ground, speed_y, ledge]: #{inspect(Nx.sum(Nx.abs(kh), axes: [1]) |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 3)))}")
  end
end
