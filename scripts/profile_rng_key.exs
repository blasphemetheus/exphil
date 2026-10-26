# Attribute the fresh-key portion of stochastic controller latency. This does
# not change the deployed sampler; the host construction is a measured candidate.
alias ExPhil.Training.Output

fresh = fn ->
  <<seed::unsigned-32>> = :crypto.strong_rand_bytes(4)
  Nx.Random.key(seed)
end

host_key = fn seed ->
  Nx.tensor([0, seed], type: :u32, backend: Nx.BinaryBackend)
  |> Nx.backend_transfer({EXLA.Backend, client: EXLA.Client.default_name()})
end

for seed <- [0, 1, 42, 0xFFFFFFFF] do
  true = Nx.to_binary(host_key.(seed)) == Nx.to_binary(Nx.Random.key(seed))
end

candidate = fn ->
  <<seed::unsigned-32>> = :crypto.strong_rand_bytes(4)
  host_key.(seed)
end

results =
  for {label, fun} <- [current: fresh, host_constructed: candidate] do
    times =
      for i <- 1..210, reduce: [] do
        acc ->
          # Force completion, including the same small return transfer for both.
          {us, _} = :timer.tc(fn -> fun.() |> Nx.to_binary() end)
          if i > 10, do: [us | acc], else: acc
      end
      |> Enum.sort()

    result = %{
      kind: label,
      samples: 200,
      median_us: Enum.at(times, 100),
      p95_us: Enum.at(times, 189),
      times_us: times
    }

    Output.puts(inspect(Map.delete(result, :times_us)))
    result
  end

[out] = System.argv()

File.write!(
  out,
  Jason.encode!(
    %{
      client: EXLA.Client.default_name(),
      results: results,
      key_parity_seeds: [0, 1, 42, 0xFFFFFFFF],
      note:
        "Standalone key construction and return synchronization. Host candidate is not deployed. Both include OS randomness."
    },
    pretty: true
  )
)
