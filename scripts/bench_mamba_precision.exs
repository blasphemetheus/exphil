# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
Code.require_file("lib/mamba_precision_experiment.exs", __DIR__)

{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [capture: :string, mode: :string, out: :string])

data = opts[:capture] |> File.read!() |> Nx.deserialize()
MambaPrecisionExperiment.install(opts[:mode] || "baseline")

rows =
  for n <- [1, 8] do
    state = Map.new(data.initial, fn {k, v} -> {k, Nx.slice_along_axis(v, 0, n, axis: 0)} end)

    frames =
      data.frames |> Nx.slice_along_axis(0, n, axis: 0) |> Nx.slice_along_axis(0, 79, axis: 1)

    x = Nx.take(frames, 0, axis: 1)
    step = Nx.Defn.jit(&Edifice.SSM.Mamba.step/3, compiler: EXLA, precision: :highest)
    replay = Edifice.Stateful.jit_replay(Edifice.SSM.Mamba, EXLA, precision: :highest)

    for {label, call} <- [
          {:step,
           fn ->
             {y, _} = step.(data.params, state, x)
             Nx.to_binary(y)
           end},
          {:replay,
           fn ->
             s = replay.(data.params, state, frames)
             Nx.to_binary(s.conv)
           end}
        ] do
      for _ <- 1..4, do: call.()

      times =
        for _ <- 1..40 do
          {us, _} = :timer.tc(call)
          us
        end
        |> Enum.sort()

      %{rows: n, operation: label, median_us: Enum.at(times, 20), p95_us: Enum.at(times, 37)}
    end
  end
  |> List.flatten()

report = %{mode: opts[:mode] || "baseline", client: EXLA.Client.default_name(), results: rows}
File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true))
ExPhil.Training.Output.puts(Jason.encode!(report, pretty: true))
