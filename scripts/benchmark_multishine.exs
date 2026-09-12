alias ExPhil.Eval.MultishineBenchmark

case System.argv() do
  [manifest, output] ->
    report = MultishineBenchmark.run_manifest(manifest)
    File.write!(output, Jason.encode!(report, pretty: true) <> "\n", [:exclusive])

    Enum.each(report.runs, fn run ->
      IO.puts(
        "#{run.id} #{run.scenario}: #{Float.round(run.metrics.ground_shines_per_minute, 1)}/min " <>
          "chain=#{run.metrics.max_chain} recovery=#{run.metrics.recovery.completed} completed/" <>
          "#{run.metrics.recovery.censored} censored"
      )
    end)

  _ ->
    IO.puts(
      :stderr,
      "usage: mix run scripts/benchmark_multishine.exs MANIFEST.json NEW_REPORT.json"
    )

    System.halt(2)
end
