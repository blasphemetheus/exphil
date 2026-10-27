# AI-generated; method and attribution: docs/guides/MAMBA_BATCH_PARITY.md.
# Compare identical replay observations and supported phase schedules across builds.
[baseline, candidate, output] = System.argv()
read = fn path -> path |> File.read!() |> Jason.decode!() end
left = read.(baseline)
right = read.(candidate)

unless left["hashes"] == right["hashes"] and left["stagger"] == right["stagger"] and
         left["client"] == right["client"],
       do: raise("inputs, source, backend, or schedule differ")

unless length(left["events"]) == length(right["events"]), do: raise("frame counts differ")

checks =
  for {a, b} <- Enum.zip(left["events"], right["events"]) do
    unless a["tick"] == b["tick"], do: raise("frame order differs")

    features =
      if a["features"] do
        unless length(a["features"]) == length(b["features"]), do: raise("feature shape differs")
        values = Enum.zip(a["features"], b["features"])

        %{
          max_abs: values |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max(),
          close: Enum.all?(values, fn {x, y} -> abs(x - y) <= 1.0e-4 + 1.0e-4 * abs(x) end)
        }
      end

    %{
      tick: a["tick"],
      controllers_equal: a["controllers"] == b["controllers"],
      features: features
    }
  end

passed = Enum.all?(checks, &(&1.controllers_equal and (is_nil(&1.features) or &1.features.close)))

report = %{
  passed: passed,
  baseline: baseline,
  candidate: candidate,
  checks: checks,
  controller_frames_differ: Enum.count(checks, &(not &1.controllers_equal)),
  sampled_feature_checks: Enum.count(checks, & &1.features),
  atol: 1.0e-4,
  rtol: 1.0e-4,
  scope:
    "Deterministic trained-policy inference on identical replay frames. Does not validate training updates, fused scans, or match quality."
}

File.write!(output, Jason.encode!(report, pretty: true))

IO.puts(
  "Framework inference comparison passed=#{passed}; #{report.controller_frames_differ} controller frames differ"
)

unless passed, do: System.halt(1)
