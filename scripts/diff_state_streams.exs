# State-stream reconciliation (task #8 phase 1, GOTCHAS #81).
#
# Diffs the PARSED stream (what Peppi reads out of a .slp — the space every
# policy is trained in) against the LIVE stream (what the libmelee bridge
# reports at inference) for the SAME run, and prints the per-action
# action_frame mapping between them.
#
# The diff math lives in ExPhil.Eval.StateStreamDiff (unit-tested, and the
# mapping is pinned by test/exphil/eval/state_stream_diff_test.exs). This
# script is the CLI printer.
#
#   # both committed pairs
#   mix run scripts/diff_state_streams.exs
#
#   # one explicit pair
#   mix run scripts/diff_state_streams.exs --slp a.slp --trace a.live-trace.log
#
#   # options: --port N (default 1), --json
#
# A pair is only valid if the .slp and the trace come from the SAME run; see
# test/fixtures/statestream/README.md.

alias ExPhil.Eval.StateStreamDiff

{opts, _rest, _} =
  OptionParser.parse(System.argv(),
    strict: [slp: :string, trace: :string, port: :integer, json: :boolean]
  )

port = Keyword.get(opts, :port, 1)
fixture_dir = "test/fixtures/statestream"

pairs =
  case {opts[:slp], opts[:trace]} do
    {nil, nil} ->
      fixture_dir
      |> Path.join("*.slp")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(fn slp ->
        {Path.basename(slp, ".slp"), slp, Path.rootname(slp) <> ".live-trace.log"}
      end)

    {slp, trace} when is_binary(slp) and is_binary(trace) ->
      [{Path.basename(slp, ".slp"), slp, trace}]

    _ ->
      IO.puts(:stderr, "error: --slp and --trace must be given together")
      System.halt(2)
  end

if pairs == [] do
  IO.puts(:stderr, "error: no pairs found in #{fixture_dir}")
  System.halt(2)
end

pct = fn r -> "#{Float.round(r * 100, 1)}%" end

reports =
  Enum.map(pairs, fn {name, slp, trace} ->
    case StateStreamDiff.diff(slp, trace, port: port) do
      {:ok, report} ->
        unless opts[:json] do
          IO.puts("\n=== #{name} ===")
          IO.puts("  parsed frame = trace frame + #{report.offset}")
          IO.puts("  frames compared: #{report.frames_compared}")

          IO.puts(
            "  alignment: action #{pct.(report.agreement.action)}, " <>
              "on_ground #{pct.(report.agreement.on_ground)}, " <>
              "y #{pct.(report.agreement.y)} (tol #{report.y_tolerance})"
          )

          if report.agreement.action < 1.0 do
            IO.puts("  ** action agreement below 100% — alignment is suspect, mapping unreliable")
          end

          IO.puts("  action_frame agreement: #{pct.(report.agreement.action_frame)}")
          IO.puts("  fields that SHIFT: #{inspect(report.shifted_fields)}")

          IO.puts("\n  action | n    | parsed af | live af  | live - parsed")
          IO.puts("  -------+------+-----------+----------+--------------")

          report.mapping
          |> Enum.sort()
          |> Enum.each(fn {act, m} ->
            d = if m.consistent?, do: "#{m.delta}", else: "MIXED #{inspect(m.deltas)}"

            IO.puts(
              "  #{String.pad_leading(to_string(act), 6)} |" <>
                " #{String.pad_leading(to_string(m.n), 4)} |" <>
                " #{String.pad_leading("#{m.parsed_af.first}..#{m.parsed_af.last}", 9)} |" <>
                " #{String.pad_leading("#{m.live_af.first}..#{m.live_af.last}", 8)} |" <>
                " #{d}"
            )
          end)

          if report.inconsistent_actions != [] do
            IO.puts(
              "\n  ** actions with a NON-constant offset: #{inspect(report.inconsistent_actions)}"
            )

            IO.puts(
              "     a per-action table cannot express these — investigate before relying on it"
            )
          end
        end

        {name, report}

      {:error, reason} ->
        IO.puts(:stderr, "#{name}: FAILED — #{inspect(reason)}")
        {name, nil}
    end
  end)

ok = Enum.reject(reports, fn {_, r} -> is_nil(r) end)

if opts[:json] do
  ok
  |> Map.new(fn {name, r} ->
    {name,
     %{
       offset: r.offset,
       frames_compared: r.frames_compared,
       agreement: r.agreement,
       shifted_fields: r.shifted_fields,
       inconsistent_actions: r.inconsistent_actions,
       mapping:
         Map.new(r.mapping, fn {act, m} ->
           {act,
            %{
              delta: m.delta,
              n: m.n,
              parsed_af: [m.parsed_af.first, m.parsed_af.last],
              live_af: [m.live_af.first, m.live_af.last]
            }}
         end)
     }}
  end)
  |> Jason.encode!(pretty: true)
  |> IO.puts()
else
  # Cross-pair consistency: an action whose delta differs BETWEEN runs would
  # mean the convention is context-dependent, not a fixed table.
  merged =
    ok
    |> Enum.flat_map(fn {_, r} -> Enum.map(r.mapping, fn {act, m} -> {act, m.delta} end) end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {act, ds} -> {act, Enum.uniq(ds)} end)
    |> Enum.sort()

  conflicts = Enum.filter(merged, fn {_, ds} -> length(ds) > 1 end)

  IO.puts("\n=== combined (#{length(ok)} pair(s)) ===")

  if conflicts == [] do
    IO.puts("  every action has ONE delta across all pairs — mapping is consistent")

    zero = merged |> Enum.filter(fn {_, [d]} -> d == 0 end) |> Enum.map(&elem(&1, 0))
    one = merged |> Enum.filter(fn {_, [d]} -> d == 1 end) |> Enum.map(&elem(&1, 0))
    other = merged |> Enum.reject(fn {_, [d]} -> d in [0, 1] end)

    IO.puts("  live af == parsed af      : #{inspect(zero)}")
    IO.puts("  live af == parsed af + 1  : #{inspect(one)}")
    if other != [], do: IO.puts("  other deltas: #{inspect(other)}")
  else
    IO.puts("  ** CONFLICT — these actions disagree between pairs: #{inspect(conflicts)}")
  end
end
