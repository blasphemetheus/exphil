# Shield-run forensics for the "constant 215" mystery (WS1, 2026-07-17).
# Hypothesis under test (H1): shield-run p95 pins at ~215f across rounds
# because 215f is Melee's HARD-SHIELD-BREAK ceiling (~3.58s from full 60HP)
# and the prev-action shield-lock rides shield all the way to break
# (INTERP_ROADMAP.md:323: ablation dropped max hold 215f->16f).
#
#   mix run scripts/shield_run_stats.exs <replay.slp> [more.slp ...]
#
# Per replay, port 1 (the policy in probes):
#   - shield-run length stats over [178,179,180] (EXACTLY the gate's set —
#     ReplayStats.shield_stats; ReportCard's 178..182 is a different set
#     used only for OOS classification)
#   - tail histogram (buckets 180-240+)
#   - for every run >= 200f: the action state on the frame AFTER the run
#     ends. Successors in the shield-break family (205..211: ShieldBreakFly/
#     Fall/Down/Stand + FuraFura dizzy) = CONFIRMED break.

alias ExPhil.Interp.ReplayStats

shield_set = MapSet.new([178, 179, 180])
break_family = MapSet.new(205..211)

# Runs WITH end positions: [{run_length, successor_action | nil}]
runs_with_successors = fn actions ->
  actions
  |> Enum.with_index()
  |> Enum.chunk_by(fn {a, _i} -> MapSet.member?(shield_set, a) end)
  |> Enum.filter(fn [{a, _} | _] -> MapSet.member?(shield_set, a) end)
  |> Enum.map(fn chunk ->
    {_, last_i} = List.last(chunk)
    {length(chunk), Enum.at(actions, last_i + 1)}
  end)
end

pct = fn xs, p ->
  case Enum.sort(xs) do
    [] -> nil
    s -> Enum.at(s, min(length(s) - 1, floor(p * length(s))))
  end
end

totals = %{runs: 0, long: 0, breaks: 0, tail: []}

totals =
  Enum.reduce(System.argv(), totals, fn path, acc ->
    data =
      try do
        ReplayStats.load(path)
      rescue
        e ->
          IO.puts("\n== #{Path.basename(path)} — UNPARSEABLE, skipped (#{Exception.message(e)})")
          nil
      end

    if data == nil do
      acc
    else
    runs = runs_with_successors.(data.p1.actions)
    lengths = Enum.map(runs, &elem(&1, 0))

    long = Enum.filter(runs, fn {len, _} -> len >= 200 end)

    breaks =
      Enum.filter(long, fn {_, succ} ->
        succ != nil and MapSet.member?(break_family, succ)
      end)

    tail_hist =
      lengths
      |> Enum.filter(&(&1 >= 180))
      |> Enum.frequencies_by(fn l -> min(div(l, 10) * 10, 240) end)
      |> Enum.sort()

    IO.puts("\n== #{Path.basename(path)} (#{data.n} frames)")

    IO.puts(
      "  shield runs: #{length(runs)}  p50=#{pct.(lengths, 0.5)} p95=#{pct.(lengths, 0.95)} max=#{Enum.max(lengths, fn -> 0 end)}"
    )

    IO.puts("  tail histogram (>=180f, 10f buckets, 240=240+): #{inspect(tail_hist)}")

    for {len, succ} <- long do
      verdict = if succ && MapSet.member?(break_family, succ), do: "BREAK", else: "no-break"
      IO.puts("  run #{len}f -> successor action #{inspect(succ)} [#{verdict}]")
    end

      %{
        runs: acc.runs + length(runs),
        long: acc.long + length(long),
        breaks: acc.breaks + length(breaks),
        tail: acc.tail ++ Enum.filter(lengths, &(&1 >= 200))
      }
    end
  end)

IO.puts("\n== AGGREGATE across #{length(System.argv())} replays")
IO.puts("  total shield runs: #{totals.runs}")
IO.puts("  runs >= 200f: #{totals.long}; confirmed BREAK successors: #{totals.breaks}")
IO.puts("  >=200f lengths sorted: #{inspect(Enum.sort(totals.tail))}")

cond do
  totals.long == 0 ->
    IO.puts("  VERDICT: no long runs at all — 215 came from percentile math on shorter tails?")

  totals.breaks / max(totals.long, 1) >= 0.5 ->
    IO.puts("  VERDICT: H1 CONFIRMED — long shields terminate in BREAKS (hard-shield ceiling ~215f)")

  true ->
    IO.puts("  VERDICT: H1 NOT confirmed — long runs end without breaks; inspect successors above")
end
