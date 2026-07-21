# Score a checkpoint-ladder directory into Elo standings (task #19).
#
#   mix run --no-compile scripts/ladder_score.exs probes/ladder/<ts>
#
# Reads m###_<A>__vs__<B>/*.slp match dirs written by
# checkpoint_ladder.sh (A = port 1). Winner = more stocks on the final
# frame; equal stocks -> lower percent wins; still equal -> draw.
# Matches with no replay (timeout/crash before game start) are skipped
# and reported. Emits a standings table + ladder_results.json (full
# ledger, re-foldable by Eval.Elo.rate_all).
#
# NO-MIX: one beam, run only after every game BEAM is reaped (#67) —
# checkpoint_ladder.sh sequences this correctly.

alias ExPhil.Eval.Elo
alias ExPhil.Interp.ReplayStats
alias ExPhil.Training.Output

ladder_dir =
  case System.argv() do
    [dir | _] -> dir
    [] -> raise "usage: ladder_score.exs LADDER_DIR"
  end

match_dirs =
  ladder_dir
  |> Path.join("m*__vs__*")
  |> Path.wildcard()
  |> Enum.sort()

if match_dirs == [], do: raise("no match dirs under #{ladder_dir}")

Output.banner("Checkpoint ladder scoring")
Output.puts("#{length(match_dirs)} match dirs under #{ladder_dir}")

# Final-frame verdict from P1's perspective: 1.0 win / 0.5 draw / 0.0 loss
verdict = fn replay_path ->
  d = ReplayStats.load(replay_path)
  p1 = d.p1.players |> Enum.reverse() |> Enum.find(& &1)
  p2 = d.p2.players |> Enum.reverse() |> Enum.find(& &1)

  s1 = (p1 && trunc(p1.stock || 0)) || 0
  s2 = (p2 && trunc(p2.stock || 0)) || 0
  pct1 = (p1 && (p1.percent || 0.0)) || 0.0
  pct2 = (p2 && (p2.percent || 0.0)) || 0.0

  cond do
    s1 > s2 -> 1.0
    s1 < s2 -> 0.0
    pct1 < pct2 -> 1.0
    pct1 > pct2 -> 0.0
    true -> 0.5
  end
end

{ledger, skipped} =
  Enum.reduce(match_dirs, {[], []}, fn dir, {led, skip} ->
    [a, b] =
      dir
      |> Path.basename()
      |> String.replace(~r/^m\d+_/, "")
      |> String.split("__vs__", parts: 2)

    case Path.wildcard(Path.join(dir, "*.slp")) do
      [] ->
        {led, [Path.basename(dir) | skip]}

      slps ->
        # One game per dir expected; largest file wins if strays exist
        # (size, not mtime — GOTCHAS #69)
        slp = Enum.max_by(slps, fn f -> File.stat!(f).size end)
        {[{a, b, verdict.(slp), Path.basename(dir)} | led], skip}
    end
  end)

ledger = Enum.reverse(ledger)
skipped = Enum.reverse(skipped)

if skipped != [] do
  Output.warning("#{length(skipped)} matches had no replay (timeout/crash): #{inspect(skipped)}")
end

matches = Enum.map(ledger, fn {a, b, s, _} -> {a, b, s} end)
standings = Elo.standings(matches)

Output.puts("")
Output.puts("  #  rating   W-L-D   player")

standings
|> Enum.with_index(1)
|> Enum.each(fn {row, rank} ->
  Output.puts(
    "  #{rank}  #{row.rating}   #{row.wins}-#{row.losses}-#{row.draws}   #{row.player}"
  )
end)

out = Path.join(ladder_dir, "ladder_results.json")

json = %{
  ladder_dir: ladder_dir,
  engine: "mainline",
  matches:
    Enum.map(ledger, fn {a, b, s, dir} -> %{p1: a, p2: b, p1_score: s, match: dir} end),
  skipped: skipped,
  standings: standings
}

File.write!(out, Jason.encode!(json, pretty: true))
Output.success("ledger + standings -> #{out}")
