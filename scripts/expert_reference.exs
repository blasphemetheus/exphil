# Expert reference for the fidelity scorecard (2026-10-02).
#
# Computes ExPhil.Eval.PlayStats for the Fox player over expert replays on one
# stage, plus the SPLIT-HALF distance (games alternately assigned to two
# halves): the noise floor any model-vs-expert distance must be read against.
#
#   mix run scripts/expert_reference.exs [--split SPLIT.json] [--stage 32]
#     [--max-games 150] [--character fox] [--out eval_runs/1002_fidelity/expert_fd.json]
#     [--silence-map-out eval_runs/1002_fidelity/expert_silence_map_fd.json]
#
# --silence-map-out also writes the expert ExPhil.Eval.SilenceMap (input-change
# hazards by situation, with split-half hazards as the noise floor; 10-06).
alias ExPhil.Data.Peppi
alias ExPhil.Eval.{PlayStats, SilenceMap}
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [split: :string, stage: :integer, max_games: :integer, character: :string, out: :string, silence_map_out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

split = (opts[:split] || "checkpoints/coh_base/split.json") |> File.read!() |> Jason.decode!()
stage = opts[:stage] || 32
char = opts[:character] || "fox"
out = opts[:out] || "eval_runs/1002_fidelity/expert_fd.json"
edge = GA.stage_edge(stage)
files = split["validation"] ++ split["train"]

Output.banner("Expert reference: #{char} on stage #{stage}")

chosen =
  files
  |> Task.async_stream(fn p -> with {:ok, m} <- Peppi.metadata(p), do: {p, m} end, max_concurrency: 16, timeout: 60_000)
  |> Enum.flat_map(fn
    {:ok, {p, %{stage: ^stage, players: [_, _] = ps}}} ->
      case Enum.filter(ps, &(String.downcase(&1.character_name || "") == char)) do
        [own] -> [{p, own, Enum.find(ps, &(&1.port != own.port))}]
        _ -> []
      end

    _ ->
      []
  end)
  |> Enum.take(opts[:max_games] || 150)

Output.puts("#{length(chosen)} games (of #{length(files)} scanned); opponents: " <>
  inspect(Enum.frequencies_by(chosen, fn {_, _, o} -> o.character_name end)))

per_game =
  chosen
  |> Task.async_stream(
    fn {path, own, opp} ->
      {:ok, replay} = Peppi.parse(path, player_port: own.port)

      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&%{own: &1.game_state.players[own.port], opp: &1.game_state.players[opp.port], controller: &1.controller})
      |> then(fn frames -> {PlayStats.from_game(frames, edge), SilenceMap.from_game(frames, stage: stage, edge: edge)} end)
    end,
    max_concurrency: 8,
    timeout: 300_000
  )
  |> Enum.map(fn {:ok, s} -> s end)

silence_games = Enum.map(per_game, &elem(&1, 1))
per_game = Enum.map(per_game, &elem(&1, 0))

total = Enum.reduce(per_game, PlayStats.empty(), &PlayStats.merge/2)
{half_a, half_b} =
  per_game
  |> Enum.with_index()
  |> Enum.reduce({PlayStats.empty(), PlayStats.empty()}, fn {s, i}, {a, b} ->
    if rem(i, 2) == 0, do: {PlayStats.merge(a, s), b}, else: {a, PlayStats.merge(b, s)}
  end)

summary = PlayStats.summarize(total)
sa = PlayStats.summarize(half_a)
sb = PlayStats.summarize(half_b)
floor = PlayStats.compare(sa, sb)

File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(%{character: char, stage: stage, games: length(per_game), summary: summary,
  split_half_distance: floor, half_rates: [sa.rates, sb.rates]}, pretty: true))

Output.puts("RESULT expert #{char} stage #{stage}: #{length(per_game)} games, #{summary.rates["minutes"]} min")
for {k, v} <- Enum.sort(summary.rates) do
  Output.puts("  #{String.pad_trailing(k, 26)} #{inspect(v)}   (halves #{inspect(sa.rates[k])} / #{inspect(sb.rates[k])})")
end
Output.puts("split-half distances (noise floor): " <> Enum.map_join(Enum.sort(floor), "  ", fn {k, v} -> "#{k} #{v}" end))
Output.puts("jump_peak hist: #{inspect(Enum.sort_by(summary.hists["jump_peak"], fn {k, _} -> String.to_integer(k) end))}")
Output.puts("landing_lag hist: #{inspect(Enum.sort_by(summary.hists["landing_lag"], fn {k, _} -> String.to_integer(k) end))}")
Output.success("wrote #{out}")

if sm_out = opts[:silence_map_out] do
  pool = fn games -> games |> Enum.reduce(SilenceMap.empty(), &SilenceMap.merge/2) |> SilenceMap.summarize() end
  {ga, gb} = silence_games |> Enum.with_index() |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)
  all = pool.(silence_games)
  File.write!(sm_out, Jason.encode!(%{character: char, stage: stage, games: length(silence_games), summary: all,
    halves: [pool.(Enum.map(ga, &elem(&1, 0))), pool.(Enum.map(gb, &elem(&1, 0)))]}, pretty: true))
  states = all |> Enum.filter(fn {k, _} -> String.starts_with?(k, "state:") or String.starts_with?(k, "age:") end) |> Enum.sort()
  Output.puts("RESULT expert silence map enter_silence by state: " <>
    Enum.map_join(states, "  ", fn {k, v} -> "#{k} #{v.enter_silence} (n=#{v.active})" end))
  Output.success("wrote #{sm_out}")
end
