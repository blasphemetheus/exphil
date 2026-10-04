# Fidelity scorecard over LIVE .slp replays (2026-10-04): the same
# ExPhil.Eval.PlayStats the sim scorecard computes, over games the bot played
# against a human, compared with the expert reference. Puts a live session on
# the same table as `fidelity_scorecard.exs` so the sim's prediction can be
# checked against the replay record.
#
#   mix run scripts/live_scorecard.exs --label L [--bot-port 1] [--stage 32]
#     [--reference eval_runs/1002_fidelity/expert_fd.json] [--min-bytes 150000]
#     [--out FILE.json] GAME.slp ...
#
# Games on another stage than --stage are skipped (the reference is per
# stage); stubs under --min-bytes are skipped (CSS restarts).
alias ExPhil.{Data.Peppi, Eval.PlayStats, Training.Output}
alias ExPhil.Sim.GA

{opts, files, bad} =
  OptionParser.parse(System.argv(),
    strict: [label: :string, bot_port: :integer, stage: :integer, reference: :string, min_bytes: :integer, out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
label = opts[:label] || "live"
bot_port = opts[:bot_port] || 1
stage = opts[:stage] || 32
min_bytes = opts[:min_bytes] || 150_000
ref = (opts[:reference] || "eval_runs/1002_fidelity/expert_fd.json") |> File.read!() |> Jason.decode!()
edge = GA.stage_edge(stage)

Output.banner("Live scorecard: #{label}")

games =
  files
  |> Enum.filter(&(File.stat!(&1).size >= min_bytes))
  |> Enum.flat_map(fn path ->
    case Peppi.metadata(path) do
      {:ok, %{stage: ^stage, players: ps}} ->
        case Enum.find(ps, &(&1.port == bot_port)) do
          nil -> []
          own -> [{path, own, Enum.find(ps, &(&1.port != bot_port))}]
        end

      {:ok, %{stage: s}} ->
        Output.warning("skip #{Path.basename(path)}: stage #{s} (reference is stage #{stage})")
        []

      _ ->
        []
    end
  end)

per_game =
  games
  |> Task.async_stream(
    fn {path, own, opp} ->
      {:ok, replay} = Peppi.parse(path, player_port: own.port)

      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&%{own: &1.game_state.players[own.port], opp: &1.game_state.players[opp.port], controller: &1.controller})
      |> PlayStats.from_game(edge)
    end,
    max_concurrency: 8,
    timeout: 300_000
  )
  |> Enum.map(fn {:ok, s} -> s end)

# same derived rates + headline mean as fidelity_scorecard.exs
derived = fn s ->
  lag = s["hists"]["landing_lag"] || %{}
  p = fn ks -> Enum.sum(Enum.map(ks, &Map.get(lag, "#{&1}", 0.0))) end
  hit = p.([7, 9, 10, 11])
  miss = p.([15, 18, 20, 22])
  peaks = s["hists"]["jump_peak"] || %{}
  low = peaks |> Enum.filter(fn {k, _} -> String.to_integer(k) in 5..19 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  real = peaks |> Enum.filter(fn {k, _} -> String.to_integer(k) >= 5 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  rr = fn a, b -> if b == 0 or b == 0.0, do: nil, else: Float.round(a / b, 3) end
  %{"l_cancel_rate" => rr.(hit, hit + miss), "short_hop_share" => rr.(low, real)}
end
summary =
  per_game |> Enum.reduce(PlayStats.empty(), &PlayStats.merge/2) |> PlayStats.summarize()
  |> Jason.encode!() |> Jason.decode!()
summary = %{summary | "rates" => Map.merge(summary["rates"], derived.(summary))}
dist = PlayStats.compare(%{hists: summary["hists"]}, %{hists: ref["summary"]["hists"]})
headline = ~w(action_group position stick_zone stick_dwell hold_mean landing_lag jump_peak)
mean_dist = Float.round(Enum.sum(Enum.map(headline, &(dist[&1] || 0.0))) / length(headline), 3)
floor = ref["split_half_distance"]
exp = Map.merge(ref["summary"]["rates"], derived.(ref["summary"]))

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{label: label, games: length(per_game), summary: summary, distances: dist,
    fidelity_distance: mean_dist, expert_rates: exp}, pretty: true))
end

Output.puts("RESULT #{label}: #{length(per_game)} live games, #{summary["rates"]["minutes"]} min")
Output.puts("RESULT #{label} fidelity distance (mean of #{length(headline)} histograms, 0 = expert-like): #{mean_dist}")
Output.puts("RESULT #{label} distances  " <>
  Enum.map_join(Enum.sort(dist), "  ", fn {k, v} -> "#{k} #{v} (floor #{floor[k]})" end))
Output.puts("RESULT #{label} rates model | expert  " <>
  Enum.map_join(Enum.sort(summary["rates"]), "  ", fn {k, v} -> "#{k} #{inspect(v)}|#{inspect(exp[k])}" end))
