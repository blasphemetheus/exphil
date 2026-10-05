# Expert reference for the recovery-means scorecard (2026-10-05): the
# expert's choice of recovery tool per situation bucket (height x distance x
# jumps) over every offstage episode in the split's Fox games on one stage,
# plus the split-half mismatch / JS floor the model must be read against.
#
#   mix run scripts/expert_recovery_means.exs [--split SPLIT.json] [--stage 32]
#     [--max-games 150] [--character fox] [--out eval_runs/1002_fidelity/expert_recovery_means_fd.json]
alias ExPhil.Data.Peppi
alias ExPhil.Eval.RecoveryMeans
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [split: :string, stage: :integer, max_games: :integer, character: :string, out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

split = (opts[:split] || "checkpoints/coh_base/split.json") |> File.read!() |> Jason.decode!()
stage = opts[:stage] || 32
char = opts[:character] || "fox"
out = opts[:out] || "eval_runs/1002_fidelity/expert_recovery_means_fd.json"
edge = GA.stage_edge(stage)
files = split["validation"] ++ split["train"]

Output.banner("Expert recovery means: #{char} on stage #{stage}")

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

Output.puts("#{length(chosen)} games (of #{length(files)} scanned)")

episodes =
  chosen
  |> Task.async_stream(
    fn {path, own, opp} ->
      {:ok, replay} = Peppi.parse(path, player_port: own.port)

      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&%{own: &1.game_state.players[own.port], opp: &1.game_state.players[opp.port], controller: &1.controller})
      |> RecoveryMeans.episodes(edge)
    end,
    max_concurrency: 8,
    timeout: 300_000
  )
  |> Enum.flat_map(fn {:ok, eps} -> eps end)

table = RecoveryMeans.table(episodes)
self_score = RecoveryMeans.score(episodes, table)
floor = RecoveryMeans.split_half(episodes)

File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(%{
  "stage" => stage, "character" => char, "games" => length(chosen),
  "table" => table, "self_score" => self_score, "split_half" => floor
}, pretty: true))

# the episodes themselves (with traces) go in a sibling file so the
# reference table stays small and the model's died trips can be read
# against the expert's on the same situations
episodes_out = Path.rootname(out) <> "_episodes.json"
File.write!(episodes_out, Jason.encode!(
  Enum.map(episodes, &Map.new(&1, fn {k, v} -> {Atom.to_string(k), if(is_atom(v) and not is_boolean(v), do: Atom.to_string(v), else: v)} end))))
Output.puts("episodes with traces -> #{episodes_out}")

Output.puts("RESULT expert recovery means: #{length(episodes)} offstage episodes over #{length(chosen)} games")
Output.puts("RESULT expert first means: " <> Enum.map_join(Enum.sort_by(self_score["first_means"], &(-elem(&1, 1))), "  ", fn {k, n} -> "#{k} #{n}" end))
Output.puts("RESULT expert by height (n, return rate, side_b share): " <>
  Enum.map_join(~w(high ledge low deep), "  ", fn h ->
    b = self_score["by_height"][h] || %{"n" => 0, "return_rate" => nil, "first" => %{}}
    sb = if b["n"] > 0, do: Float.round(Map.get(b["first"], "side_b", 0) / b["n"], 2), else: nil
    "#{h} #{b["n"]} #{b["return_rate"]} #{sb}"
  end))
Output.puts("RESULT expert split-half floor: mismatch #{floor["mismatch_rate"]}  means_js #{floor["means_js"]}")
Output.puts("RESULT expert named rates: side_b_low #{self_score["side_b_low"]}  airdodge_with_jump #{self_score["airdodge_with_jump"]}  nothing_died #{self_score["nothing_died"]}")
Output.puts("RESULT expert timing: side_b fired from low/deep #{self_score["side_b_fired_low"]}  first-means latency median #{self_score["first_means_latency_median"]} f  side_b latency #{self_score["side_b_latency_median"]} f")
Output.puts("RESULT expert edge SDs: carried-off share #{self_score["carried_off_share"]}  died #{self_score["carried_off_died"]}  decided return #{self_score["decided_return_rate"]} (n=#{self_score["decided_n"]})  by move #{inspect(self_score["carried_off_by_move"])}")
Output.success("wrote #{out}")
