# Scenario-anchored DAgger harvest (bridge 2 of the bot's Improoover
# loop, 2026-08-12): turn a scenario_suite.exs practice session into
# training data.
#
# The suite already ran N sampled attempts per bookmarked moment and
# scored each response window. This script reads the scoreboard, keeps
# the SUCCESSFUL attempts (pass=true, or --min-score), cuts the
# response window [handoff, handoff+window] from each attempt's replay,
# and emits a MixFrames envelope of the policy's OWN successful
# responses — reward-filtered self-imitation, ready for
# build_snippet_corpus + --mix-corpus stacking.
#
#   mix run scripts/scenario_dagger_mine.exs \
#     --scores logs/scenario_scores_<ts>.json \
#     --out eval_runs/<name>_scenario_snippets \
#     [--min-score 0.5]   # otherwise pass=true gates
#     [--port 1]          # the policy's port in suite runs
#
# Semantics note: keeping recorded controllers is valid here because the
# recorder IS the student succeeding — the same rule that lets human
# conversions train without relabeling, applied to the policy's own
# wins. Failed attempts are dropped, not relabeled (no expert exists
# for arbitrary moments; when one does, relabel instead).

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [scores: :string, out: :string, min_score: :float, port: :integer, pre: :integer]
  )

scores_path = opts[:scores] || raise("--scores is required")
out_dir = opts[:out] || raise("--out is required")
min_score = opts[:min_score]
port = opts[:port] || 1
# A short pre-handoff lead-in gives the temporal trunk real history at
# the window start (the prefix frames are the ORIGINAL game's — valid
# context, not policy output)
pre = opts[:pre] || 45

board = scores_path |> File.read!() |> Jason.decode!()
runs = board["runs"] || []

keep? = fn run ->
  cond do
    min_score != nil -> is_number(run["score"]) and run["score"] >= min_score
    true -> run["pass"] == true
  end
end

kept = Enum.filter(runs, fn r -> keep?.(r) and r["diverged"] != true end)

File.mkdir_p!(out_dir)
Output.banner("Scenario DAgger harvest")
Output.config([
  {"Scoreboard", scores_path},
  {"Runs", "#{length(runs)} total, #{length(kept)} successful kept"},
  {"Gate", if(min_score, do: "score >= #{min_score}", else: "pass == true")}
])

snippets =
  Enum.flat_map(kept, fn run ->
    replay =
      run["replay_dir"]
      |> Path.join("**/*.slp")
      |> Path.wildcard()
      |> List.first()

    with true <- replay != nil,
         {:ok, parsed} <- Peppi.parse(replay, player_port: port) do
      frames =
        parsed
        |> Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
        |> Enum.reject(&(&1.game_state.frame < 0))

      handoff = run["frame"]
      window = run["window"] || 300

      cut =
        frames
        |> Enum.filter(fn f ->
          f.game_state.frame >= handoff - pre and f.game_state.frame <= handoff + window
        end)

      if length(cut) >= 60, do: [cut], else: []
    else
      _ ->
        Output.warning("no replay for run #{run["run"]} (#{run["type"]}) — skipped")
        []
    end
  end)

total = snippets |> Enum.map(&length/1) |> Enum.sum()

File.write!(
  Path.join(out_dir, "snippets.frames"),
  :erlang.term_to_binary(
    %{
      expert: "scenario_self_success",
      exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      action_delay: 0,
      frame_lists: snippets
    },
    [:compressed]
  )
)

Output.success(
  "#{length(snippets)} successful response windows / #{total} frames -> #{out_dir}/snippets.frames"
)

Output.puts("  next: build_snippet_corpus (stack with the standing mixes) -> retrain")
