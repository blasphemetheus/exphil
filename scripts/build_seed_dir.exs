# Build a seed-rollout dir from scenario_suite scoreboards (drills design
# 2026-07-23, feature A; extends #38).
#
# scenario_suite --finalize --runs N --temperature T writes one finalized
# .slp per attempt (in per-run replay dirs) plus a scoreboard JSON. This
# collects the usable attempts into a flat seed dir and writes
# seed_meta.json so dagger_drill can slice each seed to its
# [handoff, handoff+window] response — without that, N attempts from one
# handoff would inject N identical prefix copies + N SD tails into the
# training pool.
#
#   mix run scripts/build_seed_dir.exs \
#     --scoreboard logs/scenario_scores_20260723_*.json \
#     --out seeds/r17_conversion
#
#   NEWERA8_SEED_ROLLOUTS_DIR=seeds/r17_conversion scripts/overnight_newera8.sh
#
# Options:
#   --scoreboard GLOB[,GLOB]  Scoreboard JSON(s) (required)
#   --out DIR                 Seed dir (required; merges into existing)
#   --min-score F             Keep attempts with score >= F
#   --max-score F             Keep attempts with score <= F (relabel-
#                             correction curricula want the FAILURES)
#   --allow-unfinalized       Keep truncated != finalized runs (their .slp
#                             usually fails Peppi — GOTCHA #73)
#   --quiet
#
# Pure file ops — no Dolphin, no GPU. UNTESTED as of 2026-07-23; test
# plan in docs/planning/DRILLS_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      scoreboard: :string,
      out: :string,
      min_score: :float,
      max_score: :float,
      allow_unfinalized: :boolean,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:scoreboard] && opts[:out] do
  Output.error("--scoreboard and --out are required")
  System.halt(1)
end

scoreboard_paths =
  opts[:scoreboard]
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.uniq()

if scoreboard_paths == [] do
  Output.error("No scoreboards match #{opts[:scoreboard]}")
  System.halt(1)
end

out_dir = Path.expand(opts[:out])
File.mkdir_p!(out_dir)
meta_path = Path.join(out_dir, "seed_meta.json")

existing_meta =
  case File.read(meta_path) do
    {:ok, bin} -> Jason.decode!(bin)
    _ -> %{}
  end

Output.banner("Seed Dir Builder")

Output.config([
  {"Scoreboards", length(scoreboard_paths)},
  {"Out", out_dir},
  {"Existing seeds", map_size(existing_meta)},
  {"Score filter", "#{inspect(opts[:min_score])}..#{inspect(opts[:max_score])}"}
])

in_score_range = fn score ->
  (opts[:min_score] == nil or score >= opts[:min_score]) and
    (opts[:max_score] == nil or score <= opts[:max_score])
end

{meta, stats} =
  Enum.reduce(scoreboard_paths, {existing_meta, %{kept: 0, skipped: %{}}}, fn sb_path,
                                                                              {meta, stats} ->
    sb = sb_path |> File.read!() |> Jason.decode!()
    sb_ts = sb["timestamp"] || Path.basename(sb_path, ".json")

    sb["runs"]
    |> Enum.with_index()
    |> Enum.reduce({meta, stats}, fn {run, seq}, {meta, stats} ->
      skip = fn reason ->
        {meta, %{stats | skipped: Map.update(stats.skipped, reason, 1, &(&1 + 1))}}
      end

      cond do
        run["error"] ->
          skip.(:error)

        run["diverged"] ->
          skip.(:diverged)

        # "finalized" = the --finalize SD reached GAME!; "game_ended" = the
        # game ended naturally during the window (kill/death) — both leave
        # a finalized, parseable .slp.
        run["truncated"] not in ["finalized", "game_ended"] and
            not (opts[:allow_unfinalized] || false) ->
          skip.(:unfinalized)

        not in_score_range.(run["score"] || 0.0) ->
          skip.(:score_filter)

        true ->
          case Path.wildcard(Path.join(run["replay_dir"] || "", "*.slp")) do
            [slp] ->
              name = "#{run["type"]}_#{sb_ts}_s#{seq}_r#{run["run"]}.slp"
              File.cp!(slp, Path.join(out_dir, name))

              entry = %{
                "handoff" => run["frame"],
                "window" => run["window"],
                "type" => run["type"],
                "score" => run["score"],
                "pass" => run["pass"],
                "source_slp" => run["slp"],
                "source_scoreboard" => sb_path
              }

              {Map.put(meta, name, entry), %{stats | kept: stats.kept + 1}}

            [] ->
              skip.(:no_slp_in_replay_dir)

            _many ->
              skip.(:multiple_slp_in_replay_dir)
          end
      end
    end)
  end)

File.write!(meta_path, Jason.encode!(meta, pretty: true))

Output.puts("")
Output.puts("kept:    #{stats.kept} new seed(s), #{map_size(meta)} total in dir")
Output.puts("skipped: #{inspect(stats.skipped)}")
Output.success("seed_meta.json -> #{meta_path}")

if stats.kept == 0 do
  Output.warning(
    "No seeds kept — check --finalize was on (truncated must be \"finalized\") " <>
      "and the score filter."
  )
end
