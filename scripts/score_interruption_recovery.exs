alias ExPhil.Data.Peppi
alias ExPhil.Eval.MultishineBenchmark, as: Benchmark
{opts, [], []} = OptionParser.parse(System.argv(), strict: [scores: :string, out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
scores = opts |> Keyword.fetch!(:scores) |> File.read!() |> Jason.decode!()

results =
  Enum.map(scores["runs"], fn run ->
    complete = run["truncated"] == nil and run["frames_observed"] == 360

    ended =
      run["truncated"] == "game_ended" and is_integer(run["frames_observed"]) and
        run["frames_observed"] in 1..360

    valid =
      run["timing_valid"] == true and run["error"] == nil and
        (complete or ended) and run["diverged"] == false

    base = %{frame: run["frame"], run: run["run"], valid: valid, directory: run["replay_dir"]}

    if valid do
      paths = Path.wildcard(Path.join(run["replay_dir"], "**/*.slp"))
      [path] = paths
      {:ok, replay} = Peppi.parse(path)

      response =
        Enum.filter(
          replay.frames,
          &(&1.frame_number > run["frame"] + 3 and
              &1.frame_number <= run["frame"] + 360)
        )

      non_neutral =
        Enum.count(response, fn frame ->
          not ExPhil.Eval.ScenarioOpponent.recorded_neutral?(frame.players[2].controller)
        end)

      damage_frames =
        response
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.flat_map(fn [before, after_value] ->
          if after_value.players[1].stock == before.players[1].stock and
               after_value.players[1].percent > before.players[1].percent + 0.01,
             do: [after_value.frame_number],
             else: []
        end)

      rows =
        Benchmark.rows(replay, 1)
        |> Enum.filter(&(&1.frame > run["frame"] and &1.frame <= run["frame"] + 360))

      unless length(rows) == run["frames_observed"], do: raise("response replay length mismatch")
      metrics = Benchmark.score(rows)
      episode = Enum.find(metrics.recovery.episodes, &(&1.cause == :hitstun))

      latency =
        if episode && episode.outcome == :reentered, do: episode.finish - episode.ready, else: nil

      ready_followup =
        if episode && episode.ready,
          do: (episode.finish || episode.last_observed) - episode.ready,
          else: nil

      Map.merge(base, %{
        replay: path,
        metrics: metrics,
        initial_hit_episode: episode,
        game_ended: ended,
        opponent_input_audit: %{
          after_settling_frames: 3,
          frames: length(response),
          non_neutral: non_neutral
        },
        later_damage_increase_frames: damage_frames,
        ready_to_cycle_frames: latency,
        ready_followup_frames: ready_followup,
        observed_full_readiness_deadline: is_number(ready_followup) and ready_followup >= 60,
        resumed_within60_ready_frames: is_number(latency) and latency <= 60,
        replay_sha256: :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
      })
    else
      Map.put(base, :outcome, :invalid_harness_run)
    end
  end)

File.write!(
  out,
  Jason.encode!(
    %{
      policy: scores["policy"],
      response_opponent: scores["response_opponent"] || "replay",
      source_scoreboard: opts[:scores],
      policy_sha256:
        :crypto.hash(:sha256, File.read!(scores["policy"])) |> Base.encode16(case: :lower),
      metric: "existing_multishine_reentry_v1_ready_proxy",
      readiness_deadline_frames: 60,
      response_frames: 360,
      interpretation:
        "Grounded zero-hitstun readiness proxy, not exact actionable time. Rehits/deaths/end censor; invalid runs excluded.",
      valid_runs: Enum.count(results, & &1.valid),
      results: results
    },
    pretty: true
  ),
  [:exclusive]
)
