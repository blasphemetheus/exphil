alias ExPhil.Data.Peppi
alias ExPhil.Interp.{ActionNames, ReplayStats}
[directory] = System.argv()

reports =
  Path.wildcard(Path.join(directory, "*/session.json"))
  |> Enum.map(fn path ->
    session = File.read!(path) |> Jason.decode!()
    files = Path.wildcard(Path.join(Path.dirname(path), "**/*.slp"))

    unless session["status"] == "ok" and session["errors"] == 0 and
             session["measured_latency"] == 1 and session["game_ended"] and length(files) == 1,
           do: raise("invalid live session: #{path}")

    [file] = files
    {:ok, replay} = Peppi.parse(file)

    rows =
      ExPhil.Eval.MultishineBenchmark.rows(replay, session["player_port"])
      |> Enum.filter(&(&1.frame <= session["last_frame"]))

    unless length(rows) == session["last_frame"] + 1,
      do: raise("live replay frame coverage mismatch")

    actions = Enum.map(rows, & &1.action)
    pairs = Enum.chunk_every(rows, 2, 1, :discard)

    damage =
      Enum.reduce(pairs, 0.0, fn [a, b], total ->
        if a.stock == b.stock,
          do: total + max(b.player.percent - a.player.percent, 0),
          else: total
      end)

    horizontal = Enum.count(rows, &(abs(&1.player.controller.main_stick_x - 0.5) > 0.05))
    idle_runs = ReplayStats.run_lengths(actions, [14])

    %{
      session: session,
      replay: file,
      replay_sha256:
        :crypto.hash(:sha256, File.read!(file))
        |> Base.encode16(case: :lower),
      scored_frames: length(rows),
      forced_finish_tail_excluded: true,
      shield: ReplayStats.shield_stats(actions),
      idle_fraction: Enum.count(actions, &(&1 == 14)) / length(actions),
      longest_idle_frames: Enum.max(idle_runs, fn -> 0 end),
      horizontal_input_fraction: horizontal / length(rows),
      damage_received: damage,
      stocks_lost: Enum.sum(Enum.map(pairs, fn [a, b] -> max(a.stock - b.stock, 0) end)),
      top_actions:
        ReplayStats.action_histogram(actions, 12)
        |> Enum.map(fn {id, count} -> %{id: id, name: ActionNames.name(id), frames: count} end)
    }
  end)

report = %{protocol: "fox_v3_sampled_live_v1", sessions: reports}
File.write!(Path.join(directory, "report.json"), Jason.encode!(report, pretty: true))
IO.puts(Jason.encode!(report, pretty: true))
