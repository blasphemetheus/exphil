# Read-only replay audit; safe while a separate evaluation is running.
# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/local_multishine_report.exs MATRIX_DIR
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain

defmodule RespawnAudit do
  def score(rows) do
    rows
    |> Enum.chunk_by(& &1.stock)
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [before, after_death] ->
      death = hd(after_death)
      ready_rows = Enum.drop_while(after_death, fn row ->
        not (row.grounded and row.hitstun == 0 and
          (row.action in 14..23 or ExPhil.Eval.ShineChain.family(row.action) == :ground_reflect))
      end)
      finish = Enum.reduce_while(ready_rows, [], fn row, prefix ->
        prefix = [row | prefix]
        if ExPhil.Eval.MultishineBenchmark.score(Enum.reverse(prefix)).completed_cycles > 0,
          do: {:halt, row.frame}, else: {:cont, prefix}
      end)
      %{death_frame: death.frame, stocks_before: hd(before).stock, stocks_after: death.stock,
        ready_frame: if(ready_rows != [], do: hd(ready_rows).frame),
        cycle_frame: if(is_integer(finish), do: finish),
        ready_to_cycle_frames: if(is_integer(finish), do: finish - hd(ready_rows).frame),
        outcome: cond do
          is_integer(finish) -> :resumed
          ready_rows == [] -> :no_grounded_readiness_observed
          true -> :no_cycle_before_stock_or_window_end
        end}
    end)
  end
end

[directory] = System.argv()

runs =
  Path.wildcard(Path.join(directory, "*/result.json"))
  |> Enum.map(fn path ->
    result = path |> File.read!() |> JSON.decode!()

    if result["valid"] do
      hash = :crypto.hash(:sha256, File.read!(result["replay"])) |> Base.encode16(case: :lower)
      true = hash == result["replay_sha256"]
      {:ok, replay} = Peppi.parse(result["replay"])

      rows =
        ExPhil.Eval.MultishineBenchmark.rows(replay, 1)
        |> Enum.filter(&(&1.frame <= result["session"]["last_frame"]))

      metrics = ExPhil.Eval.MultishineBenchmark.score(rows) |> JSON.encode!() |> JSON.decode!()
      recovery = metrics["recovery"]
      positions = Enum.map(rows, & &1.player.x)
      # Peppi's ControllerFrame uses [0, 1], with neutral at 0.5.
      horizontal = Enum.count(rows, &(abs(&1.player.controller.main_stick_x - 0.5) > 0.01))
      looping = Enum.count(rows, &(ShineChain.family(&1.action) != :other and &1.hitstun == 0))
      hit_episodes = Enum.filter(recovery["episodes"], &(&1["cause"] == "hitstun"))

      Map.merge(
        Map.take(result, ["id", "stage", "opponent", "valid", "replay", "replay_sha256"]),
        %{
          "metrics_semantics" => "slippi_state_flag_hitstun_v2",
          "frozen_stadium" => replay.metadata.frozen_stadium,
          "source_result" => path,
          "max_chain" => metrics["max_chain"],
          "completed_cycles" => metrics["completed_cycles"],
          "deaths" => metrics["deaths"],
          "respawns" => RespawnAudit.score(rows),
          "stationary_input_frames" => length(rows) - horizontal,
          "horizontal_input_frames" => horizontal,
          "x_range" => [Enum.min(positions), Enum.max(positions)],
          "loop_fraction" => looping / length(rows),
          "recovery_completed" => recovery["completed"],
          "recovery_censored" => recovery["censored"],
          "hit_episodes" => length(hit_episodes),
          "hit_reentries" => Enum.count(hit_episodes, &(&1["outcome"] == "reentered")),
          "censor_reasons" =>
            recovery["episodes"]
            |> Enum.reject(&(&1["outcome"] == "reentered"))
            |> Enum.frequencies_by(& &1["outcome"]),
          "worst_ready_to_cycle_frames" =>
            Enum.max(recovery["ready_to_cycle_frames"], fn -> nil end),
          "standing_gate" => metrics["max_chain"] >= 30 and metrics["deaths"] == 0
        }
      )
    else
      Map.take(result, ["id", "stage", "opponent", "valid", "error"])
    end
  end)

IO.puts(Jason.encode!(%{runs: runs}, pretty: true))
