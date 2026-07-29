# Drift-vs-jitter discriminator for offset-calibration flatness (task:
# headless chains gap, cal 0.67 vs windowed 0.78).
#
# probe_cycle_margins.exs reports ONE cal table per replay — a whole-game
# aggregate. A flat/low peak has two very different causes with different
# fixes:
#   DRIFT      - the input-application offset moves over the game (thermal
#                ramp, accumulating clock skew): each game THIRD is sharp,
#                but peaked at different offsets / decaying concentration.
#   STATIONARY - a per-frame race (Null-gfx frame boundary, enet dispatch):
#                every third equally flat.
#
# Usage:
#   mix run scripts/probe_cal_drift.exs --policy checkpoints/ms_open_z.bin \
#     --replays "eval_runs/stand_headless/replays/**/*.slp,eval_runs/stand_windowed/r*.slp"
#
# Same transition-parity calibration as probe_cycle_margins (B/X button
# parity at candidate offsets), computed per game third.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [policy: :string, replays: :string])

policy_path = opts[:policy] || raise "--policy required"

replays =
  (opts[:replays] || raise("--replays required"))
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.sort()

if replays == [], do: raise("no replays matched")

Output.banner("Calibration drift probe")
Output.config([{"Policy", policy_path}, {"Replays", length(replays)}])

loaded = Activations.load_heads(policy_path)
window = loaded.window
offsets = [0, -1, -2, -3, -4]

for replay <- replays do
  case Peppi.parse(replay) do
    {:ok, parsed} ->
      frames =
        parsed
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      ds =
        frames
        |> Data.from_frames()
        |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

      emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
      {total, _} = Nx.shape(emb)

      logits =
        (window - 1)..(total - 1)
        |> Enum.chunk_every(512)
        |> Enum.flat_map(fn ts ->
          wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
          out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
          buttons = elem(out, 0)
          b = buttons[[.., 1]] |> Nx.to_flat_list()
          x = buttons[[.., 2]] |> Nx.to_flat_list()
          Enum.zip(b, x)
        end)

      frames_arr = List.to_tuple(frames)
      logits_arr = List.to_tuple(logits)
      logit_at = fn t -> if t >= window - 1 and t < total, do: elem(logits_arr, t - (window - 1)) end

      transitions =
        for t <- window..(total - 1),
            f = elem(frames_arr, t),
            p = elem(frames_arr, t - 1),
            f.controller.button_b != p.controller.button_b or
              f.controller.button_x != p.controller.button_x,
            do: t

      cal_for = fn ts ->
        Map.new(offsets, fn off ->
          hits =
            Enum.count(ts, fn t ->
              case logit_at.(t + off) do
                nil ->
                  false

                {b, x} ->
                  f = elem(frames_arr, t)
                  (b > 0.0) == f.controller.button_b and (x > 0.0) == f.controller.button_x
              end
            end)

          {off, Float.round(hits / max(length(ts), 1), 3)}
        end)
      end

      third = max(div(total, 3), 1)

      slices =
        [
          {"1st", Enum.filter(transitions, &(&1 < third))},
          {"2nd", Enum.filter(transitions, &(&1 >= third and &1 < 2 * third))},
          {"3rd", Enum.filter(transitions, &(&1 >= 2 * third))}
        ]

      Output.puts("#{Path.basename(replay)} (#{length(transitions)} transitions):")

      for {label, ts} <- slices do
        cal = cal_for.(ts)
        {peak_off, peak} = Enum.max_by(cal, fn {_o, v} -> v end)

        Output.puts(
          "  #{label} third (n=#{length(ts)}): peak #{peak} @ #{peak_off}  " <>
            "cal=#{inspect(Enum.map(offsets, &{&1, cal[&1]}))}"
        )
      end

    _ ->
      Output.warning("#{replay}: parse failed, skipping")
  end
end

Output.puts("")
Output.puts("Read: thirds sharp but peaks move/decay = DRIFT (clock/thermal); " <>
  "all thirds equally flat = STATIONARY race (Null-gfx/enet boundary).")
