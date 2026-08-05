# Opponent-dependence score (INTERP_ROADMAP_V2 W2): does a checkpoint's
# cycle CONSULT opponent state at all?
#
# Motivation: ms_g6_sp1 set the stand-dummy record (434/min) and scored
# ZERO shines against a human; ms_g4_d2mix (424/min) is the best human
# performer (40 shines, chain 2). If static overfit = opponent-input
# blindness, an offline sensitivity probe separates them for GPU-seconds
# instead of a human session.
#
# Method: embed a COMMON replay's frames (default: g4's deterministic FD
# stand run — controlled comparison, same states for every policy) three
# ways — baseline, opponent teleported far (+120 x), opponent neutralized
# (Wait af0, 0%, no speeds) — and measure per-frame B/X head logit deltas.
# Consulting policies move; blind policies don't.
#
# Validation contrast (pre-registered): g6_sp1 (human-zero) should score
# LOWEST dependence, g4_d2mix (human-best) higher; g2_mdq_ss third point.
# A wrong ranking REFUTES the blindness hypothesis — also a result.
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_opponent_dependence.exs \
#     [--policies "checkpoints/ms_g6_sp1.bin,checkpoints/ms_g4_d2mix.bin,checkpoints/ms_g2_mdq_ss.bin"] \
#     [--replay eval_runs/0804_stage_final_destination/r1.slp] \
#     [--delay-id 3] [--stride 2] [--out eval_runs/interp/opp_dependence.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policies: :string,
      replay: :string,
      delay_id: :integer,
      stride: :integer,
      out: :string,
      fine: :boolean
    ]
  )

policies =
  (opts[:policies] ||
     "checkpoints/ms_g6_sp1.bin,checkpoints/ms_g4_d2mix.bin,checkpoints/ms_g2_mdq_ss.bin")
  |> String.split(",", trim: true)

replay_path = opts[:replay] || "eval_runs/0804_stage_final_destination/r1.slp"
delay_id = opts[:delay_id] || 3
stride = opts[:stride] || 2
out_path = opts[:out]

Output.banner("Opponent-dependence probe (W2)")
Output.config([{"Replay", replay_path}, {"Delay id", delay_id}, {"Stride", stride}])

{:ok, replay} = Peppi.parse(replay_path)

frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

perturb = fn frames, fun ->
  Enum.map(frames, fn f ->
    opp = f.game_state.players[2]
    %{f | game_state: %{f.game_state | players: Map.put(f.game_state.players, 2, fun.(opp))}}
  end)
end

variants =
  [
    {:baseline, frames},
    {:far, perturb.(frames, fn o -> %{o | x: o.x + 120.0} end)},
    {:neutral,
     perturb.(frames, fn o ->
       %{
         o
         | action: 14,
           action_frame: 0,
           percent: 0.0,
           speed_air_x_self: 0.0,
           speed_ground_x_self: 0.0,
           speed_y_self: 0.0,
           speed_x_attack: 0.0,
           speed_y_attack: 0.0
       }
     end)}
  ] ++
    if opts[:fine] do
      # Single-channel attribution: which opponent variable carries the
      # sensitivity the coarse probe measured?
      [
        {:x_only, perturb.(frames, fn o -> %{o | x: o.x + 120.0} end)},
        {:x_small, perturb.(frames, fn o -> %{o | x: o.x + 20.0} end)},
        {:y_only, perturb.(frames, fn o -> %{o | y: o.y + 30.0} end)},
        {:action_only, perturb.(frames, fn o -> %{o | action: 20, action_frame: 5} end)},
        {:percent_only, perturb.(frames, fn o -> %{o | percent: 80.0} end)},
        {:facing_only, perturb.(frames, fn o -> %{o | facing: -o.facing} end)}
      ]
    else
      []
    end

results =
  for path <- policies do
    seed = Path.basename(path, ".bin")
    loaded = Activations.load_heads(path)
    window = loaded.window
    n = length(frames)

    logits_for = fn fs ->
      ds = Activations.embed_frames(fs, loaded.config, delay_id: delay_id)
      emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)

      (window - 1)..(n - 1)//stride
      |> Enum.chunk_every(512)
      |> Enum.flat_map(fn ts ->
        wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
        out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
        buttons = elem(out, 0)
        Enum.zip(Nx.to_flat_list(buttons[[.., 1]]), Nx.to_flat_list(buttons[[.., 2]]))
      end)
    end

    [base | rest] = Enum.map(variants, fn {name, fs} -> {name, logits_for.(fs)} end)
    {_, base_logits} = base

    deltas =
      for {name, var_logits} <- rest do
        pairs = Enum.zip(base_logits, var_logits)
        m = length(pairs)

        db = Enum.sum(Enum.map(pairs, fn {{b0, _}, {b1, _}} -> abs(b1 - b0) end)) / m
        dx = Enum.sum(Enum.map(pairs, fn {{_, x0}, {_, x1}} -> abs(x1 - x0) end)) / m

        flips =
          Enum.count(pairs, fn {{b0, x0}, {b1, x1}} ->
            b0 > 0 != b1 > 0 or x0 > 0 != x1 > 0
          end) / m

        Output.puts(
          "#{String.pad_trailing(seed, 16)} #{String.pad_trailing(to_string(name), 8)} " <>
            "dB=#{Float.round(db, 4)} dX=#{Float.round(dx, 4)} flip=#{Float.round(flips, 4)}"
        )

        %{variant: name, d_b: db, d_x: dx, flip_frac: flips}
      end

    score = deltas |> Enum.map(&(&1.d_b + &1.d_x)) |> Enum.sum() |> Kernel./(length(deltas))
    Output.puts("#{String.pad_trailing(seed, 16)} DEPENDENCE=#{Float.round(score, 4)}")
    %{policy: seed, deltas: deltas, dependence: score}
  end

Output.puts("")
Output.puts("Ranking (higher = consults opponent more):")

results
|> Enum.sort_by(&(-&1.dependence))
|> Enum.each(fn r -> Output.puts("  #{String.pad_trailing(r.policy, 16)} #{Float.round(r.dependence, 4)}") end)

if out_path do
  File.mkdir_p!(Path.dirname(out_path))
  File.write!(out_path, Jason.encode!(results))
  Output.success("Wrote #{out_path}")
end
