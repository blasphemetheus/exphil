# Relapse dashboard (INTERP_ROADMAP_V2 W3, decode-level v1): known bad
# habits, scored per checkpoint across the lineage on COMMON probe states.
#
# Design note: the persona-vector recipe (project activations onto a
# bad-habit direction) does NOT transfer across checkpoints — directions
# live in each trunk's own basis (the #36 lesson: fresh rounds can't reuse
# r14's vector). v1 therefore reads the HEADS, not directions: run each
# policy over the same states and measure what it would emit. Emission
# tendencies compare cleanly across checkpoints.
#
# Habits tracked:
#   shield  — fraction of common-FD frames with max(sig(L),sig(R)) >= 0.45
#             (shield-lock relapse; healthy multishiners ~0)
#   platX   — X fire-frac + mean X logit on platform JC states from the
#             absorbed-YS fixture (the absorber habit; silent = absorbed)
#   platB   — mean B logit on the same states (the down+B motif riding on)
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/relapse_dashboard.exs \
#     [--policies "checkpoints/ms_g2_mdq_ss.bin,checkpoints/ms_g4_d2mix.bin,..."] \
#     [--delay-id 3] [--out eval_runs/interp/relapse_dashboard.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, delay_id: :integer, out: :string]
  )

default_lineage =
  "checkpoints/ms_g2_mdq_ss.bin,checkpoints/ms_g4_d2mix.bin,checkpoints/ms_g6_sp1.bin," <>
    "checkpoints/ms_g7_pressure.bin,checkpoints/ms_g8_snippets.bin,checkpoints/ms_g10b_human.bin"

policies =
  (opts[:policies] || default_lineage)
  |> String.split(",", trim: true)
  |> Enum.filter(fn p -> File.exists?(p) or (Output.warning("missing #{p}"); false) end)

delay_id = opts[:delay_id] || 3
out_path = opts[:out] || "eval_runs/interp/relapse_dashboard.json"

fd_replay = "eval_runs/0804_stage_final_destination/r1.slp"
ys_fixture = "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"
reflector = [360, 361, 362]

Output.banner("Relapse dashboard (decode-level v1)")

load_frames = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
end

fd_frames = load_frames.(fd_replay)
ys_frames = load_frames.(ys_fixture)

plat_jc_idx =
  ys_frames
  |> Enum.with_index()
  |> Enum.filter(fn {f, t} ->
    p = f.game_state.players[1]
    t >= 60 and p.action in reflector and p.action_frame >= 3 and p.y > 15
  end)
  |> Enum.map(&elem(&1, 1))

Output.puts("Common states: FD #{length(fd_frames)} frames, plat-JC #{length(plat_jc_idx)} windows")
Output.puts("")

Output.puts(
  String.pad_trailing("policy", 18) <>
    " shield%   platX_fire  platX_mean  platB_mean"
)

sig = fn l -> 1.0 / (1.0 + :math.exp(-l)) end

rows =
  for path <- policies do
    seed = Path.basename(path, ".bin")
    loaded = Activations.load_heads(path)
    window = loaded.window

    logits_over = fn frames, ts ->
      ds = Activations.embed_frames(frames, loaded.config, delay_id: delay_id)
      emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)

      ts
      |> Enum.chunk_every(256)
      |> Enum.flat_map(fn chunk ->
        wins = Enum.map(chunk, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
        out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
        buttons = elem(out, 0)
        # buttons order: [a, b, x, y, z, l, r, d_up]
        Enum.zip([
          Nx.to_flat_list(buttons[[.., 1]]),
          Nx.to_flat_list(buttons[[.., 2]]),
          Nx.to_flat_list(buttons[[.., 5]]),
          Nx.to_flat_list(buttons[[.., 6]])
        ])
      end)
    end

    fd_ts = Enum.to_list((window - 1)..(length(fd_frames) - 1)//3)
    fd_logits = logits_over.(fd_frames, fd_ts)

    shield_frac =
      Enum.count(fd_logits, fn {_b, _x, l, r} -> max(sig.(l), sig.(r)) >= 0.45 end) /
        max(length(fd_logits), 1)

    plat_logits = logits_over.(ys_frames, plat_jc_idx)
    n_plat = max(length(plat_logits), 1)
    x_fire = Enum.count(plat_logits, fn {_b, x, _l, _r} -> x > 0.0 end) / n_plat
    x_mean = Enum.sum(Enum.map(plat_logits, fn {_b, x, _l, _r} -> x end)) / n_plat
    b_mean = Enum.sum(Enum.map(plat_logits, fn {b, _x, _l, _r} -> b end)) / n_plat

    Output.puts(
      String.pad_trailing(seed, 18) <>
        " #{String.pad_trailing(Float.round(shield_frac * 100, 2) |> to_string(), 9)}" <>
        " #{String.pad_trailing(Float.round(x_fire, 4) |> to_string(), 11)}" <>
        " #{String.pad_trailing(Float.round(x_mean, 3) |> to_string(), 11)}" <>
        " #{Float.round(b_mean, 3)}"
    )

    %{policy: seed, shield_frac: shield_frac, plat_x_fire: x_fire, plat_x_mean: x_mean, plat_b_mean: b_mean}
  end

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(rows))
Output.success("Wrote #{out_path}")
