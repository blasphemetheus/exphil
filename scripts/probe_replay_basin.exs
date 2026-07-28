# Probe v2 (INIT_FORENSICS option 1, real-manifold edition): run a policy's
# buttons head offline over the ACTUAL windows of a live replay and read the
# B logit through the absorbed spell. The synthetic-block probe (v1,
# probe_crouch_boundary.exs) showed dead seeds firing B fine on
# training-style basin windows — so the live failure must live off the
# covered manifold (different entry history, af >> 40). This measures it
# directly, with a parity check: deterministic offline decode must
# reproduce the live bot's actual B presses, else the reconstruction is
# wrong and nothing downstream can be trusted.
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_replay_basin.exs \
#     --policy checkpoints/ms_crouch_g.bin \
#     --replay eval_runs/0727_crouch_g_idle/r1.slp \
#     [--sample-every 30] [--out eval_runs/interp/replay_basin_g_r1.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replay: :string, sample_every: :integer, out: :string]
  )

policy_path = opts[:policy] || raise "--policy required"
replay_path = opts[:replay] || raise "--replay required"
sample_every = opts[:sample_every] || 30
out_path = opts[:out]

Output.banner("Replay-manifold basin probe")
Output.config([{"Policy", policy_path}, {"Replay", replay_path}])

{:ok, replay} = Peppi.parse(replay_path)

frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

n = length(frames)
Output.puts("Frames: #{n}")

ds =
  frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)

loaded = Activations.load_heads(policy_path)
window = loaded.window

# Predict in chunks to bound memory: windows ending at t for t in window-1..n-1
batch = 512

b_logits =
  (window - 1)..(n - 1)
  |> Enum.chunk_every(batch)
  |> Enum.flat_map(fn ts ->
    windows = Enum.map(ts, fn t -> Nx.slice_along_axis(emb, t - window + 1, window, axis: 0) end)
    out = loaded.predict_fn.(loaded.params, Nx.stack(windows))
    elem(out, 0)[[.., 1]] |> Nx.to_flat_list()
  end)

offset = window - 1
squat = ExPhil.Constants.squat()
squat_wait = ExPhil.Constants.squat_wait()

rows =
  b_logits
  |> Enum.with_index(offset)
  |> Enum.map(fn {logit, t} ->
    f = Enum.at(frames, t)
    p = f.game_state.players[1]

    %{
      t: t,
      action: p.action,
      af: p.action_frame,
      in_basin: p.action in [squat, squat_wait],
      b_logit: logit,
      pressed_b: f.controller.button_b
    }
  end)

# ---------------------------------------------------------------------------
# Parity: deterministic decode (logit > 0) vs what the live bot pressed
# ---------------------------------------------------------------------------
parity = Enum.count(rows, fn r -> r.b_logit > 0.0 == r.pressed_b end) / max(length(rows), 1)
Output.puts("B-press parity (offline argmax vs live): #{Float.round(parity * 100, 1)}%")

# ---------------------------------------------------------------------------
# Basin analysis: longest basin spell, logit stats by af band
# ---------------------------------------------------------------------------
basin_rows = Enum.filter(rows, & &1.in_basin)
Output.puts("Basin frames: #{length(basin_rows)}/#{length(rows)}")

bands = [{0, 40}, {41, 120}, {121, 600}, {601, 999_999}]

for {lo, hi} <- bands do
  band = Enum.filter(basin_rows, &(&1.af >= lo and &1.af <= hi))

  if band != [] do
    mean = Enum.sum(Enum.map(band, & &1.b_logit)) / length(band)
    fire = Enum.count(band, &(&1.b_logit > 0.0)) / length(band)
    mx = Enum.max_by(band, & &1.b_logit)

    Output.puts(
      "af #{lo}..#{hi}: n=#{length(band)} mean_logit=#{Float.round(mean, 3)} " <>
        "fire_frac=#{Float.round(fire * 1.0, 4)} max_logit=#{Float.round(mx.b_logit, 3)} (af #{mx.af})"
    )
  end
end

# Spell trajectory sample
spell_sample =
  basin_rows
  |> Enum.take_every(sample_every)
  |> Enum.map(fn r -> %{t: r.t, af: r.af, b_logit: Float.round(r.b_logit, 3)} end)

Output.puts("Spell trajectory (every #{sample_every}f): #{inspect(Enum.take(spell_sample, 12))}")

if out_path do
  File.mkdir_p!(Path.dirname(out_path))

  File.write!(
    out_path,
    Jason.encode!(
      %{
        policy: policy_path,
        replay: replay_path,
        parity: parity,
        rows:
          Enum.map(rows, fn r ->
            %{r | b_logit: Float.round(r.b_logit, 4)}
          end)
      },
      pretty: false
    )
  )

  Output.success("Written: #{out_path}")
end
