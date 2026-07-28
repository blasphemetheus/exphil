# Mental rollout of the crouch basin (INIT_FORENSICS option 1, v3).
#
# The v1 synthetic-block probe predicts live outcome at chance (50%) — seeds
# diverge on their SELF-GENERATED window histories, not on any single
# (state, prev) pair. This script simulates the basin closed loop entirely
# in embedding space: each step embeds a SquatWait frame whose prev-action
# is the policy's own deterministic output from the previous step, appends
# it to the 16-frame window, and asks the policy again. Basin dynamics are
# trivial (crouch persists, af increments and loops at 120) until a B EDGE
# (press after release) occurs — which live is a shine, i.e. escape.
#
# Entry histories:
#   synthetic     - the training-style block (post-shine lead-in -> Squat)
#   <replay.slp>  - the real 16 frames preceding a seed's live absorption
#                   (cross-testable: does policy X escape from policy Y's
#                   entry?)
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_basin_rollout.exs \
#     [--policies "checkpoints/ms_crouch_*.bin"] \
#     [--entries "synthetic,eval_runs/0727_crouch_g_idle/r1.slp:104,eval_runs/0727_crouch_e_idle/r1.slp:196"] \
#     [--max-frames 200]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.ControllerState
alias ExPhil.Data.{Peppi, RecoverySynth}
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, entries: :string, max_frames: :integer]
  )

policy_glob = opts[:policies] || "checkpoints/ms_crouch_*.bin"
max_frames = opts[:max_frames] || 200

entries_spec =
  String.split(
    opts[:entries] ||
      "synthetic," <>
        "eval_runs/0727_crouch_g_idle/r1.slp:104," <>
        "eval_runs/0727_crouch_e_idle/r1.slp:196," <>
        "eval_runs/0727_crouch_h_idle/r1.slp:454",
    ","
  )

policies = Path.wildcard(policy_glob) |> Enum.sort()

Output.banner("Basin mental rollout")
Output.config([{"Policies", length(policies)}, {"Entries", entries_spec}, {"Max frames", max_frames}])

window_size = 16

# ---------------------------------------------------------------------------
# Entry windows: {name, [frame maps]} -> embedded {16, embed} + last controller
# ---------------------------------------------------------------------------
fixture = "test/fixtures/replays/fox_multishine_closed.slp"

embed_frames = fn frames ->
  ds =
    frames
    |> Data.from_frames()
    |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

  Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
end

load_entry = fn spec ->
  case spec do
    "synthetic" ->
      {:ok, replay} = Peppi.parse(fixture)

      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.reject(fn %{controller: c} ->
          c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
        end)

      block = RecoverySynth.build_crouch(frames, port: 1, max_af: 40, lead_in: 16, ratio: 0.001)
      # renumber so prev threads; take through the 2 Squat frames
      base = hd(block).game_state.frame

      pre =
        block
        |> Enum.with_index()
        |> Enum.map(fn {f, i} -> %{f | game_state: %{f.game_state | frame: base + i}} end)
        |> Enum.take(length(block) - 40)

      {"synthetic", pre}

    spec ->
      [path, at] = String.split(spec, ":")
      at = String.to_integer(at)
      {:ok, replay} = Peppi.parse(path)

      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      name = Path.dirname(path) |> Path.basename() |> String.replace("0727_crouch_", "") |> then(&"#{&1}@#{at}")
      {name, Enum.slice(frames, max(at - window_size, 0), window_size)}
  end
end

entries =
  Enum.map(entries_spec, fn spec ->
    {name, frames} = load_entry.(spec)
    emb = embed_frames.(frames)
    {n, _} = Nx.shape(emb)

    # Left-pad short histories by repeating the first frame — the window
    # refills with simulated frames within a few rollout steps.
    emb =
      if n < window_size do
        pad = Nx.tile(Nx.slice_along_axis(emb, 0, 1, axis: 0), [window_size - n, 1])
        Nx.concatenate([pad, emb], axis: 0)
      else
        emb
      end

    {n, _} = Nx.shape(emb)
    last16 = Nx.slice_along_axis(emb, n - window_size, window_size, axis: 0)
    template = List.last(frames)
    {name, last16, template}
  end)

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
squat_wait = ExPhil.Constants.squat_wait()

decode_controller = fn out ->
  buttons = elem(out, 0) |> Nx.squeeze() |> Nx.to_flat_list()
  [a, b, x, y, z, l, r, _d_up] = Enum.map(buttons, &(&1 > 0.0))
  bucket = fn t -> (t |> Nx.squeeze() |> Nx.argmax() |> Nx.to_number()) / 16.0 end

  %ControllerState{
    main_stick: %{x: bucket.(elem(out, 1)), y: bucket.(elem(out, 2))},
    c_stick: %{x: bucket.(elem(out, 3)), y: bucket.(elem(out, 4))},
    l_shoulder: 0.0,
    r_shoulder: 0.0,
    button_a: a,
    button_b: b,
    button_x: x,
    button_y: y,
    button_z: z,
    button_l: l,
    button_r: r
  }
end

rollout = fn loaded, entry_window, template ->
  player0 = template.game_state.players[1]
  base_frame = template.game_state.frame

  Enum.reduce_while(1..max_frames, {entry_window, template.controller, nil}, fn t, {win, prev_ctrl, _} ->
    af = rem(t - 1, 120) + 1
    player = %{player0 | action: squat_wait, action_frame: af, on_ground: true}

    gs = %{
      template.game_state
      | frame: base_frame + t,
        players: Map.put(template.game_state.players, 1, player)
    }

    emb =
      ExPhil.Embeddings.Game.embed(gs, prev_ctrl, 1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.reshape({1, :auto})

    win = Nx.concatenate([Nx.slice_along_axis(win, 1, window_size - 1, axis: 0), emb], axis: 0)
    out = loaded.predict_fn.(loaded.params, Nx.new_axis(win, 0))
    ctrl = decode_controller.(out)

    edge = ctrl.button_b and not prev_ctrl.button_b

    if edge do
      {:halt, {win, ctrl, {:escape, t}}}
    else
      {:cont, {win, ctrl, {:absorbed, t}}}
    end
  end)
  |> elem(2)
end

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
header =
  String.pad_trailing("seed", 13) <>
    Enum.map_join(entries, "", fn {name, _, _} -> String.pad_trailing(name, 16) end)

IO.puts(header)

for path <- policies do
  seed = Path.basename(path, ".bin")
  loaded = Activations.load_heads(path)

  cells =
    Enum.map_join(entries, "", fn {_name, win, template} ->
      case rollout.(loaded, win, template) do
        {:escape, t} -> String.pad_trailing("esc@#{t}", 16)
        {:absorbed, _} -> String.pad_trailing("ABSORBED", 16)
      end
    end)

  IO.puts(String.pad_trailing(seed, 13) <> cells)
end
