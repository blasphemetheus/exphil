# Mode connectivity between crouch-recipe seeds (INIT_FORENSICS option 5).
#
# Linearly interpolate exported policy weights theta(alpha) = (1-a)*A + a*B
# and score every point with the basin mental rollout (option 1 v3): frames
# to escape from (i) the synthetic training-style entry and (ii) seed g's
# real absorbed entry — the test that separates universal escapers (a, c, i)
# from everyone else. Question: is the escape solution connected to the
# absorbed solution through a barrier (separate basins of the LOSS
# landscape), or do they blend smoothly?
#
# Known caveat, stated in advance: naive linear paths between independently
# trained nets generically cross a high-loss barrier from permutation
# mismatch alone. The information is in the CONTRAST between pairs
# (escaper<->failure vs escaper<->escaper), not in any single curve.
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/interp_mode_connectivity.exs \
#     [--pair-a checkpoints/ms_crouch_a.bin --pair-b checkpoints/ms_crouch_g.bin] \
#     [--points 11] [--max-frames 200]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.ControllerState
alias ExPhil.Data.{Peppi, RecoverySynth}
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [pair_a: :string, pair_b: :string, points: :integer, max_frames: :integer]
  )

path_a = opts[:pair_a] || "checkpoints/ms_crouch_a.bin"
path_b = opts[:pair_b] || "checkpoints/ms_crouch_g.bin"
points = opts[:points] || 11
max_frames = opts[:max_frames] || 200
window_size = 16

Output.banner("Mode connectivity: #{Path.basename(path_a)} <-> #{Path.basename(path_b)}")

# ---------------------------------------------------------------------------
# Entries (same construction as probe_basin_rollout.exs)
# ---------------------------------------------------------------------------
fixture = "test/fixtures/replays/fox_multishine_closed.slp"

embed_frames = fn frames ->
  ds =
    frames
    |> Data.from_frames()
    |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

  Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
end

pad16 = fn emb ->
  {n, _} = Nx.shape(emb)

  emb =
    if n < window_size do
      pad = Nx.tile(Nx.slice_along_axis(emb, 0, 1, axis: 0), [window_size - n, 1])
      Nx.concatenate([pad, emb], axis: 0)
    else
      emb
    end

  {n2, _} = Nx.shape(emb)
  Nx.slice_along_axis(emb, n2 - window_size, window_size, axis: 0)
end

{:ok, replay} = Peppi.parse(fixture)

fixture_frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)

block = RecoverySynth.build_crouch(fixture_frames, port: 1, max_af: 40, lead_in: 16, ratio: 0.001)
base = hd(block).game_state.frame

synth_pre =
  block
  |> Enum.with_index()
  |> Enum.map(fn {f, i} -> %{f | game_state: %{f.game_state | frame: base + i}} end)
  |> Enum.take(length(block) - 40)

{:ok, g_replay} = Peppi.parse("eval_runs/0727_crouch_g_idle/r1.slp")

g_frames =
  g_replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

entries = [
  {"synthetic", pad16.(embed_frames.(synth_pre)), List.last(synth_pre)},
  {"g@104", pad16.(embed_frames.(Enum.slice(g_frames, 104 - window_size, window_size))),
   Enum.at(g_frames, 103)}
]

# ---------------------------------------------------------------------------
# Rollout (same dynamics as probe_basin_rollout.exs)
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

rollout = fn predict_fn, params, entry_window, template ->
  player0 = template.game_state.players[1]
  base_frame = template.game_state.frame

  Enum.reduce_while(1..max_frames, {entry_window, template.controller, {:absorbed, max_frames}}, fn t,
                                                                                                    {win,
                                                                                                     prev_ctrl,
                                                                                                     _} ->
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
    out = predict_fn.(params, Nx.new_axis(win, 0))
    ctrl = decode_controller.(out)
    edge = ctrl.button_b and not prev_ctrl.button_b

    if edge, do: {:halt, {win, ctrl, {:escape, t}}}, else: {:cont, {win, ctrl, {:absorbed, t}}}
  end)
  |> elem(2)
end

# ---------------------------------------------------------------------------
# Interpolate ModelState leaves
# ---------------------------------------------------------------------------
la = Activations.load_heads(path_a)
lb = Activations.load_heads(path_b)

lerp_tree = fn lerp_tree, a, b, alpha ->
  cond do
    is_struct(a, Nx.Tensor) ->
      case Nx.type(a) do
        {t, _} when t in [:f, :bf] ->
          Nx.add(Nx.multiply(a, 1.0 - alpha), Nx.multiply(Nx.as_type(b, Nx.type(a)), alpha))

        # Non-float leaves (PRNG keys, counters) can't be interpolated —
        # keep endpoint A's verbatim.
        _ ->
          a
      end

    is_map(a) and not is_struct(a) ->
      Map.new(a, fn {k, v} -> {k, lerp_tree.(lerp_tree, v, Map.fetch!(b, k), alpha)} end)

    true ->
      a
  end
end

data_a = la.params.data
data_b = lb.params.data

IO.puts(
  String.pad_trailing("alpha", 8) <>
    Enum.map_join(entries, "", fn {name, _, _} -> String.pad_trailing(name, 14) end)
)

for i <- 0..(points - 1) do
  alpha = i / (points - 1)
  data = lerp_tree.(lerp_tree, data_a, data_b, alpha)
  params = %{la.params | data: data}

  cells =
    Enum.map_join(entries, "", fn {_name, win, template} ->
      case rollout.(la.predict_fn, params, win, template) do
        {:escape, t} -> String.pad_trailing("esc@#{t}", 14)
        {:absorbed, _} -> String.pad_trailing("ABSORBED", 14)
      end
    end)

  IO.puts(String.pad_trailing(Float.round(alpha, 2) |> to_string(), 8) <> cells)
end
