defmodule ExPhil.Interp.BasinRollout do
  @moduledoc """
  Mental rollout of the crouch basin — the offline escape-competence probe
  (INIT_FORENSICS_OPTIONS option 1 v3 / option 6).

  Simulates the basin closed loop entirely in embedding space: each step
  embeds a SquatWait frame whose prev-action is the policy's own
  deterministic output from the previous step, appends it to the rolling
  window, and asks the policy again. Basin dynamics are trivial (crouch
  persists, action_frame increments and loops at 120) until a B EDGE
  (press after release) occurs — live, that edge is a shine, i.e. escape.

  Validated 2026-07-27 against the 12-seed crouch zoo: real-entry rollouts
  predict live escape 9/12, and the misses are exactly the
  "self-consistent escaper" tier (escape live via their own routes but
  cannot exit foreign holes). See scripts/probe_basin_rollout.exs for the
  cross-entry matrix and INIT_FORENSICS_OPTIONS.md for findings.

  Used per-epoch during training (`--probe-basin` in
  train_multishine_policy.exs) as the option-6 early-reject metric:
  watching WHEN escape competence emerges, seed by seed.
  """

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Data.{Peppi, RecoverySynth}
  alias ExPhil.Interp.Activations

  @window_size 16
  @af_loop 120

  # Every entry builder and the rollout accept `:config` (the policy's
  # export config) + `:delay_id` — queue/delay-id (336-dim) policies embed
  # in their trained layout via Activations.embed_frames/embed_config_for
  # (2026-08-04 W1/W2 unblock). Omitting them keeps the classic 288-dim
  # default layout, so crouch-zoo callers are unchanged.

  @doc """
  Build the synthetic training-style entry (post-shine lead-in -> Squat)
  from the canonical fixture. Returns `{window_tensor, template_frame}`.
  """
  def entry_synthetic(fixture \\ "test/fixtures/replays/fox_multishine_closed.slp", opts \\ []) do
    {:ok, replay} = Peppi.parse(fixture)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.reject(fn %{controller: c} ->
        c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
      end)

    block = RecoverySynth.build_crouch(frames, port: 1, max_af: 40, lead_in: 16, ratio: 0.001)

    pre = Enum.take(block, length(block) - 40)
    {embed_window(pre, opts), List.last(pre)}
  end

  @doc """
  Build an entry from a real replay's frames ending just before `at`
  (e.g. the first absorbed frame from chain_break_forensics).
  """
  def entry_from_replay(path, at, opts \\ []) do
    {:ok, replay} = Peppi.parse(path)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    entry_from_frames(Enum.slice(frames, max(at - @window_size, 0), @window_size), opts)
  end

  @doc """
  Build an entry directly from a training-frame list (the last
  #{@window_size} frames become the window; shorter lists are left-padded
  by repeating the first frame). The `{window_tensor, template}` contract
  every other entry builder reduces to — public so callers can slice
  openings from an already-parsed replay or synthesis output
  (task #3, 2026-08-02).
  """
  def entry_from_frames(frames, opts \\ []) when frames != [] do
    {embed_window(frames, opts), List.last(frames)}
  end

  @doc """
  Auto-detect the absorbed entry in a dead seed's replay: the first index
  where a sustained SquatWait run begins (>= `:min_run` frames at
  >= `:occupancy` SquatWait, defaults 120 / 0.9 — seed g sat at 97.1%
  Squat occupancy from frame 104). Returns `{:ok, index}` or :none.

  This replaces hand-curated frame numbers (g@104, m@104) so farms can
  feed ANY opening-death replay via --probe-entries without forensics
  first (INIT_FORENSICS: the early-reject blind spot is exactly the
  opening routes the curated entries don't cover).
  """
  def find_absorbed_index(frames, opts \\ []) do
    min_run = Keyword.get(opts, :min_run, 120)
    occupancy = Keyword.get(opts, :occupancy, 0.9)
    squat_wait = ExPhil.Constants.squat_wait()

    flags =
      frames
      |> Enum.map(&(&1.game_state.players[1].action == squat_wait))
      |> List.to_tuple()

    n = tuple_size(flags)

    if n < min_run do
      :none
    else
      Enum.find_value(0..(n - min_run), :none, fn i ->
        if elem(flags, i) do
          hits = Enum.count(i..(i + min_run - 1), &elem(flags, &1))
          if hits >= occupancy * min_run, do: {:ok, i}
        end
      end)
    end
  end

  @doc """
  `entry_from_replay/2` with the index auto-detected by
  `find_absorbed_index/2`. Returns `{:ok, {entry, index}}` or :none
  (no sustained squat run / unparseable replay).
  """
  def entry_from_absorbed_replay(path, opts \\ []) do
    with {:ok, replay} <- Peppi.parse(path),
         frames =
           replay
           |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
           |> Enum.reject(&(&1.game_state.frame < 0)),
         {:ok, at} <- find_absorbed_index(frames, opts) do
      pre = Enum.slice(frames, max(at - @window_size, 0), @window_size)
      {:ok, {entry_from_frames(pre, opts), at}}
    else
      _ -> :none
    end
  end

  @doc """
  Roll the basin closed loop. `predict_fn`/`params` over `{1, window, embed}`
  returning the 6-head logit tuple (buttons first, B at index 1).

  Returns `{:escape, frame}` or `{:absorbed, max_frames}`.
  """
  def rollout(predict_fn, params, {entry_window, template}, opts \\ []) do
    max_frames = Keyword.get(opts, :max_frames, 200)
    squat_wait = ExPhil.Constants.squat_wait()
    player0 = template.game_state.players[1]
    base_frame = template.game_state.frame

    embed_config = Activations.embed_config_for(Keyword.get(opts, :config))
    delay_id = Keyword.get(opts, :delay_id) || 0
    queue_depth = Map.get(embed_config, :queue_depth) || 1
    ring0 = List.duplicate(template.controller, queue_depth)

    result =
      Enum.reduce_while(1..max_frames, {entry_window, ring0}, fn t, {win, ring} ->
        prev_ctrl = hd(ring)
        af = rem(t - 1, @af_loop) + 1
        player = %{player0 | action: squat_wait, action_frame: af, on_ground: true}

        gs = %{
          template.game_state
          | frame: base_frame + t,
            players: Map.put(template.game_state.players, 1, player)
        }

        emb =
          ExPhil.Embeddings.Game.embed(gs, prev_ctrl, 1,
            config: embed_config,
            queue_controllers: ring,
            delay_id: delay_id
          )
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Nx.reshape({1, :auto})

        win = Nx.concatenate([Nx.slice_along_axis(win, 1, @window_size - 1, axis: 0), emb], axis: 0)
        out = predict_fn.(params, Nx.new_axis(win, 0))
        ctrl = decode_controller(out)

        if ctrl.button_b and not prev_ctrl.button_b do
          {:halt, {:escape, t}}
        else
          {:cont, {win, [ctrl | Enum.take(ring, queue_depth - 1)]}}
        end
      end)

    case result do
      {:escape, t} -> {:escape, t}
      _ -> {:absorbed, max_frames}
    end
  end

  defp embed_window(frames, opts) do
    ds =
      Activations.embed_frames(frames, Keyword.get(opts, :config) || %{},
        delay_id: Keyword.get(opts, :delay_id),
        use_prev_action: true
      )

    emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
    {n, _} = Nx.shape(emb)

    emb =
      if n < @window_size do
        pad = Nx.tile(Nx.slice_along_axis(emb, 0, 1, axis: 0), [@window_size - n, 1])
        Nx.concatenate([pad, emb], axis: 0)
      else
        emb
      end

    {n2, _} = Nx.shape(emb)
    Nx.slice_along_axis(emb, n2 - @window_size, @window_size, axis: 0)
  end

  @doc """
  Deterministic controller decode from the 6-head logit tuple (buttons
  thresholded at 0, sticks argmaxed). Public since 2026-08-02 — CycleSim
  shares it.
  """
  def decode_controller(out) do
    buttons = out |> elem(0) |> Nx.squeeze() |> Nx.to_flat_list()
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
end
