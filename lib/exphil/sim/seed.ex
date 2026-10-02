defmodule ExPhil.Sim.Seed do
  @moduledoc """
  Seed the sim at any frame of a REAL replay (the coach's "load the game"
  step, EVALS_PROGRAM / COACH_STYLE_PRODUCTS, 2026-09-21).

  The sim has no "jump to frame N"; it has exact input replay. So: build
  the match from the replay's metadata (stage, characters, costumes, the
  game-start RNG seed), then step the NIF batch from Slippi frame −123 with
  each frame's ORIGINAL pre-frame record through the replay-exact entry
  point (`msl_batch_step_replay`, our addition to the sim clone), feeding
  exactly what the sim's own validator feeds (`tools/validation/native.c`
  `build_input`): the physical button mask, raw stick bytes (UCF lanes),
  Slippi's processed sticks as the fighter's `nml_*` lanes, physical L/R
  triggers ×140, the FrameStart RNG seed and the per-player pre-frame seed.
  Every post-frame is compared with the replay's; the first divergence is
  reported, never hidden. At frame N, save → a savestate the drill / search
  / regret tools can restore, plus the sim's own last `warm` frames as agent
  history.

  Alignment: the sim's reset row is Slippi −124 (clone patched to match the
  validator, 2026-09-21); step k produces Slippi frame −124 + k and consumes
  the pre-frame record of THAT frame. Verified exact on local Dolphin games
  (the sim validator passes the same games full-length).
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Sim.Env
  import Bitwise

  @doc """
  Options: `:frame` (target Slippi frame, default 0), `:warm` (30),
  `:backend` (:nif — the only backend with the replay-exact step),
  `:tolerance` (position tolerance, default 0.0 = bit-exact floats).

  `:frames` (list of extra Slippi frames to save along the way; each becomes an
  entry in `saves`: `%{frame, blob, state_id, history, state, diverged?}`).

  Returns `{:ok, %{sim, blob, state_id, frame, divergence, history, saves, summary, players, stage}}`.
  """
  def from_replay(path, opts \\ []) do
    target = Keyword.get(opts, :frame, 0)
    warm = Keyword.get(opts, :warm, 30)
    tol = Keyword.get(opts, :tolerance, 0.0)

    with {:ok, meta} <- Peppi.metadata(path),
         {:ok, replay} <- Peppi.parse(path) do
      ports = meta.players |> Enum.sort_by(& &1.port)
      players = Enum.map(ports, fn p -> %{character: String.downcase(p.character_name || "fox") |> String.replace(~r/[^a-z0-9]/, ""), costume: p.costume || 0} end)
      port_ids = Enum.map(ports, & &1.port)

      {:ok, sim} = Env.start(Keyword.get(opts, :backend, :nif), stage: meta.stage, players: players, batch_size: 1, seed: meta.random_seed || 0, ucf_cardinals: Keyword.get(opts, :ucf_cardinals, 1))
      frames = replay.frames |> Enum.sort_by(& &1.frame_number)
      by_frame = Map.new(frames, &{&1.frame_number, &1})
      frames = Enum.filter(frames, &(&1.frame_number <= target))
      {:ok, [gs0]} = Env.frames(sim)

      # savestates wanted along the way (coach review): every frame in :frames, plus the target
      wanted = MapSet.new(Keyword.get(opts, :frames, []) ++ [target])

      {history, divergence, gs, saves} =
        Enum.reduce_while(frames, {[], nil, gs0, []}, fn f, {hist, div, gs, saves} ->
          t = f.frame_number
          row = replay_row(f, port_ids)

          case Env.step_replay(sim, [row]) do
            {:ok, [next], _} ->
              div = div || mismatch(next, by_frame[t], port_ids, tol)
              hist = Enum.take([{gs, row} | hist], warm)

              saves =
                if MapSet.member?(wanted, next.frame) do
                  {:ok, blob, sid} = Env.save(sim, 0, keep: true)
                  [%{frame: next.frame, blob: blob, state_id: sid, history: hist |> Enum.reverse() |> Enum.map(&elem(&1, 0)), state: next, diverged?: div != nil} | saves]
                else
                  saves
                end

              if next.frame >= target, do: {:halt, {hist, div, next, saves}}, else: {:cont, {hist, div, next, saves}}

            {:error, reason} ->
              {:halt, {hist, div || {t, :step_error, reason}, gs, saves}}
          end
        end)

      last = List.first(saves) || (fn -> {:ok, blob, sid} = Env.save(sim, 0, keep: true); %{frame: gs.frame, blob: blob, state_id: sid} end).()
      blob = last.blob
      sid = last.state_id

      {:ok,
       %{
         sim: sim,
         blob: blob,
         state_id: sid,
         frame: gs.frame,
         divergence: divergence,
         history: history |> Enum.reverse() |> Enum.map(&elem(&1, 0)),
         saves: Enum.reverse(saves),
         summary: %{p1: gs.players[1] && Map.take(gs.players[1], [:x, :y, :action, :percent, :stock]), p2: gs.players[2] && Map.take(gs.players[2], [:x, :y, :action, :percent, :stock])},
         players: players,
         stage: meta.stage
       }}
    end
  end

  @doc """
  The 80-byte `MslReplayInput` row for one replay frame: header (FrameStart
  seed, fighter pre-frame seed, validity flags) + 4 × 16-byte player lanes in
  port order (`port_ids`), zero for absent ports.
  """
  def replay_row(%Peppi.GameFrame{} = f, port_ids) do
    first = Enum.find_value(port_ids, fn p -> case f.players[p] do %{controller: %{processed: %{rng_seed: s}}} when is_integer(s) -> s; _ -> nil end end)
    flags = ((if is_integer(f.frame_seed), do: 1, else: 0) ||| (if is_integer(first), do: 2, else: 0)) &&& Application.get_env(:exphil, :seed_flags_mask, 3)

    players =
      port_ids
      |> Enum.map(&player_lane(f.players[&1]))
      |> Enum.concat(List.duplicate(<<0::size(16 * 8)>>, 4))
      |> Enum.take(4)
      |> IO.iodata_to_binary()

    <<(f.frame_seed || 0)::unsigned-32-little, (first || 0)::unsigned-32-little, flags::8, 0::size(7 * 8)>> <> players
  end


  # One 16-byte MslReplayInputPlayer, validator-equivalent (native.c build_input).
  defp player_lane(%{controller: %{processed: p} = c}) when not is_nil(p) do
    # raw lanes: physical bytes when the replay carries them (UCF consumers), else the processed floats
    mx = p.raw_main_x || stick_i8(p.main_x)
    my = p.raw_main_y || stick_i8(p.main_y)
    cx = p.raw_c_x || stick_i8(p.c_x)
    cy = p.raw_c_y || stick_i8(p.c_y)
    # nml lanes: Slippi's processed (post-clamp) sticks, truncated ×80 like the validator
    <<(p.physical_buttons || 0)::unsigned-16-little, mx::signed-8, my::signed-8, cx::signed-8, cy::signed-8,
      trigger_u8(c.l_trigger)::8, trigger_u8(c.r_trigger)::8,
      stick_i8(p.main_x)::signed-8, stick_i8(p.main_y)::signed-8, stick_i8(p.c_x)::signed-8, stick_i8(p.c_y)::signed-8,
      3::8, 0::size(3 * 8)>>
  end

  defp player_lane(_), do: <<0::size(16 * 8)>>

  # native.c processed_stick_i8: clamp to ±1 then TRUNCATE (int8)(v * 80.0F) — in FLOAT32.
  # Slippi floats are f32; the f64 product of two f32s is exact, so rounding
  # that product to f32 reproduces the C multiply bit-for-bit (0.6125*80 is
  # 48.999… in f64 but 49.0 in f32 — truncation flips).
  defp stick_i8(v) when is_number(v) do
    cond do
      v <= -1.0 -> -80
      v >= 1.0 -> 80
      true -> trunc(f32(v * 80.0))
    end
  end

  defp stick_i8(_), do: 0

  # native.c trigger_u8: 0 unless > 0; 140 at >= 1; else lrintf(v * 140.0F) (ties to even)
  defp trigger_u8(v) when is_number(v) and v > 0.0 do
    if v >= 1.0, do: 140, else: lrint(f32(v * 140.0))
  end

  defp trigger_u8(_), do: 0

  defp f32(x), do: (<<x::float-32>> |> then(fn <<y::float-32>> -> y end))

  defp lrint(x) do
    n = trunc(x)
    frac = x - n
    cond do
      frac > 0.5 -> n + 1
      frac < 0.5 -> n
      rem(n, 2) == 0 -> n
      true -> n + 1
    end
  end

  # First mismatching field between the sim's post-frame and the replay's.
  defp mismatch(_gs, nil, _ports, _tol), do: nil

  # The sim keys its players by SLOT (1, 2, … in ascending replay-port order —
  # the order replay_row/2 lays the lanes out in); the replay keys them by the
  # real controller port. Until 2026-10-02 this looked the sim player up by
  # the real port, so on any game not played on ports 1+2 the lookup was nil,
  # the comparison was skipped and `divergence` stayed nil on games that had
  # drifted by tens of units (GOTCHA #137).
  defp mismatch(gs, rf, ports, tol) do
    ports
    |> Enum.with_index(1)
    |> Enum.find_value(fn {port, slot} ->
      sp = gs.players[slot]
      rp = rf.players[port]

      cond do
        rp == nil -> nil
        sp == nil -> {rf.frame_number, port, :missing_sim_player, slot, nil}
        abs(sp.x - rp.x) > tol -> {rf.frame_number, port, :x, rp.x, sp.x}
        abs(sp.y - rp.y) > tol -> {rf.frame_number, port, :y, rp.y, sp.y}
        sp.action != rp.action -> {rf.frame_number, port, :action, rp.action, sp.action}
        abs(sp.percent - rp.percent) > 0.01 -> {rf.frame_number, port, :percent, rp.percent, sp.percent}
        true -> nil
      end
    end)
  end
end
