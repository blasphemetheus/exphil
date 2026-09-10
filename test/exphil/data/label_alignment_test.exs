defmodule ExPhil.Data.LabelAlignmentTest do
  @moduledoc """
  Pins GOTCHA #113: Slippi records a frame's controller input on the frame
  whose (post-update) state it PRODUCED. Two facts, both measured on the
  multishine fixture:

    1. The format fact: at a WAIT -> DASH transition, the full-X input is
       recorded on the first DASH frame, not on the preceding WAIT frame.
    2. The pipeline consequence: training frames built at frame_delay 0
       pair the WAIT-state frame with a NEUTRAL controller (leaked
       target); at frame_delay 1 the WAIT-state frame is paired with the
       dash input that follows it (causal target).

  If (1) ever flips (a peppi/NIF change to pre/post-frame handling) or
  (2) regresses (a delay refactor), a delay-0 default silently trains a
  policy that cannot initiate actions.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Data.Peppi
  alias ExPhil.Training.Streaming

  # A movement-heavy fixture: port 1 (Mewtwo) has 47 WAIT->DASH transitions
  # (the multishine fixtures have none — Fox never dashes in them).
  @fixture "test/fixtures/replays/mewtwo_ground_neutral.slp"
  @subject_port 1
  @wait 14
  @dash 20

  defp full_x?(%{main_stick_x: x}) when is_number(x), do: abs(x - 0.5) * 2 >= 0.75
  defp full_x?(%{main_stick: %{x: x}}) when is_number(x), do: abs(x - 0.5) * 2 >= 0.75
  defp full_x?(_), do: false

  defp fox_port(_path), do: @subject_port

  @tag :nif
  test "format fact: the dash input is recorded on the DASH frame, not the WAIT frame" do
    port = fox_port(@fixture)
    {:ok, %{frames: frames}} = Peppi.parse(@fixture)
    arr = :array.from_list(frames)

    subj = fn i -> Map.get(:array.get(i, arr).players || %{}, port) end
    act = fn i -> (subj.(i) && trunc(subj.(i).action || 0)) || -1 end

    transitions =
      for i <- 1..(length(frames) - 1), act.(i) == @dash, act.(i - 1) == @wait, do: i

    # The fixture must actually contain WAIT->DASH transitions for the test
    # to mean anything.
    assert length(transitions) >= 3, "fixture has too few WAIT->DASH transitions"

    on_dash = Enum.count(transitions, fn i -> full_x?(subj.(i).controller) end)
    on_wait = Enum.count(transitions, fn i -> full_x?(subj.(i - 1).controller) end)

    assert on_dash / length(transitions) >= 0.9,
           "expected the dash input on the DASH frame (got #{on_dash}/#{length(transitions)})"

    assert on_wait / length(transitions) <= 0.1,
           "expected NO dash input on the preceding WAIT frame (got #{on_wait}/#{length(transitions)})"
  end

  @tag :nif
  test "pipeline: Peppi frames pair the WAIT frame with the dash input at frame_delay 0 (causal by construction)" do
    port = fox_port(@fixture)

    pair_rate = fn delay ->
      {:ok, frames, _} =
        Streaming.parse_chunk([{@fixture, port}],
          frame_delay: delay,
          subject_character: "Mewtwo",
          show_progress: false
        )

      arr = :array.from_list(frames)
      n = length(frames)

      act = fn i ->
        p = :array.get(i, arr).game_state.players[1]
        (p && trunc(p.action || 0)) || -1
      end

      # WAIT frames whose next STATE is DASH: the causal label is full-X
      idx = for i <- 0..(n - 2), act.(i) == @wait, act.(i + 1) == @dash, do: i
      full = Enum.count(idx, fn i -> full_x?(:array.get(i, arr).controller) end)
      {length(idx), full}
    end

    {n0, full0} = pair_rate.(0)
    assert n0 >= 3

    # delay 0 IS the causal pairing now: the WAIT frame carries the input
    # issued from it (the dash). Before the rebase this read <= 0.1 (leak).
    assert full0 / n0 >= 0.9, "frame_delay 0 must pair WAIT with the dash input, got #{full0}/#{n0}"
  end

  @tag :nif
  test "frame_delay k is additional: state[t] pairs with the input issued from t+k" do
    {:ok, replay} = Peppi.parse(@fixture, player_port: @subject_port)
    base = Peppi.to_training_frames(replay, player_port: @subject_port, opponent_port: 2)
    d2 = Peppi.to_training_frames(replay, player_port: @subject_port, opponent_port: 2, frame_delay: 2)

    assert length(base) == length(replay.frames) - 1, "causal pairing drops exactly the last frame"
    assert length(d2) == length(base) - 2

    base_arr = :array.from_list(base)

    # Every delayed frame: same state as the base frame at its observed
    # index, controller equal to the base (causal) controller 2 frames on.
    Enum.with_index(d2)
    |> Enum.take_every(97)
    |> Enum.each(fn {f, i} ->
      assert f.game_state == :array.get(i, base_arr).game_state
      assert f.controller == :array.get(i + 2, base_arr).controller
    end)
  end

  test "the leaked pairing cannot be built: causal_pairs drops non-contiguous successors" do
    mk = fn n, c -> %{game_state: %{frame: n}, controller: c} end
    frames = [mk.(0, :a), mk.(1, :b), mk.(2, :c), mk.(5, :d), mk.(6, :e)]

    assert Peppi.causal_pairs(frames) == [
             %{game_state: %{frame: 0}, controller: :b},
             %{game_state: %{frame: 1}, controller: :c},
             %{game_state: %{frame: 5}, controller: :e}
           ]
  end
end
