defmodule ExPhil.Training.LabelsTest do
  @moduledoc """
  INVARIANTS.md item 14: delayed labels come from the label source's OWN
  future. Recorded lists shift along the recording; expert lists ask the
  expert's phase-indexed label_ahead/4; shifting an expert list is not
  representable. The fixture is the oracle: on it, the expert's projected
  future and the recorded future must agree.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MultishineExpert
  alias ExPhil.Training.{Data, Labels}

  @fixture "test/fixtures/replays/fox_multishine_closed_d1.slp"

  setup_all do
    {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture, player_port: 1)

    frames =
      replay
      |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.take(1200)

    expert = MultishineExpert.from_frames(frames, player_port: 1)
    {:ok, frames: frames, expert: expert}
  end

  defp bx(c), do: {c.button_b == true, c.button_x == true}

  test "the expert learned a canonical cycle from the fixture", %{expert: expert} do
    assert length(expert.cycle) in 8..12
    assert {361, 1, true} in expert.cycle
    assert Enum.all?(expert.cycle, &Map.has_key?(expert.table, &1))
    assert map_size(expert.phase) == length(Enum.uniq(expert.cycle))
  end

  test "on the fixture, label_ahead == the recorded future for every loop frame and k in 1..5",
       %{frames: frames, expert: expert} do
    arr = List.to_tuple(frames)
    n = tuple_size(arr)

    for k <- 1..5 do
      pairs =
        for i <- 0..(n - 1 - k),
            f = elem(arr, i),
            MultishineExpert.on_loop?(expert, f.game_state.players[1]),
            target = elem(arr, i + k),
            target.game_state.frame == f.game_state.frame + k do
          {:ok, ahead} = MultishineExpert.label_ahead(expert, f.game_state.players[1], k)
          {bx(ahead), bx(target.controller)}
        end

      assert length(pairs) > 300, "k=#{k}: too few loop pairs"
      agree = Enum.count(pairs, fn {a, b} -> a == b end) / length(pairs)
      assert agree > 0.97, "k=#{k}: expert future agrees with the recording on only #{agree}"
    end
  end

  test "at_delay: a recorded list and its expert-tagged twin give the same B/X labels on the fixture",
       %{frames: frames, expert: expert} do
    recorded = Labels.tag(frames, :recorded)
    as_expert = Labels.tag(frames, {:expert, MultishineExpert})

    for k <- [1, 3, 4] do
      rec = recorded |> Labels.at_delay(k) |> Map.new(&{&1.game_state.frame, bx(&1.controller)})
      exp = as_expert |> Labels.at_delay(k, expert: expert) |> Map.new(&{&1.game_state.frame, bx(&1.controller)})

      common = rec |> Map.keys() |> Enum.filter(&Map.has_key?(exp, &1))
      assert length(common) > 500
      agree = Enum.count(common, &(rec[&1] == exp[&1])) / length(common)
      assert agree > 0.95, "k=#{k}: recorded-shift vs expert-ahead agree on only #{agree}"
    end
  end

  test "shift_actions refuses an expert-labeled list; at_delay without the expert refuses too",
       %{frames: frames, expert: expert} do
    tagged = Labels.tag(frames, {:expert, MultishineExpert})
    assert_raise ArgumentError, ~r/student's broken future/, fn -> Data.shift_actions(tagged, 4) end
    assert_raise ArgumentError, ~r/need the expert struct/, fn -> Labels.at_delay(tagged, 4) end
    # k = 0 and recorded lists are untouched
    assert Labels.at_delay(tagged, 0) == tagged
    assert length(Labels.at_delay(frames, 2)) == length(Data.shift_actions(frames, 2))
    assert length(Labels.at_delay(tagged, 2, expert: expert)) == length(tagged)
  end

  test "off the loop, label_ahead is the expert's current commitment (held); :drop omits it",
       %{frames: frames, expert: expert} do
    # A grounded WAIT state is off-loop: the expert commits to starting a shine
    wait = %{hd(frames).game_state.players[1] | action: 14, action_frame: 3, on_ground: true}
    refute MultishineExpert.on_loop?(expert, wait)
    {:ok, now} = MultishineExpert.label(expert, wait)
    {:ok, ahead} = MultishineExpert.label_ahead(expert, wait, 4)
    assert bx(ahead) == bx(now)

    off = [%{hd(frames) | game_state: %{hd(frames).game_state | players: %{1 => wait, 2 => hd(frames).game_state.players[2]}}}]
    tagged = Labels.tag(off, {:expert, MultishineExpert})
    assert length(Labels.at_delay(tagged, 4, expert: expert)) == 1
    assert Labels.at_delay(tagged, 4, expert: expert, off_loop: :drop) == []
  end
end
