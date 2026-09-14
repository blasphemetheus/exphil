defmodule ExPhil.Training.LabelSourceContractTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Data, Labels}

  defmodule CurrentOnly do
    def label(_expert, _player, _previous), do: {:ok, :held}
  end

  defmodule Broken do
    def label_ahead(_expert, _player, _delay, _previous), do: {:error, :broken}
  end

  defp frame(counter) do
    %{game_state: %{frame: counter, players: %{1 => %{}}}, controller: counter}
  end

  test "mixed sources cannot hide behind any first frame, even at delay zero" do
    recorded = Labels.tag([frame(0)], :recorded) |> hd()
    expert = Labels.tag([frame(1)], {:expert, CurrentOnly}) |> hd()

    for frames <- [[recorded, expert], [expert, recorded], [frame(0), expert]],
        delay <- [0, 1, 4] do
      assert_raise ArgumentError, ~r/Mixed label sources/, fn ->
        Labels.at_delay(frames, delay)
      end

      assert_raise ArgumentError, ~r/Mixed label sources/, fn ->
        Data.shift_actions(frames, delay)
      end
    end
  end

  test "different expert modules and malformed tags are rejected" do
    for tag <- [false, "recorded", {:expert}, {:expert, nil}, {:expert, CurrentOnly, :extra}] do
      assert_raise ArgumentError, ~r/Invalid label_source/, fn -> Labels.tag([], tag) end

      assert_raise ArgumentError, ~r/Invalid label_source/, fn ->
        Labels.source([Map.put(frame(0), :label_source, tag)])
      end
    end

    frames =
      Labels.tag([frame(0)], {:expert, CurrentOnly}) ++ Labels.tag([frame(1)], {:expert, Broken})

    assert_raise ArgumentError, ~r/Mixed/, fn -> Labels.source(frames) end
  end

  test "serialization preserves provenance and strict ingestion detects stripped tags" do
    frames = Labels.tag([frame(0), frame(1)], {:expert, CurrentOnly})
    decoded = frames |> :erlang.term_to_binary() |> :erlang.binary_to_term()
    assert Labels.source(decoded, require_tagged: true) == {:expert, CurrentOnly}
    partial = List.update_at(decoded, 1, &Map.delete(&1, :label_source))
    assert_raise ArgumentError, ~r/Mixed/, fn -> Labels.source(partial) end
    stripped = Enum.map(decoded, &Map.delete(&1, :label_source))

    assert_raise ArgumentError, ~r/Missing label_source/, fn ->
      Labels.at_delay(stripped, 0, require_tagged: true)
    end

    assert Labels.source(stripped) == :recorded
  end

  test "recorded windows must be contiguous throughout, including reset and gap combinations" do
    assert Data.shift_actions(Enum.map([0, 4, 2], &frame/1), 2) == []

    assert Data.shift_actions(Enum.map([0, 1, 5, 6], &frame/1), 1) ==
             [%{frame(0) | controller: 1}, %{frame(5) | controller: 6}]

    assert Data.shift_actions(Enum.map(0..3, &frame/1), 2) ==
             [%{frame(0) | controller: 2}, %{frame(1) | controller: 3}]
  end

  test "current commitment fallback is opt-in and strict projection can forbid it" do
    frames = Labels.tag([frame(0)], {:expert, CurrentOnly})

    # An expert with no projection cannot be dropped-by-default (that would
    # silently discard every frame) and cannot be held-by-default (the held
    # rule is measured wrong): it must be asked for.
    assert_raise ArgumentError, ~r/cannot project/, fn ->
      Labels.at_delay(frames, 3, expert: %{})
    end

    assert_raise ArgumentError, ~r/cannot project/, fn ->
      Labels.at_delay(frames, 3, expert: %{}, off_loop: :drop)
    end

    assert hd(Labels.at_delay(frames, 3, expert: %{}, off_loop: :hold)).controller == :held

    assert_raise ArgumentError, ~r/cannot project/, fn ->
      Labels.at_delay(frames, 3, expert: %{}, off_loop: :hold, require_projection: true)
    end

    assert_raise ArgumentError, ~r/off_loop/, fn ->
      Labels.at_delay(frames, 3, off_loop: :typo)
    end
  end

  test "invalid expert results and delays fail loudly" do
    frames = Labels.tag([frame(0)], {:expert, Broken})

    assert_raise ArgumentError, ~r/Invalid expert label result/, fn ->
      Labels.at_delay(frames, 2, expert: %{})
    end

    for delay <- [-1, 0.5, "2"] do
      assert_raise ArgumentError, ~r/nonnegative integer/, fn -> Labels.at_delay([], delay) end
    end
  end
end
