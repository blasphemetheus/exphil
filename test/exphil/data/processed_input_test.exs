defmodule ExPhil.Data.ProcessedInputTest do
  use ExUnit.Case, async: true
  alias ExPhil.Data.Peppi

  test "original input annotations do not change training conversion" do
    {:ok, replay} = Peppi.parse("test/fixtures/replays/fox_multishine_closed_d1.slp")
    frames = Enum.take(replay.frames, 30)
    assert Enum.all?(frames, fn f ->
      Enum.all?(f.players, fn {_, p} -> match?(%Peppi.ProcessedInput{}, p.controller.processed) end)
    end)
    without_annotations = for f <- frames do
      %{f | players: Map.new(f.players, fn {port, p} ->
        {port, %{p | controller: %{p.controller | processed: nil}}}
      end)}
    end
    assert Peppi.to_training_frames(%{replay | frames: frames}) ==
             Peppi.to_training_frames(%{replay | frames: without_annotations})
  end
end
