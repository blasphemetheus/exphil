defmodule ExPhil.Training.ConversionSamplingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.ConversionSampling

  @wait 14
  @hitstun 75

  # Minimal drill-shaped frame: ConversionSampling only reads
  # game_state.players[port].x / .action
  defp frame(p1_x, p1_action, p2_x, p2_action) do
    %{
      game_state: %{
        players: %{
          1 => %{x: p1_x, action: p1_action},
          2 => %{x: p2_x, action: p2_action}
        }
      }
    }
  end

  # Same geometry as the ReplayStats walk-in: P2 parked at 0, P1 holds at
  # -60 for 20 frames, walks in 1/frame for 45, holds at -15. Onset at
  # index 60, closure start at index 0, n = 120. If `converts?`, P2 enters
  # hitstun 10 frames after engagement.
  defp walk_in_replay(converts?) do
    p1x =
      List.duplicate(-60.0, 20) ++
        Enum.map(1..45, fn i -> -60.0 + i end) ++ List.duplicate(-15.0, 55)

    p2_actions =
      if converts? do
        List.duplicate(@wait, 70) ++ List.duplicate(@hitstun, 20) ++ List.duplicate(@wait, 30)
      else
        List.duplicate(@wait, 120)
      end

    Enum.zip(p1x, p2_actions)
    |> Enum.map(fn {x, p2a} -> frame(x, @wait, 0.0, p2a) end)
  end

  describe "frame_weights/2" do
    test "marks the converting-approach span with the weight, 1.0 elsewhere" do
      {weights, stats} = ConversionSampling.frame_weights([walk_in_replay(true)], 3.0)

      assert length(weights) == 120
      assert stats.frames == 120
      assert stats.approaches == 1
      assert stats.conversions == 1

      # Span = closure start (0) through onset(59) + 45 = 104, inclusive
      assert weights |> Enum.take(105) |> Enum.all?(&(&1 == 3.0))
      assert weights |> Enum.drop(105) |> Enum.all?(&(&1 == 1.0))
      assert stats.upweighted == 105
    end

    test "a replay with no conversions contributes all-1.0 weights" do
      {weights, stats} = ConversionSampling.frame_weights([walk_in_replay(false)], 3.0)

      assert Enum.all?(weights, &(&1 == 1.0))
      assert stats.approaches == 1
      assert stats.conversions == 0
      assert stats.upweighted == 0
    end

    test "frames without position data sample uniformly" do
      frames = List.duplicate(frame(nil, @wait, nil, @wait), 50)
      {weights, stats} = ConversionSampling.frame_weights([frames], 3.0)

      assert length(weights) == 50
      assert Enum.all?(weights, &(&1 == 1.0))
      assert stats.approaches == 0
      assert stats.conversions == 0
    end

    test "spans never bleed across replay boundaries" do
      # Converting replay first: its span ends at index 105 < 120, and the
      # second (non-converting) replay's weights start fresh at 1.0
      {weights, stats} =
        ConversionSampling.frame_weights([walk_in_replay(true), walk_in_replay(false)], 2.0)

      assert length(weights) == 240
      assert stats.conversions == 1

      {first, second} = Enum.split(weights, 120)
      assert Enum.count(first, &(&1 == 2.0)) == 105
      assert Enum.all?(second, &(&1 == 1.0))
    end

    test "empty replay list and empty replays are fine" do
      assert {[], %{frames: 0, upweighted: 0, approaches: 0, conversions: 0}} =
               ConversionSampling.frame_weights([], 3.0)

      assert {[], _} = ConversionSampling.frame_weights([[]], 3.0)
    end
  end
end
