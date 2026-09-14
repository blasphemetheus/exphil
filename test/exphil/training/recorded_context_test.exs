defmodule ExPhil.Training.RecordedContextTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Data, Labels, RecordedContext, RecordedFrames, RecordedPrefixSampling}

  defp frames do
    for number <- -5..12 do
      %{
        game_state: %{frame: number},
        controller: ExPhil.Bridge.ControllerState.neutral() |> Map.put(:button_b, true),
        label_source: :recorded
      }
    end
  end

  test "prefix shifts alongside targets, never becomes supervision, and retains earliest targets" do
    warm = RecordedContext.slice(frames(), 0, 10, 5)
    assert RecordedFrames.validate!(RecordedFrames.envelope([warm], %{})) == [warm]
    [marked] = RecordedPrefixSampling.mark([warm], "teacher")
    shifted = Labels.at_delay(marked, 2)
    dataset = Data.from_frame_lists([shifted])

    dataset = %{
      dataset
      | embedded_frames: Nx.tensor(Enum.map(shifted, &[&1.game_state.frame * 1.0]))
    }

    assert Data.sequence_target_indices(dataset, 4) == Enum.to_list(5..12)
    assert Data.sequence_target_indices(dataset, 4, 2) == [5, 7, 9, 11]
    {weights, stats} = RecordedPrefixSampling.frame_weights([shifted], 4, 2)
    assert weights == List.duplicate(0.0, 5) ++ [4.0, 4.0] ++ List.duplicate(1.0, 6)
    assert stats.targets == 8
    assert stats.draws == 14

    batches =
      Data.batched_sequences(dataset,
        lazy: true,
        gpu: false,
        shuffle: false,
        window_size: 4,
        batch_size: 64
      )
      |> Enum.to_list()

    assert hd(Nx.to_list(hd(batches).states)) == [[-3.0], [-2.0], [-1.0], [0.0]]

    weighted =
      Data.batched_sequences(dataset,
        lazy: true,
        gpu: false,
        window_size: 4,
        batch_size: 64,
        sampling_weights: weights
      )
      |> Enum.to_list()

    assert Nx.axis_size(hd(weighted).states, 0) == 14
    assert_raise ArgumentError, fn -> Data.batched(dataset) end
    assert_raise ArgumentError, fn -> Data.batched_sequences(dataset) end
  end

  test "cold starts still pad; missing history and internal input-only rows fail closed" do
    cold = RecordedContext.slice(frames(), 0, 10, 0) |> Labels.at_delay(2)
    dataset = Data.from_frame_lists([cold])

    dataset = %{
      dataset
      | embedded_frames: Nx.tensor(Enum.map(cold, &[&1.game_state.frame * 1.0]))
    }

    batch =
      Data.batched_sequences(dataset,
        lazy: true,
        gpu: false,
        shuffle: false,
        window_size: 4,
        batch_size: 64
      )
      |> Enum.at(0)

    assert hd(Nx.to_list(batch.states)) == List.duplicate([0.0], 4)
    assert_raise ArgumentError, fn -> RecordedContext.slice(frames(), 0, 10, 6) end
    broken = List.update_at(cold, 2, &Map.put(&1, :input_only, true))
    assert_raise ArgumentError, fn -> Data.from_frame_lists([broken]) end

    assert_raise ArgumentError, fn ->
      RecordedFrames.validate!(RecordedFrames.envelope([broken], %{}))
    end
  end
end
