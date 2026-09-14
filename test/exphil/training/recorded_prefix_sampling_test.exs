defmodule ExPhil.Training.RecordedPrefixSamplingTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{RecordedPrefixSampling, Labels, Data}

  defp frames(start, count) do
    for index <- 0..(count - 1) do
      %{
        game_state: %ExPhil.Bridge.GameState{frame: start + index},
        controller: %ExPhil.Bridge.ControllerState{
          button_b: rem(index, 2) == 0,
          main_stick: %{x: 0.5, y: 0.0},
          c_stick: %{x: 0.5, y: 0.5},
          l_shoulder: 0.0,
          r_shoulder: 0.0
        },
        label_source: :recorded
      }
    end
  end

  test "only marked early target frames are repeated after delaying labels" do
    original = frames(100, 8)
    [marked] = RecordedPrefixSampling.mark([original], "teacher")
    shifted = Labels.at_delay(marked, 2)

    assert Enum.map(shifted, & &1.controller) ==
             Enum.map(Labels.at_delay(original, 2), & &1.controller)

    {weights, stats} = RecordedPrefixSampling.frame_weights([frames(0, 3), shifted], 4, 2)
    assert weights == [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    assert stats.draws == 15
    assert stats.early_targets == 2
  end

  test "actual sampler retains every target and repeats complete boundary-safe windows" do
    lists = RecordedPrefixSampling.mark([frames(100, 6), frames(106, 6)], "teacher")
    dataset = Data.from_frame_lists(lists)
    dataset = %{dataset | embedded_frames: Nx.tensor(for index <- 0..11, do: [index * 1.0])}
    {weights, _} = RecordedPrefixSampling.frame_weights(lists, 4, 2)

    windows =
      Data.batched_sequences(dataset,
        lazy: true,
        window_size: 3,
        batch_size: 5,
        seed: 42,
        gpu: false,
        sampling_weights: weights
      )
      |> Enum.flat_map(&Nx.to_list(&1.states))

    counts = Enum.frequencies_by(windows, &(List.last(&1) |> hd() |> trunc()))
    for index <- 0..11, do: assert(counts[index] == if(index in [0, 1, 6, 7], do: 4, else: 1))

    assert Enum.all?(windows, fn window ->
             ids = Enum.map(window, &(hd(&1) |> trunc()))
             Enum.all?(ids, &(div(&1, 6) == div(List.last(ids), 6)))
           end)
  end

  test "invalid or unmarked weighting fails closed" do
    for {weight, prefix} <- [{0, 2}, {2, 0}, {1.5, 2}] do
      assert_raise ArgumentError, fn ->
        RecordedPrefixSampling.frame_weights([], weight, prefix)
      end
    end

    assert_raise ArgumentError, fn ->
      RecordedPrefixSampling.frame_weights([frames(0, 2)], 4, 2)
    end
  end
end
