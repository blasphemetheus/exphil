defmodule ExPhil.Training.ClipHistoryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Data

  defp frame(number) do
    %{
      game_state: %{frame: number},
      controller: %{
        button_a: false,
        button_b: true,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false,
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0
      }
    }
  end

  defp dataset do
    lists = [[], Enum.map(10..12, &frame/1), Enum.map(13..14, &frame/1), [frame(20)]]
    data = Data.from_frame_lists(lists)
    %{data | embedded_frames: Nx.tensor([[10.0], [11.0], [12.0], [13.0], [14.0], [20.0]])}
  end

  defp batches(data, opts \\ []) do
    Data.batched_sequences(
      data,
      Keyword.merge([lazy: true, gpu: false, shuffle: false, batch_size: 2, window_size: 4], opts)
    )
    |> Enum.to_list()
  end

  test "actual windows retain every early target and never borrow adjacent clip history" do
    rows = dataset() |> batches() |> Enum.flat_map(&Nx.to_list(&1.states))

    assert rows ==
             Enum.map(
               [
                 [10, 10, 10, 10],
                 [10, 10, 10, 11],
                 [10, 10, 11, 12],
                 [13, 13, 13, 13],
                 [13, 13, 13, 14],
                 [20, 20, 20, 20]
               ],
               fn row -> Enum.map(row, &[&1 * 1.0]) end
             )

    assert Data.sequence_target_indices(dataset(), 4) == Enum.to_list(0..5)
  end

  test "gaps within a clip reset history, and stride restarts at each boundary" do
    data = Data.from_frame_lists([[frame(10), frame(11), frame(20), frame(21), frame(22)]])
    data = %{data | embedded_frames: Nx.tensor([[10.0], [11.0], [20.0], [21.0], [22.0]])}
    assert Data.sequence_target_indices(data, 4, 2) == [0, 2, 4]
    rows = batches(data, stride: 2) |> Enum.flat_map(&Nx.to_list(&1.states))

    assert rows == [
             [[10.0], [10.0], [10.0], [10.0]],
             [[20.0], [20.0], [20.0], [20.0]],
             [[20.0], [20.0], [21.0], [22.0]]
           ]
  end

  test "loss, sampling and teacher rows stay aligned to padded targets" do
    teacher = Nx.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])

    [batch] =
      batches(dataset(),
        batch_size: 8,
        sampling_weights: [0, 0, 0, 1, 0, 0],
        loss_weights: [1, 2, 3, 4, 5, 6],
        distill_teacher: teacher,
        distill_mask: [false, false, false, true, false, false],
        transition_weight: 7
      )

    assert Nx.to_list(batch.states) == [[[13.0], [13.0], [13.0], [13.0]]]
    assert Nx.to_flat_list(batch.frame_weights) == [4.0]
    assert Nx.to_flat_list(batch.teacher_logits) == [3.0]
    assert Nx.to_flat_list(batch.distill_mask) == [1.0]
  end

  test "systematic probe batches use the same padded sequence IDs" do
    {stream, indices} =
      Data.strided_sequence_batches(dataset(), window_size: 4, every: 2, gpu: false)

    assert indices == [0, 2, 4]
    rows = Enum.flat_map(stream, &Nx.to_list(&1.states))
    assert Enum.map(rows, &List.last/1) == [[10.0], [12.0], [14.0]]
  end

  test "single-frame windows still weight within-clip transitions, not clip boundaries" do
    first = frame(10)
    changed = put_in(frame(11), [:controller, :button_x], true)
    data = Data.from_frame_lists([[first, changed], [first]])
    data = %{data | embedded_frames: Nx.tensor([[10.0], [11.0], [10.0]])}
    [batch] = batches(data, batch_size: 4, window_size: 1, transition_weight: 7)
    assert Nx.to_flat_list(batch.frame_weights) == [1.0, 7.0, 1.0]
  end

  test "legacy short datasets yield no backwards or out-of-bounds windows" do
    data = dataset()
    data = %{data | metadata: %{}}
    assert batches(data, window_size: 9, stride: 3) == []
    {stream, indices} = Data.strided_sequence_batches(data, window_size: 9, gpu: false)
    assert indices == []
    assert Enum.to_list(stream) == []
  end
end
