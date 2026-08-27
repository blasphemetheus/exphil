defmodule ExPhil.Training.DataLossWeightsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Data

  # AWBC loss-weight channel (task: offline RL / OFFLINE_RL_SPEC F5):
  # per-frame loss weights passed via :loss_weights must multiply into
  # the batch :frame_weights, keyed by each window's supervised (last)
  # frame — same convention as actions.

  defp neutral_controller do
    %{
      button_a: false, button_b: false, button_x: false, button_y: false,
      button_z: false, button_l: false, button_r: false, button_d_up: false,
      main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0, r_shoulder: 0.0
    }
  end

  defp dataset(num_frames, embed_dim) do
    %Data{
      frames: List.duplicate(%{controller: neutral_controller()}, num_frames),
      size: num_frames,
      embedded_frames:
        Nx.iota({num_frames, embed_dim}, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend),
      metadata: %{},
      embed_config: %{}
    }
  end

  test "loss_weights multiply into batch frame_weights at the supervised frame" do
    num_frames = 60
    window = 10

    # Distinctive per-frame weights: frame i carries weight i * 0.1
    loss_weights = Enum.map(0..(num_frames - 1), &(&1 * 0.1))

    [batch | _] =
      Data.batched_sequences(dataset(num_frames, 8),
        batch_size: 4,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: false,
        gpu: false,
        loss_weights: loss_weights
      )
      |> Enum.take(1)

    # All-neutral controllers -> base frame weight 0.25 per frame.
    # Sequence k's supervised frame = k + window - 1.
    expected =
      Enum.map(0..3, fn k -> 0.25 * ((k + window - 1) * 0.1) end)

    assert batch.frame_weights |> Nx.to_flat_list() |> Enum.zip(expected) |>
             Enum.all?(fn {got, want} -> abs(got - want) < 1.0e-5 end)
  end

  test "omitting loss_weights leaves frame_weights unchanged" do
    [batch | _] =
      Data.batched_sequences(dataset(30, 8),
        batch_size: 4,
        window_size: 10,
        stride: 1,
        lazy: true,
        shuffle: false,
        gpu: false
      )
      |> Enum.take(1)

    assert batch.frame_weights |> Nx.to_flat_list() |> Enum.all?(&(abs(&1 - 0.25) < 1.0e-6))
  end
end
