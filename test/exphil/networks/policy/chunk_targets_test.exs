defmodule ExPhil.Networks.Policy.ChunkTargetsTest do
  @moduledoc """
  Chunk targets (2026-10-02): K auxiliary heads on the temporal policy
  predict the controller at t+1..t+K; the data path stacks those targets
  into the actions map with a same-game mask; export drops the heads.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Data

  @embed 16
  @window 4

  defp build(chunk_horizon) do
    Policy.build_temporal(
      backbone: :gru,
      embed_size: @embed,
      window_size: @window,
      hidden_size: 8,
      num_layers: 1,
      dropout: 0.0,
      head: :autoregressive,
      axis_buckets: 4,
      shoulder_buckets: 2,
      chunk_horizon: chunk_horizon
    )
  end

  test "the output becomes {head, {future_1..K}} and the plain build is unchanged" do
    {init, predict} = Axon.build(build(2), mode: :train)
    inputs = Map.merge(%{"state_sequence" => Nx.broadcast(0.0, {3, @window, @embed})}, tf(3))
    params = init.(inputs, Axon.ModelState.empty())
    %{prediction: {head, futures}} = predict.(params, inputs)

    assert tuple_size(head) == 6
    assert tuple_size(futures) == 2
    assert Nx.shape(elem(head, 0)) == {3, 8}
    for j <- 0..1, do: assert(Nx.shape(elem(elem(futures, j), 0)) == {3, 8})
    for j <- 0..1, do: assert(Nx.shape(elem(elem(futures, j), 1)) == {3, 5})

    names = params.data |> Map.keys()
    assert Enum.any?(names, &String.starts_with?(&1, "future1_"))
    assert Enum.any?(names, &String.starts_with?(&1, "future2_"))

    {init0, predict0} = Axon.build(build(nil), mode: :train)
    params0 = init0.(inputs, Axon.ModelState.empty())
    %{prediction: plain} = predict0.(params0, inputs)
    assert tuple_size(plain) == 6
    refute Enum.any?(Map.keys(params0.data), &String.starts_with?(&1, "future"))
  end

  test "future targets are stacked on axis 1 and masked past the end of the game" do
    # two games of 6 frames; window 4, stride 1 -> targets 3,4,5 | 9,10,11
    frames =
      for g <- 0..1, i <- 0..5 do
        %{
          game: g,
          state: nil,
          action: %{
            buttons: %{a: rem(i, 2) == 1, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
            main_x: i,
            main_y: 8,
            c_x: 8,
            c_y: 8,
            shoulder: 0
          }
        }
      end

    ds = %Data{
      frames: frames,
      size: 12,
      embedded_frames: Nx.broadcast(0.0, {12, @embed}),
      metadata: %{window_size: @window, stride: 1, sequence_starts: List.to_tuple(List.duplicate(0, 6) ++ List.duplicate(6, 6))}
    }

    [batch] =
      Data.batched_sequences(ds,
        batch_size: 16, window_size: @window, stride: 1, lazy: true, shuffle: false, drop_last: false,
        gpu: false, neutral_weight: 1.0, chunk_horizon: 2
      )
      |> Enum.to_list()

    # boundary-aware layout: every frame is a target (padded window starts)
    a = batch.actions
    assert Nx.shape(a.future_main_x) == {12, 2}
    assert Nx.shape(a.future_buttons) == {12, 2, 8}
    assert Nx.shape(a.future_mask) == {12, 2}

    # per game, target i -> main_x of frames i+1, i+2 (neutral 8 past the end)
    game = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 8], [8, 8]]
    mask = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    assert Nx.to_list(a.future_main_x) == game ++ game
    assert Nx.to_list(a.future_mask) == mask ++ mask
    # the a button alternates per frame: future j of target i is frame i+j's
    assert Nx.to_list(a.future_buttons[0][0]) |> hd() == 1
    assert Nx.to_list(a.future_buttons[0][1]) |> hd() == 0
  end

  defp tf(b) do
    %{
      "tf_buttons" => Nx.broadcast(0.0, {b, 8}),
      "tf_main_x" => Nx.broadcast(0, {b}),
      "tf_main_y" => Nx.broadcast(0, {b}),
      "tf_c_x" => Nx.broadcast(0, {b}),
      "tf_c_y" => Nx.broadcast(0, {b})
    }
  end
end
