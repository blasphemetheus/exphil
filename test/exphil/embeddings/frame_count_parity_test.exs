defmodule ExPhil.Embeddings.FrameCountParityTest do
  use ExUnit.Case, async: true
  alias ExPhil.Embeddings.Game.Spatial

  test "single-player countdown counters use the batch path's nonnegative bounds" do
    player = %ExPhil.Bridge.Player{hitstun_frames_left: -2, action_frame: -1}
    assert Nx.to_flat_list(ExPhil.Embeddings.Player.embed_frame_info(player)) == [0.0, 0.0]
    player = %{player | hitstun_frames_left: 240, action_frame: 240}
    assert Nx.to_flat_list(ExPhil.Embeddings.Player.embed_frame_info(player)) == [1.0, 2.0]
  end

  test "countdown and gameplay frame normalization agree in single and batch paths" do
    states = Enum.map([-123, -18, -1, 0, 4, 28800, 40000], &%{frame: &1})
    single = Enum.map(states, &Spatial.embed_frame_count/1) |> Nx.stack()
    batch = Spatial.embed_frame_counts_batch(states)
    assert Nx.to_list(single) == Nx.to_list(batch)
    assert ExPhil.Constants.normalize_frame(-123) == 0.0
    assert ExPhil.Constants.normalize_frame(40000) == 1.0
  end
end
