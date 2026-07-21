defmodule ExPhil.Training.ProbeRegularizerTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.ProbeRegularizer

  @wait 14
  @guard 179

  defp frame(action) do
    %{game_state: %{players: %{1 => %{action: action, x: 0.0}, 2 => %{action: @wait, x: 10.0}}}}
  end

  describe "frame_labels/1" do
    test "labels the shield family (178..182) as 1, everything else 0" do
      frames = [frame(@wait), frame(178), frame(@guard), frame(182), frame(183), frame(75)]
      assert ProbeRegularizer.frame_labels(frames) == [0, 1, 1, 1, 0, 0]
    end

    test "a missing player labels 0" do
      f = %{game_state: %{players: %{1 => nil, 2 => %{action: @wait}}}}
      assert ProbeRegularizer.frame_labels([f]) == [0]
    end
  end

  describe "fit_direction/3" do
    test "recovers the class-mean contrast as a unit vector" do
      # 100 shield rows at [2,0,0,0], 100 free rows at the origin
      pos = Nx.broadcast(Nx.tensor([2.0, 0.0, 0.0, 0.0]), {100, 4})
      neg = Nx.broadcast(Nx.tensor([0.0, 0.0, 0.0, 0.0]), {100, 4})
      h = Nx.concatenate([pos, neg])
      labels = List.duplicate(1, 100) ++ List.duplicate(0, 100)

      assert {:ok, v} = ProbeRegularizer.fit_direction(h, labels)
      assert Nx.to_flat_list(v) == [1.0, 0.0, 0.0, 0.0]
    end

    test "refuses when a class is under min_class" do
      h = Nx.broadcast(1.0, {100, 4})
      labels = List.duplicate(1, 10) ++ List.duplicate(0, 90)

      assert {:error, {:insufficient, 10, 90}} = ProbeRegularizer.fit_direction(h, labels)
      # And accepts the same split when the floor is lowered
      assert {:error, :zero_contrast} =
               ProbeRegularizer.fit_direction(h, labels, min_class: 10)
    end

    test "identical class means are a zero contrast, not a NaN direction" do
      h = Nx.broadcast(3.5, {200, 8})
      labels = List.duplicate(1, 100) ++ List.duplicate(0, 100)

      assert {:error, :zero_contrast} = ProbeRegularizer.fit_direction(h, labels)
    end
  end

  describe "alignment_penalty/2" do
    test "mean squared projection along v" do
      h = Nx.tensor([[2.0, 0.0], [0.0, 3.0]])
      v = Nx.tensor([1.0, 0.0])

      assert ProbeRegularizer.alignment_penalty(h, v) |> Nx.to_number() == 2.0
    end

    test "zero direction is exactly inert" do
      h = Nx.tensor([[5.0, -7.0], [1.0, 2.0]])
      v = Nx.tensor([0.0, 0.0])

      assert ProbeRegularizer.alignment_penalty(h, v) |> Nx.to_number() == 0.0
    end

    test "orthogonal activations contribute nothing" do
      h = Nx.tensor([[0.0, 4.0], [0.0, -4.0]])
      v = Nx.tensor([1.0, 0.0])

      assert ProbeRegularizer.alignment_penalty(h, v) |> Nx.to_number() == 0.0
    end
  end
end
