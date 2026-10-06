defmodule ExPhil.Networks.Policy.ButtonEventsTest do
  @moduledoc """
  Press/release event button head (2026-10-02): 16 raw logits collapse to 8
  "down this frame" logits, selected per button by the previous state.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy.{Heads, Sampling}

  test "up buttons take the press logit, held buttons take minus the release logit" do
    raw = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])
    prev = Nx.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])

    assert Nx.to_flat_list(Heads.collapse_button_events(raw, prev)) ==
             [1.0, -20.0, 3.0, -40.0, 5.0, 6.0, -70.0, 8.0]
  end

  test "a {1, 8} previous state broadcasts over tiled rows" do
    raw = Nx.broadcast(Nx.tensor(1.0), {3, 16})
    prev = Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = Heads.collapse_button_events(raw, prev)

    assert Nx.shape(out) == {3, 8}
    assert out |> Nx.to_list() |> Enum.uniq() == [[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
  end

  describe "hold-or-change sticks" do
    test "probabilities sum to one and the hold mass lands on the previous bucket" do
      raw = Nx.tensor([[0.3, -1.0, 2.0, 0.5, 1.5], [0.0, 0.0, 0.0, 0.0, -2.0]])
      prev = Nx.tensor([2, 0])
      p = Heads.collapse_hold_change(raw, prev) |> Nx.exp()

      for row <- Nx.to_list(p), do: assert_in_delta(Enum.sum(row), 1.0, 1.0e-5)

      # row 0: h = sigmoid(1.5); bucket 2 gets h + (1 - h) * softmax(change)[2]
      h = 1 / (1 + :math.exp(-1.5))
      z = Enum.sum(Enum.map([0.3, -1.0, 2.0, 0.5], &:math.exp/1))
      assert_in_delta Nx.to_number(p[0][2]), h + (1 - h) * :math.exp(2.0) / z, 1.0e-5
      assert_in_delta Nx.to_number(p[0][0]), (1 - h) * :math.exp(0.3) / z, 1.0e-5
    end

    test "a confident hold keeps the previous bucket; a confident change ignores it" do
      change = [0.0, 5.0, 0.0, 0.0]
      hold = Heads.collapse_hold_change(Nx.tensor([change ++ [20.0]]), Nx.tensor([3]))
      move = Heads.collapse_hold_change(Nx.tensor([change ++ [-20.0]]), Nx.tensor([3]))

      assert Nx.to_number(Nx.argmax(hold, axis: -1)[0]) == 3
      assert Nx.to_number(Nx.argmax(move, axis: -1)[0]) == 1
    end

    test "a {1} previous bucket broadcasts over tiled rows" do
      out = Heads.collapse_hold_change(Nx.broadcast(Nx.tensor(0.0), {3, 5}), Nx.tensor([1]))
      assert Nx.shape(out) == {3, 4}
    end
  end

  describe "hold / release / change sticks (2026-10-05)" do
    # K = 5 buckets, centre = 2; raw = 5 change logits + hold + release
    test "probabilities sum to one; hold mass at prev, release mass at centre" do
      change = [0.3, -1.0, 2.0, 0.5, -0.5]
      raw = Nx.tensor([change ++ [1.5, 0.8]])
      p = Heads.collapse_hold_release_change(raw, Nx.tensor([4])) |> Nx.exp()
      [row] = Nx.to_list(p)
      assert_in_delta Enum.sum(row), 1.0, 1.0e-5

      h = 1 / (1 + :math.exp(-1.5))
      r = 1 / (1 + :math.exp(-0.8))
      z = Enum.sum(Enum.map(change, &:math.exp/1))
      sm = fn i -> :math.exp(Enum.at(change, i)) / z end
      assert_in_delta Enum.at(row, 4), h + (1 - h) * (1 - r) * sm.(4), 1.0e-5
      assert_in_delta Enum.at(row, 2), (1 - h) * r + (1 - h) * (1 - r) * sm.(2), 1.0e-5
      assert_in_delta Enum.at(row, 0), (1 - h) * (1 - r) * sm.(0), 1.0e-5
    end

    test "prev == centre merges hold and release on the centre bucket" do
      change = [0.0, 0.0, 0.0, 0.0, 0.0]
      p = Heads.collapse_hold_release_change(Nx.tensor([change ++ [0.0, 0.0]]), Nx.tensor([2])) |> Nx.exp()
      [row] = Nx.to_list(p)
      assert_in_delta Enum.sum(row), 1.0, 1.0e-5
      # h = r = 0.5: centre gets 0.5 + 0.25 + 0.25 * 0.2
      assert_in_delta Enum.at(row, 2), 0.5 + 0.25 + 0.25 * 0.2, 1.0e-5
    end

    test "a confident release goes to centre regardless of the change logits" do
      change = [0.0, 5.0, 0.0, 0.0, 0.0]
      out = Heads.collapse_hold_release_change(Nx.tensor([change ++ [-20.0, 20.0]]), Nx.tensor([3]))
      assert Nx.to_number(Nx.argmax(out, axis: -1)[0]) == 2
      assert Nx.shape(Heads.collapse_hold_release_change(Nx.broadcast(Nx.tensor(0.0), {3, 7}), Nx.tensor([1]))) == {3, 5}
    end
  end

  describe "sampler" do
    # Minimal AR head: hidden 4, residual 4, 16 button logits. Zero weights
    # with a bias make the raw logits a known constant.
    defp head_params(button_bias, stick_extra \\ 0, stick_bias \\ nil) do
      dense = fn i, o, bias -> %{"kernel" => Nx.broadcast(0.0, {i, o}), "bias" => bias || Nx.broadcast(0.0, {o})} end
      embed = fn v -> %{"kernel" => Nx.broadcast(0.0, {v, 4})} end

      cats =
        for {name, size} <- [{"main_x", 17}, {"main_y", 17}, {"c_x", 17}, {"c_y", 17}, {"shoulder", 5}], into: %{} do
          {name, size}
        end

      base = %{
        "ar_residual_proj" => dense.(4, 4, nil),
        "ar_buttons_hidden" => dense.(4, 4, nil),
        "ar_buttons_logits" => dense.(4, 16, button_bias),
        "ar_buttons_embed" => %{"kernel" => Nx.broadcast(0.0, {8, 4})}
      }

      Enum.reduce(cats, base, fn {name, size}, acc ->
        acc
        |> Map.put("ar_#{name}_hidden", dense.(4, 4, nil))
        |> Map.put("ar_#{name}_logits", if(name == "shoulder", do: dense.(4, size, nil), else: dense.(4, size + stick_extra, stick_bias)))
        |> Map.put("ar_#{name}_embed", embed.(size))
      end)
    end

    test "deterministic decode holds what is down and presses nothing new" do
      # press logits -5 (never press), release logits -5 (never release)
      bias = Nx.broadcast(-5.0, {16})
      prev = Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

      out =
        Sampling.sample_autoregressive_from_features(head_params(bias), Nx.broadcast(0.0, {1, 4}),
          deterministic: true,
          event_prev_buttons: prev
        )

      assert Nx.to_flat_list(out.buttons) == [1, 0, 0, 0, 0, 0, 1, 0]
      assert Nx.shape(out.logits.buttons) == {1, 8}
    end

    test "stick events: a confident hold reproduces the previous buckets" do
      # 17 change logits at 0 + hold logit +20 on every stick axis
      stick_bias = Nx.concatenate([Nx.broadcast(0.0, {17}), Nx.tensor([20.0])])
      params = head_params(Nx.broadcast(-5.0, {8}) |> then(&Nx.concatenate([&1, &1])), 1, stick_bias)

      out =
        Sampling.sample_autoregressive_from_features(params, Nx.broadcast(0.0, {1, 4}),
          event_prev_buttons: Nx.broadcast(0.0, {1, 8}),
          event_prev_sticks: Nx.tensor([[3, 12, 8, 0]])
        )

      assert Enum.map([:main_x, :main_y, :c_x, :c_y], &Nx.to_number(Nx.squeeze(out[&1]))) == [3, 12, 8, 0]
      assert Nx.shape(out.logits.main_x) == {1, 17}
    end

    test "an event head without the previous buttons is a clear error" do
      assert_raise ArgumentError, ~r/event_prev_buttons/, fn ->
        Sampling.sample_autoregressive_from_features(head_params(nil), Nx.broadcast(0.0, {1, 4}), deterministic: true)
      end
    end
  end
end
