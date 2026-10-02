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

  describe "sampler" do
    # Minimal AR head: hidden 4, residual 4, 16 button logits. Zero weights
    # with a bias make the raw logits a known constant.
    defp head_params(button_bias) do
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
        |> Map.put("ar_#{name}_logits", dense.(4, size, nil))
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

    test "an event head without the previous buttons is a clear error" do
      assert_raise ArgumentError, ~r/event_prev_buttons/, fn ->
        Sampling.sample_autoregressive_from_features(head_params(nil), Nx.broadcast(0.0, {1, 4}), deterministic: true)
      end
    end
  end
end
