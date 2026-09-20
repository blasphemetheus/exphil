defmodule ExPhil.Eval.AerialChainTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.AerialChain

  # slim frames: p1 attacker, p2 defender
  defp f(i, a1, a2, opts \\ []) do
    %{frame: i, p1: %{action: a1, percent: 0.0, stock: 4, x: 0.0, y: 0.0, facing: 1, on_ground: true},
      p2: %{action: a2, percent: 0.0, stock: Keyword.get(opts, :stock, 4), x: 10.0, y: 0.0, facing: -1, on_ground: true}}
  end

  defp neutral(n, from \\ 0), do: for(i <- from..(from + n - 1), do: f(i, 14, 14))

  test "two aerials before the defender is actionable = 2 connected aerials" do
    frames =
      neutral(70) ++
        [f(70, 66, 75), f(71, 66, 75), f(72, 66, 75)] ++          # fair hit, hitstun
        [f(73, 66, 76), f(74, 66, 76)] ++                          # second hit edge? no: same hitstun family, needs exit first
        [f(75, 29, 42), f(76, 65, 42)] ++                          # defender LANDING (not actionable), attacker nair
        [f(77, 65, 84), f(78, 65, 84)] ++                          # nair connects (hit edge from landing)
        [f(79, 29, 14)]                                            # defender stands: actionable -> chain ends

    [o] = AerialChain.openings(frames)
    assert o.frame == 70 and o.hits == 2 and o.connected_aerials == 2 and o.end_reason == :actionable_gap
    assert AerialChain.summary(frames).mean_connected_aerials == 2.0
  end

  test "a hit after an actionable frame is a string, not connected" do
    frames = neutral(70) ++ [f(70, 66, 75), f(71, 66, 75), f(72, 29, 14), f(73, 65, 84), f(74, 29, 14)]
    [o] = AerialChain.openings(frames)
    assert o.connected_aerials == 1 and o.end_reason == :actionable_gap and o.hits == 1
  end

  test "a grounded follow-up counts as a hit but not an aerial" do
    frames = neutral(70) ++ [f(70, 66, 75), f(71, 66, 75), f(72, 14, 42), f(73, 44, 84), f(74, 44, 84), f(75, 14, 14)]
    [o] = AerialChain.openings(frames)
    assert o.hits == 2 and o.aerials == 1 and o.connected_aerials == 1
  end

  test "openings require a neutral lead-in; combo hits are not new openings" do
    frames = neutral(70) ++ [f(70, 66, 75), f(71, 66, 75), f(72, 29, 42), f(73, 65, 84), f(74, 29, 14)]
    assert length(AerialChain.openings(frames)) == 1
    assert AerialChain.openings(neutral(10) ++ [f(10, 66, 75)]) == []
  end

  test "a stock change ends the chain" do
    frames = neutral(70) ++ [f(70, 66, 75), f(71, 66, 87), f(72, 29, 87, stock: 3)]
    [o] = AerialChain.openings(frames)
    assert o.end_reason == :stock
  end
end
