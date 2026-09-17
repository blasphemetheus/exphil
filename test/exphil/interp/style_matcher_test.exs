defmodule ExPhil.Interp.StyleMatcherTest do
  use ExUnit.Case, async: true
  alias ExPhil.Interp.StyleMatcher

  defp rows(tag, base, costume, n, seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    for g <- 1..n, do: %{tag: tag, path: "#{tag}_#{g}", port: 1, costume: costume, vec: Enum.map(base, &(&1 + (:rand.uniform() - 0.5) * 0.2))}
  end

  setup do
    by_tag = %{
      "A" => rows("A", [0.0, 0.0, 0.0], 1, 10, 1),
      "B" => rows("B", [0.15, 0.0, 0.0], 2, 10, 2),
      "C" => rows("C", [0.0, 2.0, 0.0], 3, 4, 3)
    }

    %{by_tag: by_tag, gallery: StyleMatcher.gallery(by_tag, costume_slots: 4)}
  end

  test "posterior sums to one and prefers the nearest entity", %{gallery: g} do
    post = StyleMatcher.posterior(%{vec: [0.0, 1.95, 0.0], costume: 3, path: "q", port: 1}, g)
    assert_in_delta Enum.sum(Enum.map(post, &elem(&1, 1))), 1.0, 1.0e-9
    assert {"C", _} = hd(post)
  end

  test "costume is evidence, not a filter: an off-colour game can still match on style", %{gallery: g} do
    # style says A, costume says B
    # a query between A and B on style, with B's costume
    post = StyleMatcher.posterior(%{vec: [0.075, 0.0, 0.0], costume: 2, path: "q", port: 1}, g)
    assert {"B", p_b} = hd(post)
    assert p_b < 1.0
    # same style point with A's costume flips toward A
    [{best, p_a} | _] = StyleMatcher.posterior(%{vec: [0.075, 0.0, 0.0], costume: 1, path: "q", port: 1}, g)
    assert best == "A" and p_a > 0.5 and p_a < 1.0
  end

  test "leave-one-out evaluation is accurate and coverage falls as the threshold rises", %{by_tag: by_tag, gallery: g} do
    r = StyleMatcher.evaluate(by_tag, g)
    assert r.queries == 24
    assert r.top1 >= 0.9
    assert r.at_threshold[0.5].coverage >= r.at_threshold[0.95].coverage
    assert r.at_threshold[0.95].accuracy >= 0.95
  end

  test "assign returns only games clearing the threshold", %{gallery: g} do
    q = [%{vec: [0.0, 0.0, 0.0], costume: 1, path: "near_A", port: 1}, %{vec: [1.0, 1.0, 5.0], costume: 0, path: "nowhere", port: 1}]
    assigned = StyleMatcher.assign(q, g, 0.5)
    assert Enum.map(assigned, & &1.tag) == ["A"]
    assert {"?", _} = hd(StyleMatcher.posterior(Enum.at(q, 1), g))
  end
end
