defmodule ExPhil.Eval.CoverageTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.Coverage

  @wait 14
  @hitstun 80

  defp p(overrides) do
    Map.merge(
      %{x: 0.0, y: 0.0, action: @wait, facing: 1, on_ground: true, stock: 4, percent: 0.0},
      overrides
    )
  end

  describe "bucket/2" do
    test "canonical close/mid/actionable/facing key" do
      # p1 at center facing right, p2 15 units to the right, both idle.
      key = Coverage.bucket(p(%{x: 0.0, facing: 1}), p(%{x: 15.0}))
      assert key == "d15-30|zcenter|aboth|p0-40|ftoward"
    end

    test "distance bands respect boundaries (exclusive upper)" do
      assert Coverage.bucket(p(%{}), p(%{x: 14.9})) |> String.starts_with?("d0-15")
      assert Coverage.bucket(p(%{}), p(%{x: 15.0})) |> String.starts_with?("d15-30")
      assert Coverage.bucket(p(%{}), p(%{x: 49.9})) |> String.starts_with?("d30-50")
      assert Coverage.bucket(p(%{}), p(%{x: 50.0})) |> String.starts_with?("d50+")
    end

    test "zone bands: center/mid/edge/off" do
      assert Coverage.bucket(p(%{x: 10.0}), p(%{x: 100.0})) =~ "zcenter"
      assert Coverage.bucket(p(%{x: 45.0}), p(%{x: 100.0})) =~ "zmid"
      assert Coverage.bucket(p(%{x: 80.0}), p(%{x: 100.0})) =~ "zedge"
      assert Coverage.bucket(p(%{x: 90.0}), p(%{x: 100.0})) =~ "zoff"
    end

    test "actionable reflects hitstun on each port" do
      assert Coverage.bucket(p(%{}), p(%{})) =~ "aboth"
      assert Coverage.bucket(p(%{action: @hitstun}), p(%{})) =~ "ap2"
      assert Coverage.bucket(p(%{}), p(%{action: @hitstun})) =~ "ap1"
      assert Coverage.bucket(p(%{action: @hitstun}), p(%{action: @hitstun})) =~ "anone"
    end

    test "facing toward vs away" do
      # p1 facing right (+1), p2 to the right -> toward.
      assert Coverage.bucket(p(%{x: 0.0, facing: 1}), p(%{x: 20.0})) =~ "ftoward"
      # p1 facing right, p2 to the LEFT -> away.
      assert Coverage.bucket(p(%{x: 0.0, facing: 1}), p(%{x: -20.0})) =~ "faway"
    end

    test "percent bands" do
      assert Coverage.bucket(p(%{percent: 39.9}), p(%{x: 20.0})) =~ "p0-40"
      assert Coverage.bucket(p(%{percent: 80.0}), p(%{x: 20.0})) =~ "p80-120"
      assert Coverage.bucket(p(%{percent: 200.0}), p(%{x: 20.0})) =~ "p120+"
    end
  end

  describe "ledger/1 & merge/2" do
    test "counts by bucket" do
      frames =
        List.duplicate(%{frame: 1, p1: p(%{x: 0.0}), p2: p(%{x: 20.0})}, 3) ++
          [%{frame: 4, p1: p(%{x: 0.0}), p2: p(%{x: 100.0})}]

      l = Coverage.ledger(frames)
      assert Enum.sum(Map.values(l)) == 4
      assert map_size(l) == 2
    end

    test "merge sums counts" do
      a = %{"k1" => 2, "k2" => 1}
      b = %{"k2" => 3, "k3" => 5}
      assert Coverage.merge(a, b) == %{"k1" => 2, "k2" => 4, "k3" => 5}
    end
  end

  describe "diff/3" do
    test "ranks under-visited buckets first" do
      # Bot lives in k1; corpus spends most of its time in k2 (which the bot
      # never visits) -> k2 must sort ahead of k1 (lower bot/corpus ratio).
      bot = %{"k1" => 100}
      corpus = %{"k1" => 10, "k2" => 90}

      [first | _] = Coverage.diff(bot, corpus)
      assert first.key == "k2"
      assert first.bot_count == 0
      assert first.ratio < 1.0
    end

    test "min_corpus_frac drops corpus noise" do
      bot = %{"k1" => 100}
      corpus = %{"k1" => 100, "rare" => 1}
      rows = Coverage.diff(bot, corpus, min_corpus_frac: 0.05)
      refute Enum.any?(rows, &(&1.key == "rare"))
    end
  end
end
