defmodule ExPhil.Interp.StyleCalibrationTest do
  use ExUnit.Case, async: true
  alias ExPhil.Interp.{StyleCalibration, StyleFingerprint}

  # Synthetic players: each is a base fingerprint plus small per-game noise,
  # far enough apart that retrieval must succeed and close enough that a
  # degenerate (all-zero) distance would fail.
  defp player(seed, games) do
    :rand.seed(:exsss, {seed, seed, seed})
    base = Map.new(StyleFingerprint.keys(), fn k -> {k, :rand.uniform() * 10} end)

    for g <- 1..games do
      feats = Map.new(base, fn {k, v} -> {k, v + (:rand.uniform() - 0.5) * 0.4} end)
      %{tag: "P#{seed}", path: "p#{seed}_#{g}", port: 1, corpus: "syn", features: feats}
    end
  end

  setup do
    rows = Enum.flat_map(1..6, &player(&1, 6)) ++ [%{tag: nil, path: "anon", port: 1, corpus: "syn", features: %{}}]
    %{rows: rows}
  end

  test "tagged/2 drops anonymous rows and thin tags", %{rows: rows} do
    by_tag = StyleCalibration.tagged(rows ++ [%{tag: "THIN", path: "t", port: 1, corpus: "syn", features: %{}}], 3)
    assert Map.keys(by_tag) |> Enum.sort() == Enum.map(1..6, &"P#{&1}")
  end

  test "same-player pairs are closer than different-player pairs", %{rows: rows} do
    by_tag = StyleCalibration.tagged(rows)
    d = StyleCalibration.pair_distributions(by_tag, max_pairs: 500)
    assert StyleCalibration.summary(d.same).p90 < StyleCalibration.summary(d.different).p10
    t = StyleCalibration.threshold_at_fmr(d, 0.01)
    assert t.same_accepted == 1.0
  end

  test "leave-one-out retrieval is perfect on separable players and never self-votes", %{rows: rows} do
    by_tag = StyleCalibration.tagged(rows)
    r = StyleCalibration.retrieval(by_tag)
    assert r.queries == 36 and r.tags == 6 and r.gallery_size == 6
    assert r.top1 == 1.0
  end

  test "cross-corpus retrieval uses the gallery's centroids and only shared tags", %{rows: rows} do
    by_tag = StyleCalibration.tagged(rows)
    gallery = by_tag |> Map.take(["P1", "P2", "P3"]) |> Map.new(fn {t, rs} -> {t, Enum.map(rs, &%{&1 | corpus: "other"})} end)
    r = StyleCalibration.retrieval(by_tag, gallery: gallery)
    assert r.tags == 3 and r.queries == 18 and r.gallery_size == 3
    assert r.top1 == 1.0
  end

  test "invariant metric ignores option-rate features" do
    a = %{features: Map.new(StyleFingerprint.keys(), fn k -> {k, 1.0} end)}
    b = %{a | features: Map.put(a.features, :dashdance_per_min, 50.0)}
    assert StyleFingerprint.invariant_distance(a.features, b.features) == 0.0
    assert StyleFingerprint.distance(a.features, b.features) > 0.0
  end
end
