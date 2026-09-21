defmodule ExPhil.Sim.DrillTest do
  @moduledoc "Pure parts of the curriculum env (no sim, no policy): aggregation and pool serialization."
  use ExUnit.Case, async: true

  alias ExPhil.Sim.Drill

  defp result(contact?, converted?, damage, outcomes, aerials) do
    %{
      contact?: contact?,
      converted?: converted?,
      damage: damage,
      fair: %{outcomes: outcomes},
      chain: %{mean_connected_aerials: aerials}
    }
  end

  test "aggregate reports rates over starts and conversion given contact" do
    rs = [
      result(true, true, 30.0, %{string_hit: 1}, 2.0),
      result(true, false, 12.0, %{escaped: 1}, 1.0),
      result(false, false, 0.0, %{}, 0.0),
      result(false, false, 4.0, %{}, 0.0)
    ]

    a = Drill.aggregate(rs)
    assert a.starts == 4
    assert a.contact_rate == 0.5
    assert a.conversion_rate == 0.25
    assert a.conversion_given_contact == 0.5
    assert a.mean_damage == 11.5
    assert a.mean_connected_aerials == 0.75
    assert a.outcome_kinds == %{string_hit: 1, escaped: 1}
  end

  test "aggregate on no results is all zeros" do
    a = Drill.aggregate([])
    assert a.starts == 0 and a.contact_rate == 0.0 and a.conversion_rate == 0.0 and a.conversion_given_contact == 0.0
  end

  test "pool_to_disk writes one JSON row per entry with the blob base64-encoded and no history" do
    dir = System.tmp_dir!() |> Path.join("drill_test_#{System.unique_integer([:positive])}")
    path = Path.join(dir, "pool.jsonl")
    entry = %{id: 7, blob: <<1, 2, 3>>, frame: 120, summary: %{p1: %{x: 1.0}, p2: %{x: -1.0}}, history: [:not_serialized]}
    :ok = Drill.pool_to_disk([entry, %{entry | id: 8}], path)
    rows = path |> File.read!() |> String.split("\n", trim: true) |> Enum.map(&Jason.decode!/1)
    assert Enum.map(rows, & &1["id"]) == [7, 8]
    assert Base.decode64!(hd(rows)["blob"]) == <<1, 2, 3>>
    refute Map.has_key?(hd(rows), "history")
    File.rm_rf!(dir)
  end

  test "neutral controller is 0.5-centered (GOTCHA #123)" do
    n = Drill.neutral()
    assert n.main_stick == %{x: 0.5, y: 0.5} and n.c_stick == %{x: 0.5, y: 0.5}
    refute n.button_a or n.button_y
  end
end
