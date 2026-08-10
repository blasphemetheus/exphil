defmodule ExPhil.StageCollisionTest do
  use ExUnit.Case, async: true

  alias ExPhil.StageCollision

  @legal ~w(battlefield final_destination yoshis_story fountain_of_dreams dreamland pokemon_stadium)a

  test "data loads for all six legal stages" do
    for stage <- @legal do
      assert %{vertices: _, lines: lines} = StageCollision.data(stage),
             "no collision data for #{stage}"

      assert length(lines) > 0
    end
  end

  test "accepts external ids and rejects unknowns (GOTCHA #96 semantics)" do
    assert StageCollision.data(32) == StageCollision.data(:final_destination)
    # external YS (8) must resolve to Yoshi's, not internal-FoD
    assert StageCollision.data(8) == StageCollision.data(:yoshis_story)
    assert StageCollision.data(0) == nil
    assert StageCollision.data(nil) == nil
  end

  test "FD ledge positions sit at the known edge" do
    ledges = StageCollision.ledge_positions(:final_destination)
    assert length(ledges) == 2
    for {x, y} <- ledges do
      assert_in_delta abs(x), 85.5657, 0.01
      assert_in_delta y, 0.0, 0.01
    end
  end

  test "wall distance: on an FD wall segment vs center stage" do
    # FD's walls slope INWARD below the ledge (real geometry) — test at
    # an actual wall segment's midpoint, not at x=edge
    [{x1, y1, x2, y2} | _] = StageCollision.segments(:final_destination, "wall_right")
    near = StageCollision.wall_distance(:final_destination, (x1 + x2) / 2, (y1 + y2) / 2)
    far = StageCollision.wall_distance(:final_destination, 0.0, 10.0)
    assert near < 0.01
    assert far > 60.0
  end

  test "Yoshi's slanted edges survive scaling (solid ground ends at teeter 56)" do
    # "ground" includes drop-through platforms (out to 59.5 on YS) —
    # the teeter check is over SOLID grounds, same as the extractor's
    # validation
    %{vertices: varr, lines: lines} = StageCollision.data(:yoshis_story)

    max_x =
      for l <- lines, l["class"] == "ground", not l["drop_through"] do
        [x1, _] = elem(varr, l["v1"])
        [x2, _] = elem(varr, l["v2"])
        max(abs(x1), abs(x2))
      end
      |> Enum.max()

    assert_in_delta max_x, 56.0, 0.05
  end

  test "point_segment_distance basics" do
    assert_in_delta StageCollision.point_segment_distance(0, 5, {-10, 0, 10, 0}), 5.0, 1.0e-9
    assert_in_delta StageCollision.point_segment_distance(20, 0, {-10, 0, 10, 0}), 10.0, 1.0e-9
  end
end
