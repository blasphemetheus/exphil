defmodule ExPhil.Bridge.SimRowsTest do
  use ExUnit.Case, async: true
  alias ExPhil.Bridge.SimRows

  @player %{"kind" => "struct", "itemsize" => 28, "fields" => [
    %{"name" => "buttons", "offset" => 0, "kind" => "struct", "itemsize" => 8, "fields" =>
      Enum.with_index(~w(A B X Y Z L R D_UP)) |> Enum.map(fn {n, i} -> %{"name" => n, "offset" => i, "kind" => "scalar", "fmt" => "|u1", "itemsize" => 1} end)},
    %{"name" => "main_stick_x", "offset" => 8, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4},
    %{"name" => "main_stick_y", "offset" => 12, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4},
    %{"name" => "c_stick_x", "offset" => 16, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4},
    %{"name" => "c_stick_y", "offset" => 20, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4},
    %{"name" => "shoulder", "offset" => 24, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4}
  ]}
  @input %{"kind" => "struct", "itemsize" => 112, "fields" => [%{"name" => "players", "offset" => 0, "kind" => "array", "shape" => [4], "item" => @player, "itemsize" => 112}]}
  @mini %{"kind" => "struct", "itemsize" => 16, "fields" => [
    %{"name" => "frame_id", "offset" => 0, "kind" => "scalar", "fmt" => "<i4", "itemsize" => 4},
    %{"name" => "pct", "offset" => 4, "kind" => "scalar", "fmt" => "<f4", "itemsize" => 4},
    %{"name" => "action", "offset" => 8, "kind" => "scalar", "fmt" => "<u2", "itemsize" => 2},
    %{"name" => "af", "offset" => 10, "kind" => "scalar", "fmt" => "<i2", "itemsize" => 2},
    %{"name" => "facing", "offset" => 12, "kind" => "scalar", "fmt" => "|u1", "itemsize" => 1},
    %{"name" => "team", "offset" => 13, "kind" => "scalar", "fmt" => "|i1", "itemsize" => 1},
    %{"name" => "_pad0", "offset" => 14, "kind" => "scalar", "fmt" => "<u2", "itemsize" => 2}
  ]}

  test "decodes little-endian scalars by offset and skips padding" do
    bin = <<-123::signed-32-little, 41.5::float-32-little, 322::unsigned-16-little, -1::signed-16-little, 1, -1::signed-8, 0, 0>>
    assert SimRows.decode(@mini, bin) == %{"frame_id" => -123, "pct" => 41.5, "action" => 322, "af" => -1, "facing" => 1, "team" => -1}
  end

  test "decode_rows splits consecutive rows" do
    row = <<7::signed-32-little, 0.0::float-32-little, 14::unsigned-16-little, 0::signed-16-little, 0, 0, 0, 0>>
    assert [%{"frame_id" => 7}, %{"frame_id" => 7}] = SimRows.decode_rows(@mini, row <> row, 2)
  end

  test "encodes controller input rows (atom keys, missing players zeroed) and decodes back" do
    p0 = %{buttons: %{A: 1, Y: 1}, main_stick_x: 1.0, main_stick_y: 0.5, c_stick_x: 0.5, c_stick_y: 0.5, shoulder: 0.0}
    bin = SimRows.encode(@input, %{"players" => [p0, nil]})
    assert byte_size(bin) == 112
    [d0, d1 | _] = SimRows.decode(@input, bin)["players"]
    assert d0["buttons"]["A"] == 1 and d0["buttons"]["Y"] == 1 and d0["buttons"]["B"] == 0
    assert d0["main_stick_x"] == 1.0 and d0["main_stick_y"] == 0.5
    assert d1["main_stick_x"] == 0.0 and d1["buttons"]["A"] == 0
  end
end
