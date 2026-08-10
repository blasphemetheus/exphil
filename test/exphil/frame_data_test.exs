defmodule ExPhil.FrameDataTest do
  use ExUnit.Case, async: true

  alias ExPhil.FrameData

  test "fox nair matches community-known values" do
    assert FrameData.hit_windows(:fox, :nair) == [{4, 7}, {8, 31}]
    assert %{"landingLag" => 15, "iasa" => 42} = FrameData.move(:fox, :nair)
  end

  test "hitbox_out?/3 gates on action_frame" do
    assert FrameData.hitbox_out?(:fox, :nair, 5) == true
    assert FrameData.hitbox_out?(:fox, :nair, 31) == true
    assert FrameData.hitbox_out?(:fox, :nair, 3) == false
    assert FrameData.hitbox_out?(:fox, :nair, 40) == false
  end

  test "unknown moves and characters return nil (not false)" do
    assert FrameData.hitbox_out?(:fox, :firefox, 5) == nil
    assert FrameData.data(:master_hand_unknown) == nil
    assert FrameData.data(nil) == nil
  end

  test "string names normalize (viewer/export path)" do
    assert FrameData.data("captain_falcon") != nil
    assert FrameData.data("fox") != nil
  end

  test "every mapped fighter has data with universal moves" do
    for char <- [:fox, :falco, :marth, :sheik, :captain_falcon, :peach, :jigglypuff, :mewtwo] do
      moves = FrameData.data(char)
      assert moves != nil, "no data for #{char}"
      assert Map.has_key?(moves, "nair"), "#{char} missing nair"
      assert Map.has_key?(moves, "grab"), "#{char} missing grab"
    end
  end
end
