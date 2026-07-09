defmodule ExPhil.Agents.Dummies.TechRandomTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Dummies.TechRandom

  defp player(overrides) do
    Map.merge(
      %{action: 14.0, y: 50.0, speed_y_self: 0.0, speed_y_attack: 0.0},
      Map.new(overrides)
    )
  end

  test "neutral input and plan reset outside the knockdown lifecycle" do
    dirty = %TechRandom{plan: {:tech, :left}, press_frames: 2}
    {input, state} = TechRandom.step(player(action: 14.0), dirty)

    assert input == TechRandom.neutral()
    assert state.plan == nil
  end

  test "nil player (dead/missing) is neutral" do
    assert {input, %TechRandom{plan: nil}} = TechRandom.step(nil, TechRandom.new())
    assert input == TechRandom.neutral()
  end

  test "armed tech plan holds fire while high up" do
    s = %TechRandom{plan: {:tech, :in_place}, press_frames: 2}
    p = player(action: 85.0, y: 60.0, speed_y_attack: -3.0)

    {input, s2} = TechRandom.step(p, s)
    assert input == TechRandom.neutral()
    assert s2.press_frames == 2
  end

  test "tech press fires falling near the ground, for at most 2 frames" do
    s = %TechRandom{plan: {:tech, :left}, press_frames: 2}
    p = player(action: 85.0, y: 6.0, speed_y_attack: -3.0)

    {i1, s} = TechRandom.step(p, s)
    assert i1.buttons.r
    assert i1.main_stick.x < 0.1

    {i2, s} = TechRandom.step(p, s)
    assert i2.buttons.r

    {i3, _s} = TechRandom.step(p, s)
    refute i3.buttons.r, "press must be an edge, not a hold"
  end

  test "miss plan never presses" do
    s = %TechRandom{plan: :miss, press_frames: 2}
    p = player(action: 38.0, y: 4.0, speed_y_self: -2.0)

    {input, _} = TechRandom.step(p, s)
    refute input.buttons.r
  end

  test "downed: waits out the delay then issues one getup input" do
    s = %TechRandom{plan: {:getup, :roll_right}, delay: 2, press_frames: 2}
    # 184 = DownWaitU (actionable, face up)
    p = player(action: 184.0, y: 0.0)

    {i1, s} = TechRandom.step(p, s)
    assert i1 == TechRandom.neutral()
    {i2, s} = TechRandom.step(p, s)
    assert i2 == TechRandom.neutral()

    {i3, s} = TechRandom.step(p, s)
    assert i3.main_stick.x > 0.9

    {_i4, s} = TechRandom.step(p, s)
    {i5, _s} = TechRandom.step(p, s)
    assert i5 == TechRandom.neutral(), "getup input is a tap, not a hold"
  end

  test "plans are rolled once per knockdown and vary across seeds" do
    plans =
      for seed <- 1..30 do
        :rand.seed(:exsss, {seed, seed, seed})
        {_input, s} = TechRandom.step(player(action: 85.0, y: 60.0), TechRandom.new())
        s.plan
      end

    assert Enum.any?(plans, &match?({:tech, _}, &1))
    assert Enum.any?(plans, &(&1 == :miss))
    assert plans |> Enum.filter(&match?({:tech, _}, &1)) |> Enum.uniq() |> length() > 1
  end
end
