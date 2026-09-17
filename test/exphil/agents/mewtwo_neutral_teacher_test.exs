defmodule ExPhil.Agents.MewtwoNeutralTeacherTest do
  use ExUnit.Case, async: true
  alias ExPhil.Agents.MewtwoNeutralTeacher, as: Teacher
  alias Melee.PlayerState

  defp player(x, opts \\ []),
    do:
      struct!(
        %PlayerState{
          character: 16,
          action: 14,
          on_ground: true,
          position: %{x: x, y: 0.0},
          facing: true
        },
        opts
      )

  test "shield creates a grab opportunity, but airborne and behind do not" do
    me = player(-10.0)
    shield = player(0.0, action: 179)
    assert Teacher.choose(me, shield) == :grab
    refute Teacher.choose(me, %{shield | on_ground: false, position: %{x: 0.0, y: 20.0}}) == :grab
    assert Teacher.choose(%{me | facing: false}, shield) == {:walk, :right}
    assert Teacher.choose(player(-40.0), shield) == {:run, :right}
    refute Teacher.choose(player(-60.0), shield) == {:run, :right}
  end

  test "facing and height constrain down tilt; dash must brake first" do
    fox = player(0.0)
    assert Teacher.choose(player(-16.0), fox) == :down_tilt
    assert Teacher.choose(player(16.0, facing: false), fox) == :down_tilt
    refute Teacher.choose(player(16.0), fox) == :down_tilt
    refute Teacher.choose(player(-16.0), %{fox | on_ground: false}) == :down_tilt
    assert Teacher.choose(player(-16.0, action: 20), fox) == :brake
  end

  test "active attack retreats only with room for the slide" do
    fox = player(0.0, character: 1, action: 44, action_frame: 2)
    assert Teacher.choose(player(-16.0), fox) == :shield
    assert Teacher.choose(player(-50.0), %{fox | position: %{x: -27.0, y: 0.0}}) == :shield
    refute Teacher.choose(player(-70.0), fox) == {:wavedash, :left}
  end

  test "damage discards a committed routine and releases inputs" do
    {teacher, _} = Teacher.step(Teacher.new(), player(-9.0), player(0.0))
    assert teacher.tech != nil
    {teacher, commands} = Teacher.step(teacher, player(-9.0, action: 75), player(0.0))
    assert teacher.tech == nil
    assert commands == [:release_all]
    {teacher, _} = Teacher.step(teacher, player(-16.0), player(0.0))
    assert teacher.choice == :down_tilt
  end

  test "high and low captures and throws cancel routines and prevent combo targeting" do
    {committed, _} = Teacher.step(Teacher.new(), player(-9.0), player(0.0))

    for action <- Enum.to_list(223..232) ++ Enum.to_list(239..243) do
      {teacher, commands} = Teacher.step(committed, player(-9.0, action: action), player(0.0))
      assert teacher.tech == nil
      assert commands == [:release_all]
      assert Teacher.choose(player(-16.0), player(0.0, action: action)) == :wait
    end
  end
end
