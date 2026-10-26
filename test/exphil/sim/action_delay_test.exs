defmodule ExPhil.Sim.ActionDelayTest do
  use ExUnit.Case, async: true
  alias ExPhil.Sim.ActionDelay

  test "a frame-tagged decision is applied exactly d calls later" do
    for delay <- [0, 1, 3, 4] do
      queue = ActionDelay.new(delay, 2, :neutral)

      {applied, _} =
        Enum.map_reduce(0..12, queue, fn frame, queue ->
          ActionDelay.push(queue, [{frame, 0}, {frame, 1}])
        end)

      for {row, frame} <- Enum.with_index(applied) do
        assert row ==
                 if(frame < delay,
                   do: [:neutral, :neutral],
                   else: [{frame - delay, 0}, {frame - delay, 1}]
                 )
      end
    end
  end

  test "partial resets clear stale actions without delaying surviving rows" do
    queue = ActionDelay.new(3, 2, :neutral)
    {_, queue} = ActionDelay.push(queue, [:a0, :b0])
    {_, queue} = ActionDelay.push(queue, [:a1, :b1])
    {_, queue} = ActionDelay.push(queue, [:a2, :b2])
    queue = ActionDelay.reset_rows(queue, [0])
    assert {[:neutral, :b0], queue} = ActionDelay.push(queue, [:new0, :b3])
    assert {[:neutral, :b1], queue} = ActionDelay.push(queue, [:new1, :b4])
    assert {[:neutral, :b2], queue} = ActionDelay.push(queue, [:new2, :b5])
    assert {[:new0, :b3], _} = ActionDelay.push(queue, [:new3, :b6])
  end
end
