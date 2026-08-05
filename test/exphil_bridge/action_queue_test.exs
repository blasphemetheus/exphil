defmodule ExPhil.Bridge.ActionQueueTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ActionQueue

  test "actions fire exactly at their scheduled frame, not before" do
    q = ActionQueue.new() |> ActionQueue.schedule(105, :act)

    {due, q} = ActionQueue.pop_due(q, 104)
    assert due == []

    {due, q} = ActionQueue.pop_due(q, 105)
    assert due == [:act]

    {due, _q} = ActionQueue.pop_due(q, 106)
    assert due == []
  end

  test "late polls deliver everything due, oldest frame first" do
    q =
      ActionQueue.new()
      |> ActionQueue.schedule(103, :b)
      |> ActionQueue.schedule(101, :a)
      |> ActionQueue.schedule(103, :c)

    {due, _q} = ActionQueue.pop_due(q, 110)
    assert due == [:a, :b, :c]
  end

  test "frame regression (new game) drops the whole queue" do
    q = ActionQueue.new() |> ActionQueue.schedule(500, :stale)
    {[], q} = ActionQueue.pop_due(q, 400)

    # New game: frame restarts at -123. The stale action must not fire.
    {due, q} = ActionQueue.pop_due(q, -123)
    assert due == []
    assert ActionQueue.size(q) == 0

    {due, _q} = ActionQueue.pop_due(q, 600)
    assert due == []
  end

  test "size counts all pending items" do
    q =
      ActionQueue.new()
      |> ActionQueue.schedule(1, :a)
      |> ActionQueue.schedule(1, :b)
      |> ActionQueue.schedule(2, :c)

    assert ActionQueue.size(q) == 3
  end
end
