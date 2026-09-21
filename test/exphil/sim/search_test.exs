defmodule ExPhil.Sim.SearchTest do
  @moduledoc "Pure parts of search-as-teacher (no sim): program sampling and label encoding."
  use ExUnit.Case, async: true

  alias ExPhil.Sim.{Drill, Search}

  test "random programs have exactly the requested horizon and only vocabulary macros" do
    :rand.seed(:exsss, {1, 2, 3})
    names = Search.macros() |> Enum.map(&elem(&1, 0)) |> MapSet.new()

    for h <- [1, 7, 90] do
      p = Search.random_program(h)
      assert length(p) == h
      assert Enum.all?(p, fn {name, %ExPhil.Bridge.ControllerState{}} -> MapSet.member?(names, name) end)
    end
  end

  test "button macros are short presses, movement macros can be held" do
    :rand.seed(:exsss, {4, 5, 6})
    runs = Search.random_program(2000) |> Enum.chunk_by(&elem(&1, 0))
    presses = runs |> Enum.filter(fn [{n, _} | _] -> n in [:jump, :nair, :fair_r, :shine] end) |> Enum.map(&length/1)
    moves = runs |> Enum.filter(fn [{n, _} | _] -> n in [:dash_r, :dash_l] end) |> Enum.map(&length/1)
    assert presses != [] and Enum.max(presses) <= 4
    assert moves != [] and Enum.max(moves) > 4
  end

  test "controller_json round-trips the fields we train on" do
    c = %{Drill.neutral() | button_a: true, main_stick: %{x: 1.0, y: 0.5}, l_shoulder: 1.0}
    j = Search.controller_json(c)
    assert j.a and j.mx == 1.0 and j.my == 0.5 and j.sh == 1.0 and not j.b
  end
end
