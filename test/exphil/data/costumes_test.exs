defmodule ExPhil.Data.CostumesTest do
  use ExUnit.Case, async: true
  alias ExPhil.Data.Costumes

  test "Fox slots follow ftFx_Init_CostumeStrings (Nr, Or, La, Gr)" do
    assert Enum.map(0..3, &Costumes.name("Fox", &1)) == [:default, :orange, :lavender, :green]
    assert Costumes.perceived("Fox", 1) == :red
    assert Costumes.perceived("Fox", 2) == :blue
    assert Costumes.perceived("Fox", 3) == :green
    assert Costumes.name("Fox", 4) == :unknown
  end

  test "every table starts with the neutral costume and uses known codes" do
    for {_char, codes} <- Costumes.tables() do
      assert hd(codes) == "Nr"
      for c <- codes, do: assert(Costumes.name("Fox", 0) == :default and c =~ ~r/^[A-Z][a-z]$/)
    end
  end

  test "off-roster characters and nil slots are :unknown" do
    assert Costumes.name("Kirby", 0) == :unknown
    assert Costumes.name("Fox", nil) == :unknown
  end
end
