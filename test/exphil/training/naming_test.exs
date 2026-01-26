defmodule ExPhil.Training.NamingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Naming

  describe "generate/0" do
    test "returns a string in adjective_noun format" do
      name = Naming.generate()
      assert is_binary(name)
      assert String.contains?(name, "_")

      [adj, noun] = String.split(name, "_", parts: 2)
      assert String.length(adj) > 0
      assert String.length(noun) > 0
    end

    test "generates different names on subsequent calls" do
      names = for _ <- 1..10, do: Naming.generate()
      # With 100+ adjectives and 100+ nouns, getting 10 identical names is vanishingly unlikely
      unique_names = Enum.uniq(names)
      assert length(unique_names) > 1
    end

    test "all generated names use valid adjectives and nouns" do
      valid_adjectives = Naming.adjectives()
      valid_nouns = Naming.nouns() |> MapSet.new()

      for _ <- 1..50 do
        name = Naming.generate()
        # Some adjectives contain underscores (e.g., "pivot_grabbing"), so we can't
        # simply split by "_". Instead, find which adjective the name starts with.
        {adj, rest} = find_adjective_prefix(name, valid_adjectives)
        assert adj != nil, "No valid adjective prefix found in: #{name}"
        # Rest should be "_noun", strip the leading underscore
        noun = String.trim_leading(rest, "_")
        assert MapSet.member?(valid_nouns, noun), "Invalid noun: #{noun} in name: #{name}"
      end
    end

    defp find_adjective_prefix(name, adjectives) do
      # Sort by length descending to match longest first (e.g., "pivot_grabbing" before "pivot")
      sorted = Enum.sort_by(adjectives, &(-String.length(&1)))

      Enum.find_value(sorted, {nil, name}, fn adj ->
        if String.starts_with?(name, adj <> "_") do
          {adj, String.trim_leading(name, adj)}
        end
      end)
    end
  end

  describe "generate/1 with seed" do
    test "produces consistent results with same seed" do
      name1 = Naming.generate(12345)
      name2 = Naming.generate(12345)
      assert name1 == name2
    end

    test "produces different results with different seeds" do
      name1 = Naming.generate(12345)
      name2 = Naming.generate(67890)
      # Different seeds should (almost certainly) produce different names
      assert name1 != name2
    end
  end

  describe "adjectives/0" do
    test "returns a list of adjectives" do
      adjectives = Naming.adjectives()
      assert is_list(adjectives)
      assert length(adjectives) > 50
    end

    test "includes general adjectives" do
      adjectives = Naming.adjectives()
      assert "brave" in adjectives
      assert "cosmic" in adjectives
      assert "swift" in adjectives
    end

    test "includes Melee-specific adjectives" do
      adjectives = Naming.adjectives()
      assert "wavedashing" in adjectives
      assert "techchasing" in adjectives
      assert "multishining" in adjectives
    end

    test "includes hardware/mod adjectives" do
      adjectives = Naming.adjectives()
      assert "notched" in adjectives
      assert "modded" in adjectives
      assert "phob" in adjectives
      assert "rollbacked" in adjectives
      assert "homebrewed" in adjectives
    end
  end

  describe "nouns/0" do
    test "returns a list of nouns" do
      nouns = Naming.nouns()
      assert is_list(nouns)
      assert length(nouns) > 50
    end

    test "includes general nouns (animals)" do
      nouns = Naming.nouns()
      assert "falcon" in nouns
      assert "phoenix" in nouns
      assert "dragon" in nouns
    end

    test "includes Melee concept nouns" do
      nouns = Naming.nouns()
      assert "wavedash" in nouns
      assert "tipper" in nouns
      assert "edgeguard" in nouns
    end

    test "does not include Melee character names" do
      nouns = Naming.nouns()
      # These are playable characters in Melee - should NOT be in the list
      refute "marth" in nouns
      refute "fox" in nouns
      refute "falco" in nouns
      refute "sheik" in nouns
      refute "peach" in nouns
      refute "jigglypuff" in nouns
      refute "captain_falcon" in nouns
      refute "samus" in nouns
      refute "ganondorf" in nouns
      refute "mewtwo" in nouns
    end

    test "includes hardware/mod nouns" do
      nouns = Naming.nouns()
      assert "phob" in nouns
      assert "goomwave" in nouns
      assert "rectangle" in nouns
      assert "boxx" in nouns
      assert "slippi" in nouns
      assert "rollback" in nouns
      assert "nintendont" in nouns
      assert "ucf" in nouns
    end
  end
end
