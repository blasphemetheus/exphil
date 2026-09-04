defmodule ExPhil.Data.SubjectResolverTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.SubjectResolver, as: R

  defp p(port, char, name \\ nil), do: %{port: port, character: char, netplay_name: name}

  @fox 2
  @marth 9

  describe "explicit rung" do
    test "resolves an occupied port with provenance" do
      players = [p(1, @fox), p(2, @marth)]

      assert {:ok, %{subject_port: 2, opponent_port: 1, provenance: :explicit}} =
               R.resolve(players, subject_port: 2)
    end

    test "errors on an unoccupied port — no silent default" do
      assert {:error, {:port_not_occupied, 3}} =
               R.resolve([p(1, @fox), p(2, @marth)], subject_port: 3)
    end
  end

  describe "identity rung" do
    test "matches netplay name case-insensitively" do
      players = [p(1, @fox, "INFP"), p(2, @fox, "RUDE")]

      assert {:ok, %{subject_port: 1, provenance: :identity}} =
               R.resolve(players, subject_identity: "infp")
    end

    test "anonymization placeholders never match" do
      players = [p(1, @fox, "Master Player"), p(2, @fox, "Master Player")]

      assert {:error, {:identity_not_found, _}} =
               R.resolve(players, subject_identity: "Master Player")
    end

    test "duplicate identities are ambiguous, not defaulted" do
      players = [p(1, @fox, "INFP"), p(2, @fox, "INFP")]
      assert {:error, {:identity_ambiguous, _}} = R.resolve(players, subject_identity: "INFP")
    end
  end

  describe "character rung" do
    test "unique character match, by id, atom, or name" do
      players = [p(2, @marth), p(3, @fox)]

      for char <- [2, :fox, "Fox", "fox"] do
        assert {:ok, %{subject_port: 3, opponent_port: 2, provenance: :character}} =
                 R.resolve(players, subject_character: char)
      end
    end

    test "ditto is a loud error by default" do
      players = [p(1, @fox), p(2, @fox)]
      assert {:error, :ditto_ambiguous} = R.resolve(players, subject_character: :fox)
    end

    test "explicit :port1 tie-break is provenance-marked" do
      players = [p(2, @fox), p(4, @fox)]

      assert {:ok, %{subject_port: 2, provenance: :character_tie_break}} =
               R.resolve(players, subject_character: :fox, ditto_tie_break: :port1)
    end

    test "{:port, n} tie-break must actually be one of the matches" do
      players = [p(1, @fox), p(2, @fox)]

      assert {:ok, %{subject_port: 2, provenance: :character_tie_break}} =
               R.resolve(players, subject_character: :fox, ditto_tie_break: {:port, 2})

      assert {:error, {:tie_break_port_not_a_match, 3}} =
               R.resolve(players, subject_character: :fox, ditto_tie_break: {:port, 3})
    end

    test "absent character errors" do
      assert {:error, {:character_absent, :fox}} =
               R.resolve([p(1, @marth), p(2, @marth)], subject_character: :fox)
    end
  end

  test "no anchor given is an error, never port 1" do
    assert {:error, :no_anchor_given} = R.resolve([p(1, @fox), p(2, @marth)], [])
  end

  test "external_character_id accepts ids, numeric strings, atoms, names" do
    assert {:ok, 2} = R.external_character_id(:fox)
    assert {:ok, 2} = R.external_character_id("Fox")
    assert {:ok, 2} = R.external_character_id(2)
    assert {:ok, 2} = R.external_character_id("2")
    assert {:ok, 0} = R.external_character_id("Captain Falcon")
    assert {:error, {:unknown_character, _}} = R.external_character_id("steve")
  end
end
