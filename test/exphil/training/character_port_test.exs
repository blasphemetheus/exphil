defmodule ExPhil.Training.CharacterPortTest do
  @moduledoc """
  --select-character-port (E1 fix, 2026-08-31): per-file resolution of the
  imitated port to the --train-character player. Without it the streaming
  loader imitates port 1 regardless of who sits there — fox_gen_v1's
  43%-non-fox corpus (eval_runs/0830_corpus_mix).
  """
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.{Config, Pipeline}

  # External character ids (CSS order): Fox = 2.
  @fox 2
  @marth 9

  describe "Pipeline.character_port/2" do
    test "character on port 2 selects port 2 (the 43% v1 got wrong)" do
      players = [%{port: 1, character: @marth}, %{port: 2, character: @fox}]
      assert Pipeline.character_port(players, @fox) == {:port, 2}
    end

    test "character on port 1 selects port 1" do
      players = [%{port: 1, character: @fox}, %{port: 2, character: @marth}]
      assert Pipeline.character_port(players, @fox) == {:port, 1}
    end

    test "ditto is :ditto (caller falls back to port 1)" do
      players = [%{port: 1, character: @fox}, %{port: 2, character: @fox}]
      assert Pipeline.character_port(players, @fox) == :ditto
    end

    test "absent character is :absent" do
      players = [%{port: 1, character: @marth}, %{port: 2, character: @marth}]
      assert Pipeline.character_port(players, @fox) == :absent
    end

    test "non-adjacent ports resolve to the real port, not an index" do
      players = [%{port: 2, character: @marth}, %{port: 4, character: @fox}]
      assert Pipeline.character_port(players, @fox) == {:port, 4}
    end
  end

  describe "--select-character-port flag" do
    test "defaults to false" do
      refute Config.parse_args([])[:select_character_port]
    end

    test "parses to true" do
      opts = Config.parse_args(["--select-character-port", "--train-character", "fox"])
      assert opts[:select_character_port] == true
      assert opts[:train_character] == :fox
    end
  end
end
