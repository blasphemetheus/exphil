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

  describe "to_training_frames :remap_ports (09-01 port-2 corruption fix)" do
    alias ExPhil.Data.Peppi
    alias ExPhil.Data.Peppi.{ParsedReplay, GameFrame, PlayerFrame}

    defp synthetic_replay do
      ctrl = %Peppi.Controller{
        main_stick_x: 0.5, main_stick_y: 0.5, c_stick_x: 0.5, c_stick_y: 0.5,
        l_trigger: 0.0, r_trigger: 0.0,
        button_a: false, button_b: false, button_x: false, button_y: false,
        button_z: false, button_l: false, button_r: false, button_start: false,
        button_d_up: false, button_d_down: false, button_d_left: false,
        button_d_right: false
      }

      pf = fn char, x ->
        %PlayerFrame{character: char, x: x, y: 0.0, percent: 0.0, stock: 4,
                     facing: 1, action: 14, action_frame: 0,
                     invulnerable: false, jumps_left: 2, on_ground: true,
                     shield_strength: 60.0, hitstun_frames_left: 0.0,
                     speed_air_x_self: 0.0, speed_ground_x_self: 0.0,
                     speed_y_self: 0.0, speed_x_attack: 0.0, speed_y_attack: 0.0,
                     controller: ctrl}
      end

      frames =
        Enum.map(0..5, fn i ->
          %GameFrame{frame_number: i,
                     players: %{1 => pf.(17, -10.0), 2 => pf.(1, 10.0 + i)}}
        end)

      %ParsedReplay{frames: frames, metadata: %{stage: 32, players: []}}
    end

    test "port-2 subject remaps to slot 1 with the REAL opponent in slot 2" do
      frames =
        Peppi.to_training_frames(synthetic_replay(),
          player_port: 2, opponent_port: 1, remap_ports: true)

      gs = hd(frames).game_state
      assert Map.keys(gs.players) |> Enum.sort() == [1, 2]
      # subject (was port 2, char 1) now under the embedding's own slot
      assert gs.players[1].character == 1
      # the opponent (was port 1) is PRESENT — the bug dropped it entirely
      assert gs.players[2].character == 17
      assert gs.distance > 0.0
      # convention stamp (GOTCHA #107 enforcement)
      assert gs.own_port == 1
    end

    test "embedding a stamped state from the wrong perspective RAISES" do
      frames =
        Peppi.to_training_frames(synthetic_replay(),
          player_port: 2, opponent_port: 1, remap_ports: true)

      gs = hd(frames).game_state

      # correct perspective embeds fine
      assert %Nx.Tensor{} = ExPhil.Embeddings.Game.embed_state(gs, 1)

      # swapped perspective is a loud crash, not a silent corruption
      assert_raise ArgumentError, ~r/port-convention violation/, fn ->
        ExPhil.Embeddings.Game.embed_state(gs, 2)
      end
    end

    test "without remap, a port-2 subject leaves slot 1 empty (the bug's shape)" do
      frames =
        Peppi.to_training_frames(synthetic_replay(),
          player_port: 2, opponent_port: 1)

      gs = hd(frames).game_state
      assert gs.players[1].character == 17
      assert gs.players[2].character == 1
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
