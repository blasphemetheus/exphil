defmodule ExPhil.Data.ReplayParserTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.ReplayParser
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  describe "convert helpers" do
    # These test the internal conversion functions through public API

    test "to_training_frames/1 extracts game_state and controller" do
      # Create mock parsed data structure
      parsed = %{
        frames: [
          %{
            game_state: %GameState{
              frame: 0,
              stage: 32,
              menu_state: 2,
              players: %{
                1 => %Player{x: 0.0, y: 0.0, percent: 0.0, stock: 4},
                2 => %Player{x: 50.0, y: 0.0, percent: 0.0, stock: 4}
              },
              projectiles: [],
              distance: 50.0
            },
            controller: %ControllerState{
              main_stick: %{x: 0.5, y: 0.5},
              c_stick: %{x: 0.5, y: 0.5},
              l_shoulder: 0.0,
              r_shoulder: 0.0,
              button_a: false,
              button_b: false,
              button_x: false,
              button_y: false,
              button_z: false,
              button_l: false,
              button_r: false,
              button_d_up: false
            }
          }
        ],
        metadata: %{path: "test.slp"}
      }

      frames = ReplayParser.to_training_frames(parsed)

      assert length(frames) == 1
      [frame] = frames
      assert %{game_state: gs, controller: cs} = frame
      assert %GameState{} = gs
      assert %ControllerState{} = cs
    end
  end

  describe "character_name mapping" do
    # Test the character name mapping is consistent
    @character_names %{
      captain_falcon: "captain_falcon",
      donkey_kong: "donkey_kong",
      fox: "fox",
      game_and_watch: "game_and_watch",
      kirby: "kirby",
      bowser: "bowser",
      link: "link",
      luigi: "luigi",
      mario: "mario",
      marth: "marth",
      mewtwo: "mewtwo",
      ness: "ness",
      peach: "peach",
      pikachu: "pikachu",
      ice_climbers: "ice_climbers",
      jigglypuff: "jigglypuff",
      samus: "samus",
      yoshi: "yoshi",
      zelda: "zelda",
      sheik: "sheik",
      falco: "falco",
      young_link: "young_link",
      dr_mario: "dr_mario",
      roy: "roy",
      pichu: "pichu",
      ganondorf: "ganondorf"
    }

    test "all target characters have valid mappings" do
      target_chars = [:mewtwo, :game_and_watch, :link, :ganondorf]

      for char <- target_chars do
        assert Map.has_key?(@character_names, char),
               "Target character #{char} should have a name mapping"
      end
    end
  end

  describe "error handling" do
    test "parse/1 returns error for non-existent file" do
      result = ReplayParser.parse("/nonexistent/file.slp")

      # Should return error (Python not finding file or parser error)
      assert {:error, _reason} = result
    end

    test "load_parsed/1 returns error for non-existent file" do
      result = ReplayParser.load_parsed("/nonexistent/file.json")

      assert {:error, _reason} = result
    end

    test "load_parsed/1 returns error for invalid JSON" do
      # Create a temp file with invalid content
      path = Path.join(System.tmp_dir!(), "invalid_#{:rand.uniform(10_000)}.json")

      try do
        File.write!(path, "not valid json {{{")
        result = ReplayParser.load_parsed(path)

        assert {:error, _reason} = result
      after
        File.rm(path)
      end
    end
  end

  describe "load_parsed/1" do
    test "loads and converts valid JSON file" do
      # Create mock parsed replay JSON
      mock_data = %{
        "frames" => [
          %{
            "game_state" => %{
              "frame" => 123,
              "stage" => 32,
              "menu_state" => 2,
              "distance" => 45.5,
              "players" => %{
                "1" => %{
                  "character" => 9,
                  "x" => -10.5,
                  "y" => 5.0,
                  "percent" => 42.0,
                  "stock" => 3,
                  "facing" => -1,
                  "action" => 14,
                  "action_frame" => 10,
                  "invulnerable" => false,
                  "jumps_left" => 1,
                  "on_ground" => true,
                  "shield_strength" => 55.0,
                  "hitstun_frames_left" => 0,
                  "speed_air_x_self" => 0.5,
                  "speed_y_self" => -1.0
                },
                "2" => %{
                  "character" => 2,
                  "x" => 35.0,
                  "y" => 0.0,
                  "percent" => 85.0,
                  "stock" => 4
                }
              }
            },
            "controller" => %{
              "main_stick" => %{"x" => 0.0, "y" => 0.5},
              "c_stick" => %{"x" => 0.5, "y" => 0.5},
              "l_shoulder" => 0.35,
              "r_shoulder" => 0.0,
              "button_a" => true,
              "button_b" => false,
              "button_x" => false,
              "button_y" => false,
              "button_z" => false,
              "button_l" => true,
              "button_r" => false,
              "button_d_up" => false
            }
          }
        ],
        "metadata" => %{
          "path" => "test_game.slp",
          "stage" => 32,
          "duration_frames" => 5400,
          "player_port" => 1,
          "opponent_port" => 2,
          "players" => %{
            "1" => %{"character" => "Mewtwo"},
            "2" => %{"character" => "Fox"}
          }
        },
        "success" => true
      }

      path = Path.join(System.tmp_dir!(), "mock_replay_#{:rand.uniform(10_000)}.json")

      try do
        File.write!(path, Jason.encode!(mock_data))

        {:ok, result} = ReplayParser.load_parsed(path)

        # Verify structure
        assert result.success == true
        assert length(result.frames) == 1

        # Verify frame conversion
        [frame] = result.frames
        assert %GameState{} = frame.game_state
        assert %ControllerState{} = frame.controller

        # Verify game state values
        gs = frame.game_state
        assert gs.frame == 123
        assert gs.stage == 32
        assert gs.distance == 45.5
        assert map_size(gs.players) == 2

        # Verify player conversion
        player1 = gs.players[1]
        assert %Player{} = player1
        assert player1.character == 9
        assert player1.x == -10.5
        assert player1.y == 5.0
        assert player1.percent == 42.0
        assert player1.stock == 3
        assert player1.facing == -1
        assert player1.jumps_left == 1
        assert player1.shield_strength == 55.0

        # Verify controller conversion
        cs = frame.controller
        assert cs.main_stick.x == 0.0
        assert cs.main_stick.y == 0.5
        assert cs.l_shoulder == 0.35
        assert cs.button_a == true
        assert cs.button_l == true
        assert cs.button_b == false

        # Verify metadata
        assert result.metadata.path == "test_game.slp"
        assert result.metadata.duration_frames == 5400
        assert result.metadata.player_port == 1
      after
        File.rm(path)
      end
    end

    test "loads gzipped JSON file" do
      mock_data = %{
        "frames" => [
          %{
            "game_state" => %{
              "frame" => 0,
              "stage" => 32,
              "players" => %{}
            },
            "controller" => %{
              "main_stick" => %{"x" => 0.5, "y" => 0.5}
            }
          }
        ],
        "metadata" => %{},
        "success" => true
      }

      path = Path.join(System.tmp_dir!(), "mock_replay_#{:rand.uniform(10_000)}.json.gz")

      try do
        json = Jason.encode!(mock_data)
        compressed = :zlib.gzip(json)
        File.write!(path, compressed)

        {:ok, result} = ReplayParser.load_parsed(path)

        assert result.success == true
        assert length(result.frames) == 1
      after
        File.rm(path)
      end
    end

    test "handles missing optional fields with defaults" do
      # Minimal data with only required fields
      mock_data = %{
        "frames" => [
          %{
            "game_state" => %{
              "players" => %{
                "1" => %{}
              }
            },
            "controller" => nil
          }
        ],
        "metadata" => %{}
      }

      path = Path.join(System.tmp_dir!(), "minimal_#{:rand.uniform(10_000)}.json")

      try do
        File.write!(path, Jason.encode!(mock_data))

        {:ok, result} = ReplayParser.load_parsed(path)

        [frame] = result.frames

        # Game state should have defaults
        assert frame.game_state.frame == 0
        assert frame.game_state.stage == 0
        assert frame.game_state.menu_state == 2

        # Player should have defaults
        player = frame.game_state.players[1]
        assert player.x == 0.0
        assert player.y == 0.0
        assert player.percent == 0.0
        assert player.stock == 4
        assert player.facing == 1
        assert player.jumps_left == 2
        assert player.on_ground == true
        assert player.shield_strength == 60.0

        # Controller should be nil
        assert frame.controller == nil
      after
        File.rm(path)
      end
    end
  end

  describe "to_training_frames/1" do
    test "preserves all frame data for training" do
      parsed = %{
        frames: [
          %{
            game_state: %GameState{frame: 0, stage: 32, players: %{}},
            controller: %ControllerState{
              main_stick: %{x: 0.0, y: 1.0},
              button_a: true
            },
            metadata: %{extra: "data"}
          },
          %{
            game_state: %GameState{frame: 1, stage: 32, players: %{}},
            controller: %ControllerState{
              main_stick: %{x: 1.0, y: 0.0},
              button_b: true
            }
          }
        ]
      }

      frames = ReplayParser.to_training_frames(parsed)

      assert length(frames) == 2

      [f1, f2] = frames
      assert f1.game_state.frame == 0
      assert f1.controller.button_a == true
      assert f2.game_state.frame == 1
      assert f2.controller.button_b == true
    end
  end

  describe "stream_parsed/1" do
    test "returns empty stream for non-existent directory" do
      stream = ReplayParser.stream_parsed("/nonexistent/path")
      result = Enum.to_list(stream)

      assert result == []
    end

    test "streams multiple files from directory" do
      dir = Path.join(System.tmp_dir!(), "stream_test_#{:rand.uniform(10_000)}")
      File.mkdir_p!(dir)

      try do
        # Create multiple mock files
        for i <- 1..3 do
          mock_data = %{
            "frames" => [%{"game_state" => %{"frame" => i, "players" => %{}}}],
            "metadata" => %{"path" => "game#{i}.slp"},
            "success" => true
          }

          path = Path.join(dir, "game#{i}.json")
          File.write!(path, Jason.encode!(mock_data))
        end

        results = ReplayParser.stream_parsed(dir) |> Enum.to_list()

        assert length(results) == 3
        assert Enum.all?(results, &(&1.success == true))
      after
        File.rm_rf(dir)
      end
    end

    test "skips files that fail to load" do
      dir = Path.join(System.tmp_dir!(), "stream_skip_#{:rand.uniform(10_000)}")
      File.mkdir_p!(dir)

      try do
        # Create one valid and one invalid file
        valid = %{"frames" => [], "metadata" => %{}, "success" => true}
        File.write!(Path.join(dir, "valid.json"), Jason.encode!(valid))
        File.write!(Path.join(dir, "invalid.json"), "not json {{{{")

        results = ReplayParser.stream_parsed(dir) |> Enum.to_list()

        # Should only get the valid one
        assert length(results) == 1
      after
        File.rm_rf(dir)
      end
    end
  end
end
