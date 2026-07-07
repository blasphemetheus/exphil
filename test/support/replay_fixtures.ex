defmodule ExPhil.Test.ReplayFixtures do
  @moduledoc """
  Realistic game scenario fixtures for testing.

  Provides pre-built game states representing common Melee situations:
  - Neutral game positioning
  - Combo sequences
  - Edge guard scenarios
  - Recovery situations
  - Tech chase situations

  These fixtures are deterministic and designed to test that embeddings
  and models handle diverse game states correctly.

  ## Usage

      import ExPhil.Test.ReplayFixtures

      # Get a specific scenario
      game_state = neutral_game_fixture(:mewtwo_vs_fox)

      # Get a sequence of frames
      frames = combo_sequence_fixture(:fox_upthrow_upair)
  """

  alias ExPhil.Bridge.{GameState, Player, ControllerState}
  alias ExPhil.Data.Peppi

  # Character IDs (Melee internal)
  @mewtwo 10
  @fox 2
  # @falco 20
  @marth 9
  @sheik 19
  @ganondorf 25
  @link 6
  # @game_and_watch 24

  # Stage IDs
  @final_destination 32
  @battlefield 31
  # @yoshis_story 8
  # @fountain_of_dreams 2
  # @dream_land 28
  # @pokemon_stadium 3

  # Action states (common)
  # Standing idle
  @wait 14
  # Dash start
  @dash 20
  # Running
  @run 22
  # Jump squat
  @jump_squat 24
  # Forward jump
  @jump_f 25
  # Falling
  @fall 30
  # Fox grounded Reflector / shine cycle (approximate — the precise Reflector
  # action-state ids live in the Melee action-state table; the multishine
  # fixture only relies on the state *changing* across the cycle, not on the
  # exact numeric ids, so these are safe placeholders for a synthetic test).
  @fox_shine 360
  # @landing 40       # Landing lag
  # @shield 178       # Shielding
  # @grabbed 223      # Being grabbed
  # @dead_down 0      # Dead (down)

  # ============================================================================
  # Neutral Game Fixtures
  # ============================================================================

  @doc """
  Get a neutral game state fixture.

  ## Scenarios
    - `:mewtwo_vs_fox` - Mewtwo vs Fox on FD, center stage
    - `:marth_vs_sheik` - Marth vs Sheik on Battlefield
    - `:fox_ditto` - Fox vs Fox on Yoshis
    - `:low_tier` - Ganondorf vs Link on FD
  """
  def neutral_game_fixture(scenario \\ :mewtwo_vs_fox)

  def neutral_game_fixture(:mewtwo_vs_fox) do
    %GameState{
      frame: 120,
      stage: @final_destination,
      menu_state: 2,
      players: %{
        1 => %Player{
          character: @mewtwo,
          x: -30.0,
          y: 0.0,
          percent: 24.0,
          stock: 4,
          facing: 1,
          action: @wait,
          action_frame: 15,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        },
        2 => %Player{
          character: @fox,
          x: 30.0,
          y: 0.0,
          percent: 18.0,
          stock: 4,
          facing: -1,
          action: @dash,
          action_frame: 5,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: -2.2,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        }
      },
      projectiles: [],
      items: [],
      distance: 60.0
    }
  end

  def neutral_game_fixture(:marth_vs_sheik) do
    %GameState{
      frame: 240,
      stage: @battlefield,
      menu_state: 2,
      players: %{
        1 => %Player{
          character: @marth,
          x: -45.0,
          y: 0.0,
          percent: 42.0,
          stock: 3,
          facing: 1,
          action: @run,
          action_frame: 12,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 55.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 1.6,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        },
        2 => %Player{
          character: @sheik,
          x: 20.0,
          # On side platform
          y: 27.5,
          percent: 55.0,
          stock: 3,
          facing: -1,
          action: @wait,
          action_frame: 8,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        }
      },
      projectiles: [],
      items: [],
      distance: 72.0
    }
  end

  def neutral_game_fixture(:low_tier) do
    %GameState{
      frame: 180,
      stage: @final_destination,
      menu_state: 2,
      players: %{
        1 => %Player{
          character: @ganondorf,
          x: -50.0,
          y: 0.0,
          percent: 35.0,
          stock: 4,
          facing: 1,
          action: @wait,
          action_frame: 20,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        },
        2 => %Player{
          character: @link,
          x: 35.0,
          y: 0.0,
          percent: 28.0,
          stock: 4,
          facing: -1,
          action: @wait,
          action_frame: 5,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        }
      },
      projectiles: [],
      items: [],
      distance: 85.0
    }
  end

  # ============================================================================
  # Edge Guard Fixtures
  # ============================================================================

  @doc """
  Get an edge guard scenario fixture.

  ## Scenarios
    - `:fox_recovering_low` - Fox recovering low to ledge
    - `:mewtwo_offstage` - Mewtwo offstage with double jump
    - `:ganondorf_recovering` - Ganondorf in danger recovering
  """
  def edge_guard_fixture(scenario \\ :fox_recovering_low)

  def edge_guard_fixture(:fox_recovering_low) do
    %GameState{
      frame: 450,
      stage: @final_destination,
      menu_state: 2,
      players: %{
        1 => %Player{
          character: @marth,
          # Near ledge
          x: -65.0,
          y: 0.0,
          percent: 45.0,
          stock: 3,
          facing: 1,
          action: @wait,
          action_frame: 10,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        },
        2 => %Player{
          character: @fox,
          # Offstage
          x: -95.0,
          # Below stage
          y: -40.0,
          percent: 78.0,
          stock: 3,
          facing: 1,
          # Firefox startup
          action: 355,
          action_frame: 15,
          on_ground: false,
          jumps_left: 0,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: 0.0,
          speed_y_self: 0.8,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        }
      },
      projectiles: [],
      items: [],
      distance: 50.0
    }
  end

  def edge_guard_fixture(:mewtwo_offstage) do
    %GameState{
      frame: 520,
      stage: @battlefield,
      menu_state: 2,
      players: %{
        1 => %Player{
          character: @mewtwo,
          x: -80.0,
          y: 20.0,
          percent: 62.0,
          stock: 2,
          facing: 1,
          action: @fall,
          action_frame: 8,
          on_ground: false,
          # Has double jump
          jumps_left: 1,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 1.2,
          speed_ground_x_self: 0.0,
          speed_y_self: -1.5,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        },
        2 => %Player{
          character: @sheik,
          x: -55.0,
          y: 0.0,
          percent: 95.0,
          stock: 2,
          facing: -1,
          action: @dash,
          action_frame: 3,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: -1.8,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil,
          controller_state: nil
        }
      },
      projectiles: [],
      items: [],
      distance: 30.0
    }
  end

  # ============================================================================
  # Combo Sequence Fixtures
  # ============================================================================

  @doc """
  Get a multi-frame combo sequence.

  Returns a list of {game_state, controller} tuples representing
  consecutive frames of a combo.

  ## Scenarios
    - `:fox_upthrow_upair` - Fox upthrow to upair (5 frames)
    - `:marth_grab_fthrow` - Marth grab to forward throw (4 frames)
  """
  def combo_sequence_fixture(scenario \\ :fox_upthrow_upair)

  def combo_sequence_fixture(:fox_upthrow_upair) do
    # Frame 1: Fox just threw, opponent in hitstun going up
    base_frame = 600

    [
      # Frame 1: Throw release
      {
        %GameState{
          frame: base_frame,
          stage: @final_destination,
          menu_state: 2,
          players: %{
            1 => %Player{
              character: @fox,
              x: 0.0,
              y: 0.0,
              percent: 45.0,
              stock: 3,
              facing: 1,
              # Throw release
              action: 215,
              action_frame: 18,
              on_ground: true,
              jumps_left: 2,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 0,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 0.0,
              speed_x_attack: 0.0,
              speed_y_attack: 0.0,
              nana: nil,
              controller_state: nil
            },
            2 => %Player{
              character: @marth,
              x: 0.0,
              y: 25.0,
              percent: 72.0,
              stock: 3,
              facing: -1,
              # Damage fly top
              action: 75,
              action_frame: 2,
              on_ground: false,
              jumps_left: 2,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 12,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 3.5,
              speed_x_attack: 0.0,
              speed_y_attack: 2.8,
              nana: nil,
              controller_state: nil
            }
          },
          projectiles: [],
          items: [],
          distance: 25.0
        },
        %ControllerState{
          # Holding up
          main_stick: %{x: 0.5, y: 1.0},
          c_stick: %{x: 0.5, y: 0.5},
          l_shoulder: 0.0,
          r_shoulder: 0.0,
          button_a: false,
          button_b: false,
          # Jump input
          button_x: true,
          button_y: false,
          button_z: false,
          button_l: false,
          button_r: false,
          button_d_up: false
        }
      },
      # Frame 2: Fox in jumpsquat
      {
        %GameState{
          frame: base_frame + 1,
          stage: @final_destination,
          menu_state: 2,
          players: %{
            1 => %Player{
              character: @fox,
              x: 0.0,
              y: 0.0,
              percent: 45.0,
              stock: 3,
              facing: 1,
              action: @jump_squat,
              action_frame: 1,
              on_ground: true,
              jumps_left: 2,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 0,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 0.0,
              speed_x_attack: 0.0,
              speed_y_attack: 0.0,
              nana: nil,
              controller_state: nil
            },
            2 => %Player{
              character: @marth,
              x: 0.0,
              y: 28.0,
              percent: 72.0,
              stock: 3,
              facing: -1,
              action: 75,
              action_frame: 3,
              on_ground: false,
              jumps_left: 2,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 11,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 3.2,
              speed_x_attack: 0.0,
              speed_y_attack: 2.5,
              nana: nil,
              controller_state: nil
            }
          },
          projectiles: [],
          items: [],
          distance: 28.0
        },
        neutral_controller()
      },
      # Frame 3: Fox jumping, pressing A for upair
      {
        %GameState{
          frame: base_frame + 4,
          stage: @final_destination,
          menu_state: 2,
          players: %{
            1 => %Player{
              character: @fox,
              x: 0.0,
              y: 12.0,
              percent: 45.0,
              stock: 3,
              facing: 1,
              action: @jump_f,
              action_frame: 2,
              on_ground: false,
              jumps_left: 1,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 0,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 3.8,
              speed_x_attack: 0.0,
              speed_y_attack: 0.0,
              nana: nil,
              controller_state: nil
            },
            2 => %Player{
              character: @marth,
              x: 0.0,
              y: 35.0,
              percent: 72.0,
              stock: 3,
              facing: -1,
              action: 75,
              action_frame: 6,
              on_ground: false,
              jumps_left: 2,
              shield_strength: 60.0,
              invulnerable: false,
              hitstun_frames_left: 8,
              speed_air_x_self: 0.0,
              speed_ground_x_self: 0.0,
              speed_y_self: 2.5,
              speed_x_attack: 0.0,
              speed_y_attack: 2.0,
              nana: nil,
              controller_state: nil
            }
          },
          projectiles: [],
          items: [],
          distance: 23.0
        },
        %ControllerState{
          main_stick: %{x: 0.5, y: 1.0},
          # C-stick up for upair
          c_stick: %{x: 0.5, y: 1.0},
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
    ]
  end

  # ============================================================================
  # Tech Fixtures (overfit-replication tests — see docs/planning/HANDOFF.md)
  # ============================================================================

  @doc """
  Deterministic tech-execution fixture for overfit-replication correctness tests.

  Returns a list of `{%GameState{}, %ControllerState{}}` tuples (same shape as
  `combo_sequence_fixture/1`) representing a repeating tech pattern. The point of
  this fixture is Vlad Firoiu's advice: train a model to *memorize* a short,
  legible behavior, then check it can reproduce it. If a memorized behavior can't
  be replicated, the bug is categorically in the pipeline (embedding / discretize
  / decode / sampling), not the model — collapsing the "underfitting vs bug?"
  ambiguity that famously cost Phillip ~2 years.

  ## CRITICAL correctness property

  The game **state varies in lockstep with the controller phase**. Multishine is a
  *periodic* output; if every frame fed the model an identical state, no correct
  model could produce a periodic controller sequence from constant input, and the
  test would fail a perfectly good pipeline. Real replays don't have this problem
  (Fox's action-state genuinely cycles shine → jumpsquat → airborne → shine), so
  the synthetic fixture cycles Fox's `action` and `y` with the shine phase too.

  ## Scenarios
    - `:multishine` — Fox (port 2) grounded multishine vs a static Marth (port 1)

  ## Options
    - `:frames` — total frames (default 64)
    - `:period` — frames per shine cycle (default 8); shine input lands on phase 0
    - `:stage`  — stage id (default Final Destination)

  The shine input is encoded as **B held + main stick down** (`button_b: true`,
  `main_stick.y < 0.5`), which is exactly how `ExPhil.Test.ReplicationCheck`
  detects a shine event, so the fixture and the checker agree by construction.
  """
  def tech_fixture(scenario \\ :multishine, opts \\ [])

  def tech_fixture(:multishine, opts) do
    frames = Keyword.get(opts, :frames, 64)
    period = Keyword.get(opts, :period, 8)
    stage = Keyword.get(opts, :stage, @final_destination)

    for i <- 0..(frames - 1) do
      phase = rem(i, period)
      {fox_action, fox_y, controller} = multishine_phase(phase)

      gs = %GameState{
        frame: i,
        stage: stage,
        menu_state: 2,
        players: %{
          # Static opponent — its constancy is fine; only Fox needs phase variation.
          1 => multishine_opponent(),
          2 => %Player{
            character: @fox,
            x: 0.0,
            y: fox_y,
            percent: 0.0,
            stock: 4,
            facing: 1,
            action: fox_action,
            action_frame: phase + 1,
            on_ground: fox_y == 0.0,
            jumps_left: 2,
            shield_strength: 60.0,
            invulnerable: false,
            hitstun_frames_left: 0,
            speed_air_x_self: 0.0,
            speed_ground_x_self: 0.0,
            speed_y_self: 0.0,
            speed_x_attack: 0.0,
            speed_y_attack: 0.0,
            nana: nil,
            controller_state: nil
          }
        },
        projectiles: [],
        items: [],
        distance: 40.0
      }

      {gs, controller}
    end
  end

  # Phase 0 = shine (B + down), phase 1 = jump-cancel (X), phases 2+ = airborne
  # rising between shines. State (action + y) tracks the phase so the sequence is
  # actually determinable — see the fixture's CRITICAL correctness note.
  defp multishine_phase(0) do
    {@fox_shine, 0.0, %{neutral_controller() | main_stick: %{x: 0.5, y: 0.0}, button_b: true}}
  end

  defp multishine_phase(1) do
    {@jump_squat, 0.0, %{neutral_controller() | button_x: true}}
  end

  defp multishine_phase(phase) do
    # Small upward bounce so consecutive airborne frames aren't identical either.
    {@jump_f, phase * 1.5, neutral_controller()}
  end

  defp multishine_opponent do
    %Player{
      character: @marth,
      x: 40.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: -1,
      action: @wait,
      action_frame: 1,
      on_ground: true,
      jumps_left: 2,
      shield_strength: 60.0,
      invulnerable: false,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  @doc "Create a neutral controller state"
  def neutral_controller do
    %ControllerState{
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
  end

  @doc """
  Convert fixture game states to Peppi-style parsed replay format.

  Useful for testing code that expects Peppi.ParsedReplay structures.
  """
  def to_parsed_replay(game_states, opts \\ []) when is_list(game_states) do
    stage = Keyword.get(opts, :stage, @final_destination)

    frames =
      game_states
      |> Enum.with_index()
      |> Enum.map(fn {gs, idx} ->
        %Peppi.GameFrame{
          frame_number: gs.frame || idx,
          players:
            Map.new(gs.players, fn {port, player} ->
              {port, player_to_peppi_frame(player)}
            end)
        }
      end)

    %Peppi.ParsedReplay{
      frames: frames,
      metadata: %Peppi.ReplayMeta{
        path: "fixture.slp",
        stage: stage,
        duration_frames: length(frames),
        players: [
          %Peppi.PlayerMeta{port: 1, character: 10, character_name: "Mewtwo", tag: nil},
          %Peppi.PlayerMeta{port: 2, character: 2, character_name: "Fox", tag: nil}
        ]
      }
    }
  end

  defp player_to_peppi_frame(%Player{} = p) do
    %Peppi.PlayerFrame{
      character: p.character,
      x: p.x,
      y: p.y,
      percent: p.percent,
      stock: p.stock,
      facing: p.facing,
      action: p.action,
      action_frame: p.action_frame,
      invulnerable: p.invulnerable,
      jumps_left: p.jumps_left,
      on_ground: p.on_ground,
      shield_strength: p.shield_strength,
      hitstun_frames_left: p.hitstun_frames_left,
      speed_air_x_self: p.speed_air_x_self,
      speed_ground_x_self: p.speed_ground_x_self,
      speed_y_self: p.speed_y_self,
      speed_x_attack: p.speed_x_attack,
      speed_y_attack: p.speed_y_attack,
      controller: nil
    }
  end
end
