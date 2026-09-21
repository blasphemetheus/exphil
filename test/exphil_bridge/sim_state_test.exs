defmodule ExPhil.Bridge.SimStateTest do
  @moduledoc """
  Sim-row -> GameState mapping (docs/planning/SIM_INTEGRATION.md, step 1).

  The row literals mirror melee-sim-light's `melee_sim/dtypes.py` field
  names; the expected structs mirror what `ExPhil.Data.Peppi.build_player/1`
  produces for the same frame, so the policy sees one convention from both
  sources.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.{ControllerState, GameState, Nana, Player, SimState}
  alias ExPhil.Embeddings.Game, as: GameEmbed

  # Fox (internal kind 1) vs Fox on FD (external 32), first playable frame.
  defp fox_slot(source_player, x, facing) do
    %{
      "present" => 1, "source_player" => source_player, "team_relation" => 0, "team_id" => source_player,
      "pos_x" => x, "pos_y" => 0.0,
      "speed_air_x_self" => 0.0, "speed_ground_x_self" => 0.0, "speed_y_self" => 0.0,
      "speed_x_attack" => 0.0, "speed_y_attack" => 0.0,
      "percent" => 0.0, "shield_hp" => 60.0,
      "action_id" => 322, "action_frame" => 1, "hitlag" => 0, "hitstun" => 0,
      "char_id" => 1, "stocks" => 4, "facing" => facing, "on_ground" => 1,
      "jumps_left" => 2, "hurtbox_state" => 0, "invulnerable" => 0
    }
  end

  defp empty_slot, do: %{"present" => 0}

  defp fd_row do
    %{
      "frame_id" => -123, "frame_pre_random_seed" => 0, "stage_id" => 32,
      "num_players" => 2, "viewpoint_player" => 0, "is_teams" => 0,
      "stage" => %{"randall" => %{"exists" => 0}, "fod_platforms" => %{"left" => 0.0, "right" => 0.0}},
      "slots" => [fox_slot(0, -40.0, 1), fox_slot(1, 40.0, 0), empty_slot(), empty_slot()],
      "items" => []
    }
  end

  defp expected_player(x, facing) do
    %Player{
      character: 1, x: x, y: 0.0, percent: 0.0, stock: 4, facing: facing,
      action: 322, action_frame: 1, invulnerable: false, jumps_left: 2, on_ground: true,
      shield_strength: 60.0, hitstun_frames_left: 0,
      speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0,
      speed_x_attack: 0.0, speed_y_attack: 0.0,
      nana: nil, controller_state: nil, connect_code: "", nametag: ""
    }
  end

  test "maps a Fox ditto FD row field-for-field to the Peppi convention" do
    gs = SimState.to_game_state(fd_row(), own_port: 1)

    assert %GameState{frame: -124, stage: 32, menu_state: 2, own_port: 1, projectiles: [], items: []} = gs
    assert gs.players[1] == expected_player(-40.0, 1)
    assert gs.players[2] == expected_player(40.0, -1)
    assert_in_delta gs.distance, 80.0, 1.0e-9
    assert gs.fod_platform_left == 0.0
  end

  test "atom keys work the same as string keys" do
    row = fd_row() |> Jason.encode!() |> Jason.decode!(keys: :atoms)
    assert SimState.to_game_state(row) == SimState.to_game_state(fd_row())
  end

  test "a missing field fails loudly on its key" do
    bad = update_in(fd_row(), ["slots"], fn [a | rest] -> [Map.delete(a, "shield_hp") | rest] end)
    assert_raise KeyError, ~r/shield_hp/, fn -> SimState.to_game_state(bad) end
  end

  test "a second present slot for the same player folds into nana" do
    nana = fox_slot(0, -35.0, 1) |> Map.put("char_id", 11)
    row = put_in(fd_row(), ["slots"], [fox_slot(0, -40.0, 1), fox_slot(1, 40.0, 0), nana, empty_slot()])
    gs = SimState.to_game_state(row)
    assert %Nana{x: -35.0, action: 322, facing: 1, stock: 4} = gs.players[1].nana
    assert map_size(gs.players) == 2
  end

  test "character ids are the internal identity, clamped" do
    assert SimState.character_id(1) == 1
    assert SimState.character_id(16) == 16
    assert SimState.character_id(26) == 26
    assert SimState.character_id(0x21) == 32
  end

  test "controller round-trips to the sim's float row" do
    c = %ControllerState{
      main_stick: %{x: 0.5, y: -1.0}, c_stick: %{x: 0.0, y: 0.0},
      l_shoulder: 0.0, r_shoulder: 0.7,
      button_a: false, button_b: true, button_x: false, button_y: true,
      button_z: false, button_l: false, button_r: false, button_d_up: false
    }
    row = SimState.controller_to_row(c)
    assert row.buttons == %{A: 0, B: 1, X: 0, Y: 1, Z: 0, L: 0, R: 0, D_UP: 0}
    # sim axes are [0, 1] with 0.5 neutral: +0.5 -> 0.75, -1.0 -> 0.0, 0.0 -> 0.5
    assert row.main_stick_x == 0.75 and row.main_stick_y == 0.0
    assert row.c_stick_x == 0.5 and row.c_stick_y == 0.5
    assert row.shoulder == 0.7
    assert SimState.axis(1.0) == 1.0 and SimState.axis(-1.0) == 0.0 and SimState.axis(0.0) == 0.5
  end

  test "the mapped state embeds identically to the same state built directly" do
    mapped = SimState.to_game_state(fd_row(), own_port: 1)

    direct = %GameState{
      frame: -124, stage: 32, menu_state: 2, own_port: 1, projectiles: [], items: [],
      players: %{1 => expected_player(-40.0, 1), 2 => expected_player(40.0, -1)},
      distance: 80.0, fod_platform_left: 0.0, fod_platform_right: 0.0
    }

    a = GameEmbed.embed(mapped, nil, 1) |> Nx.backend_copy(Nx.BinaryBackend)
    b = GameEmbed.embed(direct, nil, 1) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.to_binary(a) == Nx.to_binary(b)
  end
end
