defmodule ExPhil.Bridge.SimState do
  @moduledoc """
  Maps a melee-sim-light `gamestate` row to `ExPhil.Bridge.GameState`, and an
  `ExPhil.Bridge.ControllerState` to the sim's controller row.

  This is the ExPhil half of the sim boundary (`docs/planning/SIM_INTEGRATION.md`).
  The sim owns Melee's rules; this module owns the id-space and field
  conventions the policy was trained on, so every choice here is pinned to
  the frame-level convention the Peppi NIF produces:

    * **character** — sim `char_id` is the game's INTERNAL fighter kind
      (Mario 0, Fox 1, …, Roy 26). The frame-level convention is identity on
      internal ids with unknowns clamped to 32 (`native/exphil_peppi/src/lib.rs`
      `internal_character_id/1`), so the mapping is the identity.
    * **stage** — sim `stage_id` is the EXTERNAL Slippi id (FD 32, BF 31, …),
      which is what `GameState.stage` carries (GOTCHA #96).
    * **action** — sim `action_id` is the GALE01 action-state id, the same
      space as Slippi post-frames and `Player.action`.
    * **facing** — sim `u1` (1 = right) becomes `1 | -1`.
    * **booleans** — `on_ground`, `invulnerable` are `u1` in the sim.
    * **ports** — a sim slot's `source_player` is 0-based; ports are 1-based.
      A second present slot with the same `source_player` is that player's
      follower (Nana) and is folded into `Player.nana`.

  The row is a plain map as decoded from the sim worker (string or atom
  keys; nested `slots`, `stage`, `items`). Field names are the sim's
  `melee_sim/dtypes.py` names verbatim, so a mismatch fails loudly on the
  key, not silently on a value.
  """

  alias ExPhil.Bridge.{ControllerState, GameState, Item, Nana, Player}

  @unknown_character 32
  @menu_in_game 2

  @doc """
  Build a `GameState` from a sim gamestate row.

  Options:
    * `:own_port` — the subject's port (default 1); used for `distance` and
      `own_port`.
    * `:opponent_port` — default: the lowest other occupied port.
  """
  @spec to_game_state(map(), keyword()) :: GameState.t()
  def to_game_state(row, opts \\ []) when is_map(row) do
    own_port = Keyword.get(opts, :own_port, 1)
    players = slots_to_players(fetch!(row, :slots))
    opponent_port = Keyword.get(opts, :opponent_port) || default_opponent(players, own_port)

    stage_block = get(row, :stage) || %{}
    fod = get(stage_block, :fod_platforms) || %{}

    %GameState{
      frame: fetch!(row, :frame_id),
      stage: fetch!(row, :stage_id),
      menu_state: @menu_in_game,
      players: players,
      own_port: own_port,
      projectiles: [],
      items: items(get(row, :items)),
      distance: distance(players[own_port], players[opponent_port]),
      fod_platform_left: get(fod, :left),
      fod_platform_right: get(fod, :right),
      stadium_event: nil,
      stadium_type: nil,
      whispy_direction: nil
    }
  end

  @doc """
  Convert a `ControllerState` to the sim's float controller row
  (`controller_player_dtype`).

  The sim's stick floats are `(raw + 80) / 160`: **[0, 1] with 0.5 neutral**
  (`melee_sim/controller.py` `neutral_controller`, `from_raw_axis`), not
  libmelee's [-1, 1]. Sending 0.0 for a neutral y-axis is a full crouch
  (found 2026-09-21 in the step-2 smoke: 1,709 crouching frames). Shoulder
  is `raw / 140` in [0, 1] already. Buttons are 0/1. The sim takes one
  shoulder value; we send the larger of L/R.
  """
  @spec controller_to_row(ControllerState.t()) :: map()
  def controller_to_row(%ControllerState{} = c) do
    %{
      buttons: %{
        A: b(c.button_a),
        B: b(c.button_b),
        X: b(c.button_x),
        Y: b(c.button_y),
        Z: b(c.button_z),
        L: b(c.button_l),
        R: b(c.button_r),
        D_UP: b(Map.get(c, :button_d_up))
      },
      main_stick_x: axis(c.main_stick.x),
      main_stick_y: axis(c.main_stick.y),
      c_stick_x: axis(c.c_stick.x),
      c_stick_y: axis(c.c_stick.y),
      shoulder: clamp01(max(c.l_shoulder || 0.0, c.r_shoulder || 0.0) * 1.0)
    }
  end

  @doc "libmelee stick axis [-1, 1] -> sim float axis [0, 1] (0.5 neutral)."
  @spec axis(number()) :: float()
  def axis(v) when is_number(v), do: clamp01(v / 2.0 + 0.5)

  defp clamp01(v) when v < 0.0, do: 0.0
  defp clamp01(v) when v > 1.0, do: 1.0
  defp clamp01(v), do: v * 1.0

  @doc "Sim internal character kind -> frame-level character id (identity, clamped)."
  @spec character_id(integer()) :: integer()
  def character_id(kind) when is_integer(kind) and kind >= 0 and kind <= 0x20, do: kind
  def character_id(_), do: @unknown_character

  # -- slots ------------------------------------------------------------------

  defp slots_to_players(slots) when is_list(slots) do
    present = Enum.filter(slots, &(truthy(get(&1, :present))))

    present
    |> Enum.group_by(&(get(&1, :source_player) + 1))
    |> Map.new(fn {port, [leader | followers]} ->
      {port, %{to_player(leader) | nana: followers |> List.first() |> to_nana()}}
    end)
  end

  defp to_player(s) do
    %Player{
      character: character_id(fetch!(s, :char_id)),
      x: f(fetch!(s, :pos_x)),
      y: f(fetch!(s, :pos_y)),
      percent: f(fetch!(s, :percent)),
      stock: fetch!(s, :stocks),
      facing: facing(fetch!(s, :facing)),
      action: fetch!(s, :action_id),
      action_frame: fetch!(s, :action_frame),
      invulnerable: truthy(fetch!(s, :invulnerable)),
      jumps_left: fetch!(s, :jumps_left),
      on_ground: truthy(fetch!(s, :on_ground)),
      shield_strength: f(fetch!(s, :shield_hp)),
      hitstun_frames_left: fetch!(s, :hitstun),
      speed_air_x_self: f(fetch!(s, :speed_air_x_self)),
      speed_ground_x_self: f(fetch!(s, :speed_ground_x_self)),
      speed_y_self: f(fetch!(s, :speed_y_self)),
      speed_x_attack: f(fetch!(s, :speed_x_attack)),
      speed_y_attack: f(fetch!(s, :speed_y_attack)),
      nana: nil,
      controller_state: nil,
      connect_code: "",
      nametag: ""
    }
  end

  defp to_nana(nil), do: nil

  defp to_nana(s) do
    %Nana{
      x: f(fetch!(s, :pos_x)),
      y: f(fetch!(s, :pos_y)),
      percent: f(fetch!(s, :percent)),
      stock: fetch!(s, :stocks),
      action: fetch!(s, :action_id),
      facing: facing(fetch!(s, :facing))
    }
  end

  defp items(nil), do: []

  defp items(list) when is_list(list) do
    for it <- list, truthy(get(it, :exists)) do
      %Item{
        x: f(get(it, :pos_x)),
        y: f(get(it, :pos_y)),
        type: get(it, :type),
        facing: facing(get(it, :direction)),
        owner: get(it, :owner),
        held_by: nil,
        spawn_id: get(it, :spawn_id),
        timer: get(it, :timer)
      }
    end
  end

  # -- helpers ----------------------------------------------------------------

  defp default_opponent(players, own_port) do
    players |> Map.keys() |> Enum.reject(&(&1 == own_port)) |> Enum.min(fn -> nil end)
  end

  defp distance(%Player{} = a, %Player{} = b) do
    dx = a.x - b.x
    dy = a.y - b.y
    :math.sqrt(dx * dx + dy * dy)
  end

  defp distance(_, _), do: 0.0

  # sim facing: u1 (1 = right, 0 = left) on players; items carry a float direction
  defp facing(v) when is_number(v) and v > 0, do: 1
  defp facing(_), do: -1

  defp truthy(v) when is_boolean(v), do: v
  defp truthy(v) when is_integer(v), do: v != 0
  defp truthy(_), do: false

  defp b(true), do: 1
  defp b(_), do: 0

  defp f(nil), do: nil
  defp f(v) when is_number(v), do: v * 1.0

  defp get(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, v} -> v
      :error -> Map.get(map, Atom.to_string(key))
    end
  end

  defp get(_, _), do: nil

  defp fetch!(map, key) do
    case get(map, key) do
      nil -> raise KeyError, key: key, term: map
      v -> v
    end
  end
end
