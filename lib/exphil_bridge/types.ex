defmodule ExPhil.Bridge.GameState do
  @moduledoc """
  Represents the full game state from libmelee.

  This struct mirrors the data from `melee.GameState` in Python.
  """

  @type t :: %__MODULE__{
    frame: integer(),
    stage: integer(),
    menu_state: integer(),
    players: %{integer() => ExPhil.Bridge.Player.t()},
    projectiles: [ExPhil.Bridge.Projectile.t()],
    items: [ExPhil.Bridge.Item.t()],
    distance: float()
  }

  defstruct [
    :frame,
    :stage,
    :menu_state,
    :players,
    :projectiles,
    :items,
    :distance
  ]

  @doc """
  Check if the game is in an active game state (not in menus).
  """
  def in_game?(%__MODULE__{menu_state: menu_state}) do
    # Menu.IN_GAME = 2, Menu.SUDDEN_DEATH = 3
    menu_state in [2, 3]
  end

  @doc """
  Get player by port number.
  """
  def get_player(%__MODULE__{players: players}, port) when is_integer(port) do
    Map.get(players, port)
  end

  @doc """
  Get both players as {p1, p2} tuple.
  """
  def get_players(%__MODULE__{players: players}) do
    {Map.get(players, 1), Map.get(players, 2)}
  end
end

defmodule ExPhil.Bridge.Player do
  @moduledoc """
  Represents a player's state from libmelee.

  Mirrors `melee.PlayerState` from Python.
  """

  @type t :: %__MODULE__{
    character: integer(),
    x: float(),
    y: float(),
    percent: float(),
    stock: integer(),
    facing: integer(),
    action: integer(),
    action_frame: integer(),
    invulnerable: boolean(),
    jumps_left: integer(),
    on_ground: boolean(),
    shield_strength: float(),
    hitstun_frames_left: integer(),
    speed_air_x_self: float(),
    speed_ground_x_self: float(),
    speed_y_self: float(),
    speed_x_attack: float(),
    speed_y_attack: float(),
    nana: ExPhil.Bridge.Nana.t() | nil,
    controller_state: ExPhil.Bridge.ControllerState.t() | nil
  }

  defstruct [
    :character,
    :x,
    :y,
    :percent,
    :stock,
    :facing,
    :action,
    :action_frame,
    :invulnerable,
    :jumps_left,
    :on_ground,
    :shield_strength,
    :hitstun_frames_left,
    :speed_air_x_self,
    :speed_ground_x_self,
    :speed_y_self,
    :speed_x_attack,
    :speed_y_attack,
    :nana,
    :controller_state
  ]

  @doc """
  Check if player is currently dying.
  """
  def dying?(%__MODULE__{action: action}) when is_integer(action) do
    action <= 0x0A
  end
  def dying?(_), do: false

  @doc """
  Check if player is offstage.
  """
  def offstage?(%__MODULE__{x: x, y: y}, stage_edge_x \\ 85.0) do
    abs(x) > stage_edge_x or y < 0
  end

  @doc """
  Check if player is in hitstun.
  """
  def in_hitstun?(%__MODULE__{hitstun_frames_left: frames}) when is_integer(frames) do
    frames > 0
  end
  def in_hitstun?(_), do: false

  @doc """
  Check if player is shielding.
  """
  def shielding?(%__MODULE__{action: action}) when is_integer(action) do
    # Action.SHIELD_START = 178, SHIELD = 179, SHIELD_RELEASE = 180
    action in [178, 179, 180]
  end
  def shielding?(_), do: false
end

defmodule ExPhil.Bridge.Nana do
  @moduledoc """
  Represents Nana (Ice Climbers partner) state.
  """

  @type t :: %__MODULE__{
    x: float(),
    y: float(),
    percent: float(),
    stock: integer(),
    action: integer(),
    facing: integer()
  }

  defstruct [:x, :y, :percent, :stock, :action, :facing]
end

defmodule ExPhil.Bridge.ControllerState do
  @moduledoc """
  Represents the controller state for a player.

  Used for imitation learning - captures what buttons/sticks
  a player is pressing at a given frame.
  """

  @type t :: %__MODULE__{
    main_stick: %{x: float(), y: float()},
    c_stick: %{x: float(), y: float()},
    l_shoulder: float(),
    r_shoulder: float(),
    button_a: boolean(),
    button_b: boolean(),
    button_x: boolean(),
    button_y: boolean(),
    button_z: boolean(),
    button_l: boolean(),
    button_r: boolean(),
    button_d_up: boolean()
  }

  defstruct [
    :main_stick,
    :c_stick,
    :l_shoulder,
    :r_shoulder,
    :button_a,
    :button_b,
    :button_x,
    :button_y,
    :button_z,
    :button_l,
    :button_r,
    :button_d_up
  ]

  @doc """
  Create a neutral controller state (no buttons pressed, sticks centered).
  """
  def neutral do
    %__MODULE__{
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
  Convert to input format expected by MeleePort.send_controller/2.
  """
  def to_input(%__MODULE__{} = cs) do
    %{
      main_stick: cs.main_stick,
      c_stick: cs.c_stick,
      shoulder: cs.l_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end
end

defmodule ExPhil.Bridge.Projectile do
  @moduledoc """
  Represents a projectile in the game (fireballs, arrows, bombs, etc.).
  """

  @type t :: %__MODULE__{
    owner: integer(),
    x: float(),
    y: float(),
    type: integer(),
    subtype: integer(),
    speed_x: float(),
    speed_y: float()
  }

  defstruct [:owner, :x, :y, :type, :subtype, :speed_x, :speed_y]
end

defmodule ExPhil.Bridge.Item do
  @moduledoc """
  Represents an item in the game (bombs, capsules, barrels, etc.).

  Items differ from projectiles:
  - Items can be held, thrown, or on the ground
  - Items have a spawn timer and lifetime
  - Items can damage their owner (e.g., Link's bombs)

  ## Item Types (common ones for target characters)

  Link-specific:
  - Bomb: Can be pulled, held, thrown, and deals self-damage

  Common items:
  - Capsule, Barrel, Crate: Container items
  - Beam Sword, Bat, Fan: Melee weapons
  - Ray Gun, Super Scope: Projectile weapons
  """

  @type t :: %__MODULE__{
    x: float(),
    y: float(),
    type: integer(),
    facing: integer(),
    owner: integer() | nil,
    held_by: integer() | nil,
    spawn_id: integer(),
    timer: integer()
  }

  defstruct [
    :x,
    :y,
    :type,
    :facing,
    :owner,
    :held_by,
    :spawn_id,
    :timer
  ]

  # Common item type constants (from libmelee)
  @item_types %{
    # Explosives/Bombs
    link_bomb: 0x2C,         # Link's bomb
    young_link_bomb: 0x2D,   # Young Link's bomb
    bob_omb: 0x13,           # Bob-omb

    # Character-specific thrown items
    peach_turnip: 0x32,      # Peach's turnip (50)
    mr_saturn: 0x15,         # Mr. Saturn (21)

    # Melee weapons
    beam_sword: 0x04,        # Beam Sword
    home_run_bat: 0x05,      # Home Run Bat
    fan: 0x06,               # Fan

    # Ranged weapons
    ray_gun: 0x08,           # Ray Gun
    super_scope: 0x09,       # Super Scope

    # Containers
    capsule: 0x00,           # Capsule
    crate: 0x01,             # Crate
    barrel: 0x02             # Barrel
  }

  @doc """
  Check if item is a bomb (Link/Young Link).
  """
  def bomb?(%__MODULE__{type: type}) do
    type in [@item_types.link_bomb, @item_types.young_link_bomb, @item_types.bob_omb]
  end
  def bomb?(_), do: false

  @doc """
  Check if item is a Peach turnip.
  """
  def turnip?(%__MODULE__{type: type}), do: type == @item_types.peach_turnip
  def turnip?(_), do: false

  @doc """
  Check if item is Mr. Saturn.
  """
  def mr_saturn?(%__MODULE__{type: type}), do: type == @item_types.mr_saturn
  def mr_saturn?(_), do: false

  @doc """
  Check if item is currently held by a player.
  """
  def held?(%__MODULE__{held_by: held_by}) when is_integer(held_by) and held_by > 0, do: true
  def held?(_), do: false

  @doc """
  Check if item belongs to given player.
  """
  def owned_by?(%__MODULE__{owner: owner}, player_port) when is_integer(player_port) do
    owner == player_port
  end
  def owned_by?(_, _), do: false

  @doc """
  Get normalized item type for embedding.

  Returns a simplified category:
  - 0: None/unknown
  - 1: Bomb (Link, Young Link, Bob-omb)
  - 2: Melee weapon (Beam Sword, Bat, Fan)
  - 3: Ranged weapon (Ray Gun, Super Scope)
  - 4: Container (Capsule, Crate, Barrel)
  - 5: Thrown/Character-specific (Peach turnip, Mr. Saturn, etc.)
  """
  def item_category(%__MODULE__{type: type}) do
    cond do
      type in [@item_types.link_bomb, @item_types.young_link_bomb, @item_types.bob_omb] -> 1
      type in [@item_types.beam_sword, @item_types.home_run_bat, @item_types.fan] -> 2
      type in [@item_types.ray_gun, @item_types.super_scope] -> 3
      type in [@item_types.capsule, @item_types.crate, @item_types.barrel] -> 4
      type in [@item_types.peach_turnip, @item_types.mr_saturn] -> 5
      true -> 5  # Unknown items fall into "other/thrown" category
    end
  end
  def item_category(_), do: 0
end

defmodule ExPhil.Bridge.ControllerInput do
  @moduledoc """
  Helper module for creating controller inputs to send to the game.
  """

  @doc """
  Create a neutral input (no buttons, sticks centered).
  """
  def neutral do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{}
    }
  end

  @doc """
  Create input with main stick direction.
  """
  def main_stick(x, y) when is_number(x) and is_number(y) do
    %{neutral() | main_stick: %{x: clamp(x), y: clamp(y)}}
  end

  @doc """
  Create input with a button press.
  """
  def button(button) when button in [:a, :b, :x, :y, :z, :l, :r, :d_up] do
    %{neutral() | buttons: %{button => true}}
  end

  @doc """
  Combine multiple inputs (later inputs override earlier ones).
  """
  def combine(inputs) when is_list(inputs) do
    Enum.reduce(inputs, neutral(), fn input, acc ->
      acc
      |> Map.merge(input, fn
        :buttons, b1, b2 -> Map.merge(b1, b2)
        _key, _v1, v2 -> v2
      end)
    end)
  end

  # Common moves

  @doc "Jump (X button)"
  def jump, do: button(:x)

  @doc "Short hop (tap Y quickly)"
  def short_hop, do: button(:y)

  @doc "A attack"
  def a, do: button(:a)

  @doc "B special"
  def b, do: button(:b)

  @doc "Shield (L button)"
  def shield, do: %{neutral() | shoulder: 1.0, buttons: %{l: true}}

  @doc "Grab (Z button)"
  def grab, do: button(:z)

  @doc "Move left"
  def left, do: main_stick(0.0, 0.5)

  @doc "Move right"
  def right, do: main_stick(1.0, 0.5)

  @doc "Crouch"
  def crouch, do: main_stick(0.5, 0.0)

  @doc "Up tilt"
  def up_tilt, do: combine([main_stick(0.5, 0.7), button(:a)])

  @doc "Down tilt"
  def down_tilt, do: combine([main_stick(0.5, 0.3), button(:a)])

  @doc "Forward tilt (right)"
  def forward_tilt_right, do: combine([main_stick(0.7, 0.5), button(:a)])

  @doc "Forward tilt (left)"
  def forward_tilt_left, do: combine([main_stick(0.3, 0.5), button(:a)])

  @doc "Up smash"
  def up_smash, do: %{neutral() | c_stick: %{x: 0.5, y: 1.0}}

  @doc "Down smash"
  def down_smash, do: %{neutral() | c_stick: %{x: 0.5, y: 0.0}}

  @doc "Forward smash (right)"
  def forward_smash_right, do: %{neutral() | c_stick: %{x: 1.0, y: 0.5}}

  @doc "Forward smash (left)"
  def forward_smash_left, do: %{neutral() | c_stick: %{x: 0.0, y: 0.5}}

  defp clamp(v), do: max(0.0, min(1.0, v))
end
