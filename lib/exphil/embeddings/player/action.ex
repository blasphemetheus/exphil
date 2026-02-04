defmodule ExPhil.Embeddings.Player.Action do
  @moduledoc """
  Action state categorization for Melee players.

  Maps Melee's 399 action states into 25 meaningful categories for compact
  embedding. Essential for Ice Climbers tech detection (handoffs, regrabs,
  desyncs).

  ## Action Categories (25 total)

  | ID | Category | Action Range | Description |
  |----|----------|--------------|-------------|
  | 0 | DEAD | 0x00-0x0A | Dead, respawning |
  | 1 | ENTRY | 0x0B-0x0D | Entry, rebirth |
  | 2 | IDLE | 0x0E-0x14 | Standing, waiting |
  | 3 | WALK | 0x15-0x18 | Walking |
  | 4 | DASH_RUN | 0x19-0x1E | Dashing, running, turn |
  | 5 | JUMP_SQUAT | 0x1F-0x22 | Jump startup |
  | 6 | JUMP_AERIAL | 0x23-0x2C | Jumping, double jump |
  | 7 | FALL | 0x2D-0x36 | Falling, fast fall |
  | 8 | LAND | 0x37-0x3C | Landing, landing lag |
  | 9 | CROUCH | 0x3D-0x42 | Crouching |
  | 10 | ATTACK_GROUND | 0x43-0x54 | Jab, tilts |
  | 11 | ATTACK_SMASH | 0x55-0x60 | Smash attacks |
  | 12 | ATTACK_AIR | 0x61-0x6B | Aerials |
  | 13 | SPECIAL_N | 0x6C-0x7F | Neutral B (IC: Ice Shot) |
  | 14 | SPECIAL_S | 0x80-0x93 | Side B (IC: Squall Hammer) |
  | 15 | SPECIAL_U | 0x94-0xA7 | Up B (IC: Belay) |
  | 16 | SPECIAL_D | 0xA8-0xBB | Down B (IC: Blizzard) |
  | 17 | GRAB | 0xBC-0xC7 | Grabbing, pummel |
  | 18 | THROW | 0xC8-0xD3 | Throws |
  | 19 | GRABBED | 0xD4-0xDF | Being grabbed |
  | 20 | SHIELD | 0xE0-0xED | Shielding |
  | 21 | DODGE | 0xEE-0xFF | Roll, spotdodge, airdodge |
  | 22 | DAMAGE | 0x100-0x130 | Hitstun |
  | 23 | DOWN_TECH | 0x131-0x160 | Lying down, getup, tech |
  | 24 | LEDGE | 0x161+ | Ledge grab, ledge actions |

  ## IC Tech Detection

  These categories enable detection of:
  - **Desyncs**: Popo and Nana in different categories
  - **Handoffs**: GRAB (17) + partner attacking
  - **Regrabs**: THROW (18) + partner grabbing
  - **Wobbling**: GRAB (17) sustained

  ## See Also

  - `ExPhil.Embeddings.Nana` - Uses categories for IC tech flags
  - `ExPhil.Embeddings.Player` - Main player embedding
  """

  @doc """
  Number of action categories.
  """
  @spec num_categories() :: non_neg_integer()
  def num_categories, do: 25

  @doc """
  Map a Melee action state ID to its category (0-24).

  ## Examples

      iex> action_to_category(0)   # Dead
      0

      iex> action_to_category(0x14) # Idle/Wait
      2

      iex> action_to_category(0xBC) # Grab
      17

      iex> action_to_category(0x161) # Ledge
      24

  """
  @spec to_category(integer()) :: non_neg_integer()
  def to_category(action) when is_integer(action) do
    cond do
      # 0: DEAD - Dead, respawning (0x00-0x0A)
      action <= 0x0A -> 0
      # 1: ENTRY - Entry, rebirth (0x0B-0x0D)
      action <= 0x0D -> 1
      # 2: IDLE - Standing, waiting (0x0E-0x14)
      action <= 0x14 -> 2
      # 3: WALK - Walking (0x15-0x18)
      action <= 0x18 -> 3
      # 4: DASH_RUN - Dashing, running, turn (0x19-0x1E)
      action <= 0x1E -> 4
      # 5: JUMP_SQUAT - Jump startup (0x1F-0x22)
      action <= 0x22 -> 5
      # 6: JUMP_AERIAL - Jumping, double jump (0x23-0x2C)
      action <= 0x2C -> 6
      # 7: FALL - Falling, fast fall (0x2D-0x36)
      action <= 0x36 -> 7
      # 8: LAND - Landing, landing lag (0x37-0x3C)
      action <= 0x3C -> 8
      # 9: CROUCH - Crouching (0x3D-0x42)
      action <= 0x42 -> 9
      # 10: ATTACK_GROUND - Jab, tilts (0x43-0x54)
      action <= 0x54 -> 10
      # 11: ATTACK_SMASH - Smash attacks (0x55-0x60)
      action <= 0x60 -> 11
      # 12: ATTACK_AIR - Aerials: nair, fair, bair, uair, dair (0x61-0x6B)
      action <= 0x6B -> 12
      # 13: SPECIAL_N - Neutral special (0x6C-0x7F) - IC: Ice Shot
      action <= 0x7F -> 13
      # 14: SPECIAL_S - Side special (0x80-0x93) - IC: Squall Hammer
      action <= 0x93 -> 14
      # 15: SPECIAL_U - Up special (0x94-0xA7) - IC: Belay
      action <= 0xA7 -> 15
      # 16: SPECIAL_D - Down special (0xA8-0xBB) - IC: Blizzard
      action <= 0xBB -> 16
      # 17: GRAB - Grabbing, pummel (0xBC-0xC7)
      action <= 0xC7 -> 17
      # 18: THROW - Throws: fthrow, bthrow, uthrow, dthrow (0xC8-0xD3)
      action <= 0xD3 -> 18
      # 19: GRABBED - Being grabbed, pummeled (0xD4-0xDF)
      action <= 0xDF -> 19
      # 20: SHIELD - Shielding, shield stun (0xE0-0xED)
      action <= 0xED -> 20
      # 21: DODGE - Roll, spotdodge, airdodge (0xEE-0xFF)
      action <= 0xFF -> 21
      # 22: DAMAGE - Hitstun, knockback (0x100-0x130)
      action <= 0x130 -> 22
      # 23: DOWN_TECH - Lying down, getup, tech (0x131-0x160)
      action <= 0x160 -> 23
      # 24: LEDGE - Ledge grab, ledge actions, edge (0x161+)
      true -> 24
    end
  end

  # Default to DEAD for nil/invalid
  def to_category(_), do: 0

  @doc """
  Check if an action category is an attacking state (10-12: ground, smash, air).
  """
  @spec attacking?(non_neg_integer()) :: boolean()
  def attacking?(category) when category in 10..12, do: true
  def attacking?(_), do: false

  @doc """
  Check if an action category is a grabbing state (17: grab).
  """
  @spec grabbing?(non_neg_integer()) :: boolean()
  def grabbing?(17), do: true
  def grabbing?(_), do: false

  @doc """
  Check if an action category allows the player to act (idle-like states).

  Includes: IDLE (2), WALK (3), DASH_RUN (4), CROUCH (9).
  """
  @spec can_act?(non_neg_integer()) :: boolean()
  def can_act?(category) when category in [2, 3, 4, 9], do: true
  def can_act?(_), do: false

  @doc """
  Get the category name as an atom for debugging/display.
  """
  @spec category_name(non_neg_integer()) :: atom()
  def category_name(0), do: :dead
  def category_name(1), do: :entry
  def category_name(2), do: :idle
  def category_name(3), do: :walk
  def category_name(4), do: :dash_run
  def category_name(5), do: :jump_squat
  def category_name(6), do: :jump_aerial
  def category_name(7), do: :fall
  def category_name(8), do: :land
  def category_name(9), do: :crouch
  def category_name(10), do: :attack_ground
  def category_name(11), do: :attack_smash
  def category_name(12), do: :attack_air
  def category_name(13), do: :special_n
  def category_name(14), do: :special_s
  def category_name(15), do: :special_u
  def category_name(16), do: :special_d
  def category_name(17), do: :grab
  def category_name(18), do: :throw
  def category_name(19), do: :grabbed
  def category_name(20), do: :shield
  def category_name(21), do: :dodge
  def category_name(22), do: :damage
  def category_name(23), do: :down_tech
  def category_name(24), do: :ledge
  def category_name(_), do: :unknown
end
