defmodule ExPhil.Embeddings.Nana do
  @moduledoc """
  Helper functions for Nana (Ice Climbers partner) embeddings.

  Nana embedding has three modes:
  - `:compact` - 39 dims with action category one-hot (preserves IC tech)
  - `:enhanced` - 14 dims continuous (action ID handled separately by learned embedding)
  - `:full` - ~455 dims (full player-like embedding)

  This module extracts common value extraction and IC tech flag computation
  shared across compact and enhanced modes.

  ## Ice Climbers Technical Mechanics

  IC tech (wobbling, handoffs, desyncs, chain grabs) requires knowing:
  - Nana's action state relative to Popo's
  - Whether Nana is attacking, grabbing, or can act
  - Sync state (same action = synced)

  These flags are critical for learning IC-specific techniques.

  ## See Also

  - `ExPhil.Embeddings.Player` - Main player embedding module
  - `ExPhil.Embeddings.Player.action_to_category/1` - Action category mapping
  """

  alias ExPhil.Bridge.Nana, as: NanaStruct
  alias ExPhil.Embeddings.Player, as: PlayerEmbed

  # Action category constants for IC tech detection
  # Reference: ExPhil.Embeddings.Player.action_to_category/1
  #
  # Attack categories: ground attacks (10, 11), aerials (12), specials (13-16)
  @attack_categories [10, 11, 12, 13, 14, 15, 16]
  # Grab (17) and throw (18) categories
  @grab_categories [17, 18]
  # Actionable states: idle (2), walk (3), dash/run (4), jump (6), fall (7), crouch (9)
  @actionable_categories [2, 3, 4, 6, 7, 9]

  @typedoc "Extracted Nana values from a batch of players"
  @type batch_values :: %{
          exists: [boolean()],
          xs: [float()],
          ys: [float()],
          facings: [boolean()],
          percents: [float()],
          stocks: [integer()],
          actions: [integer()]
        }

  @typedoc "Computed IC tech flags for a batch"
  @type batch_flags :: %{
          is_attacking: [boolean()],
          is_grabbing: [boolean()],
          can_act: [boolean()],
          is_synced: [boolean()],
          on_ground: [boolean()],
          categories: [integer()]
        }

  @doc """
  Extract Nana values from a batch of players.

  Returns a map with lists of values suitable for batch embedding.
  Nil Nanas get default values (0.0/false/0).

  ## Example

      iex> players = [%{nana: %{x: 10.0, y: 5.0}}, %{nana: nil}]
      iex> values = Nana.extract_batch_values(players)
      iex> values.xs
      [10.0, 0.0]
  """
  @spec extract_batch_values([map() | nil]) :: batch_values()
  def extract_batch_values(players) do
    nanas = Enum.map(players, fn p -> p && p.nana end)

    %{
      exists: Enum.map(nanas, fn n -> n != nil end),
      xs: Enum.map(nanas, fn n -> (n && n.x) || 0.0 end),
      ys: Enum.map(nanas, fn n -> (n && n.y) || 0.0 end),
      facings: Enum.map(nanas, fn n -> (n && n.facing) || false end),
      percents: Enum.map(nanas, fn n -> (n && n.percent) || 0.0 end),
      stocks: Enum.map(nanas, fn n -> (n && n.stock) || 0 end),
      actions: Enum.map(nanas, fn n -> (n && n.action) || 0 end)
    }
  end

  @doc """
  Compute IC tech flags for a batch of Nanas.

  Returns flags indicating combat state and sync with Popo.
  These flags are critical for learning IC-specific techniques.

  ## Parameters

  - `nana_values` - Map from `extract_batch_values/1`
  - `popo_actions` - List of Popo action IDs (same length as batch)
  - `opts` - Options:
    - `:sync_mode` - `:category` (default) or `:exact`
      - `:category` - synced if same action category (compact mode)
      - `:exact` - synced if exact same action ID (enhanced mode, more precise)

  ## Example

      iex> values = %{actions: [100, 200], ys: [0.0, 50.0]}
      iex> popo_actions = [100, 150]
      iex> flags = Nana.compute_batch_flags(values, popo_actions, sync_mode: :exact)
      iex> flags.is_synced
      [true, false]
  """
  @spec compute_batch_flags(batch_values(), [integer()], keyword()) :: batch_flags()
  def compute_batch_flags(nana_values, popo_actions, opts \\ []) do
    sync_mode = Keyword.get(opts, :sync_mode, :category)

    nana_actions = nana_values.actions
    nana_ys = nana_values.ys

    # Compute action categories
    nana_categories = Enum.map(nana_actions, &PlayerEmbed.action_to_category/1)
    popo_categories = Enum.map(popo_actions, &PlayerEmbed.action_to_category/1)

    # Compute sync based on mode
    is_synced =
      case sync_mode do
        :exact ->
          # Enhanced mode: compare exact actions for more precise sync detection
          Enum.zip(nana_actions, popo_actions)
          |> Enum.map(fn {n, p} -> n == p end)

        :category ->
          # Compact mode: compare action categories
          Enum.zip(nana_categories, popo_categories)
          |> Enum.map(fn {n, p} -> n == p end)
      end

    %{
      categories: nana_categories,
      is_attacking: Enum.map(nana_categories, fn c -> c in @attack_categories end),
      is_grabbing: Enum.map(nana_categories, fn c -> c in @grab_categories end),
      can_act: Enum.map(nana_categories, fn c -> c in @actionable_categories end),
      is_synced: is_synced,
      on_ground: Enum.map(nana_ys, fn y -> y < 5.0 end)
    }
  end

  @doc """
  Extract values from a single Nana struct.

  Returns a map with single values (not lists).
  Used for single-player embedding functions.
  """
  @spec extract_values(NanaStruct.t() | nil) :: map()
  def extract_values(nil) do
    %{
      exists: false,
      x: 0.0,
      y: 0.0,
      facing: false,
      percent: 0.0,
      stock: 0,
      action: 0
    }
  end

  def extract_values(%NanaStruct{} = nana) do
    %{
      exists: true,
      x: nana.x || 0.0,
      y: nana.y || 0.0,
      facing: nana.facing || false,
      percent: nana.percent || 0.0,
      stock: nana.stock || 0,
      action: nana.action || 0
    }
  end

  @doc """
  Compute IC tech flags for a single Nana.

  ## Parameters

  - `nana_action` - Nana's action ID
  - `popo_action` - Popo's action ID
  - `nana_y` - Nana's Y position (for ground detection)
  - `opts` - Options (same as `compute_batch_flags/3`)
  """
  @spec compute_flags(integer(), integer(), float(), keyword()) :: map()
  def compute_flags(nana_action, popo_action, nana_y, opts \\ []) do
    sync_mode = Keyword.get(opts, :sync_mode, :category)

    nana_category = PlayerEmbed.action_to_category(nana_action)
    popo_category = PlayerEmbed.action_to_category(popo_action)

    is_synced =
      case sync_mode do
        :exact -> nana_action == popo_action
        :category -> nana_category == popo_category
      end

    %{
      category: nana_category,
      is_attacking: nana_category in @attack_categories,
      is_grabbing: nana_category in @grab_categories,
      can_act: nana_category in @actionable_categories,
      is_synced: is_synced,
      on_ground: nana_y < 5.0
    }
  end

  @doc """
  Check if any Nana exists in a batch.

  Optimization: if no Nanas exist, we can return zero tensor directly.
  """
  @spec any_nana_exists?([map() | nil]) :: boolean()
  def any_nana_exists?(players) do
    Enum.any?(players, fn p -> p && p.nana != nil end)
  end

  @doc """
  Get Nana action ID for learned embedding.

  Returns the action ID for the learned action embedding layer.
  Returns 0 if Nana doesn't exist.
  """
  @spec get_action_id(map() | nil) :: integer()
  def get_action_id(nil), do: 0
  def get_action_id(%{nana: nil}), do: 0
  def get_action_id(%{nana: %{action: action}}) when is_integer(action), do: action
  def get_action_id(%{nana: _}), do: 0

  @doc """
  Number of action categories for one-hot encoding.

  Delegates to `ExPhil.Embeddings.Player` for the canonical value.
  """
  @spec num_categories() :: pos_integer()
  def num_categories, do: 25
end
