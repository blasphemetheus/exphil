defmodule ExPhil.Embeddings.Player do
  @moduledoc """
  Player state embedding - converts player data to Nx tensors.

  A player's state includes position, action, damage, character, and more.
  This module converts all these values to a single concatenated tensor
  suitable for neural network input.

  ## Embedding Structure

  The default player embedding contains (in order):
  1. Percent (1 dim) - Damage percentage, scaled
  2. Facing (1 dim) - Direction facing (-1 left, +1 right)
  3. X position (1 dim) - Horizontal position, scaled
  4. Y position (1 dim) - Vertical position, scaled
  5. Action (399 dims) - One-hot action state
  6. Character (33 dims) - One-hot character
  7. Invulnerable (1 dim) - Boolean
  8. Jumps left (7 dims) - One-hot remaining jumps
  9. Shield strength (1 dim) - Current shield HP, scaled
  10. On ground (1 dim) - Boolean

  Optional additions:
  - Speed values (5 dims) - Various velocity components
  - Nana (Ice Climbers partner) - Same structure as above

  Total base size: ~447 dimensions
  """

  # Uses Nx with module prefix
  alias ExPhil.Constants
  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Embeddings.Nana, as: NanaEmbed
  alias ExPhil.Embeddings.Player.{Action, Ids}
  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Bridge.Nana

  @doc """
  Configuration for player embedding.
  """
  defstruct xy_scale: 0.05,
            shield_scale: 0.01,
            speed_scale: 0.5,
            # Enabled for momentum/velocity info (+5 dims per player)
            with_speeds: true,
            with_nana: true,
            # :full, :compact (39 dims), or :enhanced (14 dims + action ID)
            nana_mode: :compact,
            # Hitstun frames + action frame (+2 dims per player)
            with_frame_info: true,
            # Stock count (+1 dim per player)
            with_stock: true,
            # Distance to nearest ledge (+1 dim per player)
            with_ledge_distance: true,
            # Use 1-dim normalized float instead of 7-dim one-hot (saves 6 dims/player)
            jumps_normalized: true,
            # :one_hot (399 dims) or :learned (action embedded in network, 0 dims here)
            action_mode: :learned,
            # :one_hot (33 dims) or :learned (character embedded in network, 0 dims here)
            character_mode: :learned

  @type nana_mode :: :full | :compact | :enhanced
  @type action_mode :: :one_hot | :learned
  @type character_mode :: :one_hot | :learned

  @type config :: %__MODULE__{
          xy_scale: float(),
          shield_scale: float(),
          speed_scale: float(),
          with_speeds: boolean(),
          with_nana: boolean(),
          nana_mode: nana_mode(),
          with_frame_info: boolean(),
          with_stock: boolean(),
          with_ledge_distance: boolean(),
          jumps_normalized: boolean(),
          action_mode: action_mode(),
          character_mode: character_mode()
        }

  # ==========================================================================
  # Action Categories (delegated to Action module)
  # ==========================================================================

  @nana_action_categories Action.num_categories()

  @doc """
  Map a Melee action state ID to its category (0-24).

  Delegates to `ExPhil.Embeddings.Player.Action.to_category/1`.
  See that module for the full category mapping.
  """
  defdelegate action_to_category(action), to: Action, as: :to_category

  @doc """
  Default player embedding configuration.
  """
  @spec default_config() :: config()
  def default_config, do: %__MODULE__{}

  @doc """
  Calculate the embedding size for a given configuration.
  """
  @spec embedding_size(config()) :: non_neg_integer()
  def embedding_size(config \\ default_config()) do
    base_size = base_embedding_size(config)

    speed_size = if config.with_speeds, do: 5, else: 0

    # Hitstun frames remaining + action frame (how far into animation)
    frame_info_size = if config.with_frame_info, do: 2, else: 0

    # Stock count (lives remaining)
    stock_size = if config.with_stock, do: 1, else: 0

    # Distance to nearest ledge
    ledge_size = if config.with_ledge_distance, do: 1, else: 0

    nana_size =
      if config.with_nana do
        case config.nana_mode do
          :compact ->
            compact_nana_embedding_size()

          :enhanced ->
            enhanced_compact_nana_embedding_size()

          :full ->
            # Nana has base embedding + exists flag + optional speeds + optional frame info + stock
            nana_base = base_embedding_size(config) + 1
            nana_speeds = if config.with_speeds, do: 5, else: 0
            nana_frames = if config.with_frame_info, do: 2, else: 0
            nana_stock = if config.with_stock, do: 1, else: 0
            # Nana doesn't need ledge_distance (shares with Popo)
            nana_base + nana_speeds + nana_frames + nana_stock
        end
      else
        0
      end

    base_size + speed_size + frame_info_size + stock_size + ledge_size + nana_size
  end

  # Compact Nana embedding size (~39 dims instead of 455)
  # Preserves all info needed for IC tech: handoffs, regrabs, desyncs
  defp compact_nana_embedding_size do
    # exists
    # x, y position
    # facing
    # on_ground
    # percent
    # stock
    # hitstun_frames (normalized)
    # action_frame (normalized)
    # invulnerable
    # action category one-hot (25)
    # is_attacking (boolean)
    # is_grabbing (boolean)
    # can_act (boolean)
    # is_synced_hint (boolean - same action category as Popo)
    1 +
      2 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      @nana_action_categories +
      1 +
      1 +
      1 +
      1
  end

  # Enhanced Compact Nana embedding size (14 dims continuous)
  # Uses action ID for learned embedding instead of 25-dim category one-hot
  # Critical for precise IC tech: dair vs fair timing, exact action frames
  # Note: Nana action ID is appended separately by GameEmbed
  defp enhanced_compact_nana_embedding_size do
    # exists
    # x, y position (scaled)
    # facing
    # on_ground
    # percent (scaled)
    # stock (normalized)
    # hitstun_frames (normalized, REAL data when available)
    # action_frame (normalized, REAL data - critical for hitbox timing!)
    # invulnerable
    # IC tech flags (keep these for quick boolean checks)
    # is_attacking (boolean)
    # is_grabbing (boolean)
    # can_act (boolean)
    # is_synced_hint (boolean - same action as Popo)
    1 +
      2 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1 +
      1

    # = 14 dims total (action ID handled separately)
  end

  defp base_embedding_size(config) do
    jumps_size = if config.jumps_normalized, do: 1, else: Primitives.embedding_size(:jumps_left)

    # Action: 399 dims for one-hot, 0 dims for learned (handled by network)
    action_size =
      case config.action_mode do
        :one_hot -> Primitives.embedding_size(:action)
        :learned -> 0
      end

    # Character: 33 dims for one-hot, 0 dims for learned (handled by network)
    character_size =
      case config.character_mode do
        :one_hot -> Primitives.embedding_size(:character)
        :learned -> 0
      end

    # percent
    # facing
    # x
    # y
    # invulnerable
    # shield
    # on_ground
    1 +
      1 +
      1 +
      1 +
      action_size +
      character_size +
      1 +
      jumps_size +
      1 +
      1
  end

  @doc """
  Embed a player state into an Nx tensor.

  ## Examples

      iex> player = %ExPhil.Bridge.Player{x: 0.0, y: 0.0, percent: 50.0, ...}
      iex> embed(player)
      #Nx.Tensor<f32[447]>

  """
  @spec embed(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed(player, config \\ default_config())

  def embed(nil, config) do
    # Return zeros for missing player
    Nx.broadcast(0.0, {embedding_size(config)})
  end

  def embed(%PlayerState{} = player, config) do
    embeddings = [
      embed_base(player, config)
    ]

    embeddings =
      if config.with_speeds do
        [embed_speeds(player, config) | embeddings]
      else
        embeddings
      end

    embeddings =
      if config.with_frame_info do
        [embed_frame_info(player) | embeddings]
      else
        embeddings
      end

    embeddings =
      if config.with_stock do
        [embed_stock(player) | embeddings]
      else
        embeddings
      end

    embeddings =
      if config.with_ledge_distance do
        [embed_ledge_distance(player) | embeddings]
      else
        embeddings
      end

    embeddings =
      if config.with_nana do
        # Pass Popo's action for desync detection in compact mode
        [embed_nana(player.nana, config, player.action) | embeddings]
      else
        embeddings
      end

    # Reverse and concatenate (we prepended for efficiency)
    embeddings
    |> Enum.reverse()
    |> Nx.concatenate()
  end

  @doc """
  Batch embed multiple players at once - MUCH faster than calling embed/2 in a loop.

  This extracts all values first (Elixir work), then does all Nx operations
  in batch, reducing overhead from O(N * ops) to O(ops).

  ## Returns
    Tensor of shape [batch_size, embedding_size]
  """
  @spec embed_batch([PlayerState.t() | nil], config()) :: Nx.Tensor.t()
  def embed_batch(players, config \\ default_config()) when is_list(players) do
    # Handle empty list
    if Enum.empty?(players) do
      Nx.broadcast(0.0, {0, embedding_size(config)})
    else
      embed_batch_base(players, config)
    end
  end

  defp embed_batch_base(players, config) do
    # Extract all values into lists (pure Elixir - fast)
    percents = Enum.map(players, fn p -> (p && p.percent) || 0.0 end)
    facings = Enum.map(players, fn p -> (p && p.facing) || false end)
    xs = Enum.map(players, fn p -> (p && p.x) || 0.0 end)
    ys = Enum.map(players, fn p -> (p && p.y) || 0.0 end)
    actions = Enum.map(players, fn p -> (p && p.action) || 0 end)
    characters = Enum.map(players, fn p -> (p && p.character) || 0 end)
    invulnerables = Enum.map(players, fn p -> (p && p.invulnerable) || false end)
    jumps = Enum.map(players, fn p -> (p && p.jumps_left) || 0 end)
    shields = Enum.map(players, fn p -> (p && p.shield_strength) || 0.0 end)
    on_grounds = Enum.map(players, fn p -> (p && p.on_ground) || false end)

    # Batch Nx operations (few calls instead of N)
    percent_emb = Primitives.batch_float_embed(percents, scale: 0.01, lower: 0.0, upper: 5.0)
    facing_emb = Primitives.batch_bool_embed(facings, on: 1.0, off: -1.0)
    x_emb = Primitives.batch_float_embed(xs, scale: config.xy_scale)
    y_emb = Primitives.batch_float_embed(ys, scale: config.xy_scale)
    invuln_emb = Primitives.batch_bool_embed(invulnerables)

    # Jumps: normalized (1-dim) or one-hot (7-dim) based on config
    jumps_emb =
      if config.jumps_normalized do
        # Normalized: jumps/6 in [0, 1] range
        Primitives.batch_float_embed(Enum.map(jumps, &min(&1 / 6, 1.0)),
          scale: 1.0,
          lower: 0.0,
          upper: 1.0
        )
      else
        # One-hot: 7 dimensions
        Primitives.batch_one_hot(Nx.tensor(jumps, type: :s32), size: 7, clamp: true)
      end

    shield_emb = Primitives.batch_float_embed(shields, scale: 0.01, lower: 0.0, upper: 1.0)
    ground_emb = Primitives.batch_bool_embed(on_grounds)

    # Base embeddings - build list based on config
    base_embs = [
      # [batch, 1]
      percent_emb,
      # [batch, 1]
      facing_emb,
      # [batch, 1]
      x_emb,
      # [batch, 1]
      y_emb
    ]

    # Add action one-hot only if not using learned embeddings
    base_embs =
      case config.action_mode do
        :one_hot ->
          action_emb =
            Primitives.batch_one_hot(Nx.tensor(actions, type: :s32),
              size: Constants.num_actions(),
              clamp: true
            )

          # [batch, num_actions]
          base_embs ++ [action_emb]

        :learned ->
          # Action IDs handled separately by network
          base_embs
      end

    # Add character one-hot only if not using learned embeddings
    base_embs =
      case config.character_mode do
        :one_hot ->
          char_emb =
            Primitives.batch_one_hot(Nx.tensor(characters, type: :s32),
              size: Constants.num_characters(),
              clamp: true
            )

          # [batch, num_characters]
          base_embs ++ [char_emb]

        :learned ->
          # Character IDs handled separately by network
          base_embs
      end

    # Continue with rest of features
    base_embs =
      base_embs ++
        [
          # [batch, 1]
          invuln_emb,
          # [batch, 1] or [batch, 7]
          jumps_emb,
          # [batch, 1]
          shield_emb,
          # [batch, 1]
          ground_emb
        ]

    # Add speeds if configured
    embs_with_speeds =
      if config.with_speeds do
        speed_air_x = Enum.map(players, fn p -> (p && p.speed_air_x_self) || 0.0 end)
        speed_ground_x = Enum.map(players, fn p -> (p && p.speed_ground_x_self) || 0.0 end)
        speed_y = Enum.map(players, fn p -> (p && p.speed_y_self) || 0.0 end)
        speed_x_attack = Enum.map(players, fn p -> (p && p.speed_x_attack) || 0.0 end)
        speed_y_attack = Enum.map(players, fn p -> (p && p.speed_y_attack) || 0.0 end)

        speed_embs = [
          Primitives.batch_float_embed(speed_air_x, scale: 0.5),
          Primitives.batch_float_embed(speed_ground_x, scale: 0.5),
          Primitives.batch_float_embed(speed_y, scale: 0.5),
          Primitives.batch_float_embed(speed_x_attack, scale: 0.5),
          Primitives.batch_float_embed(speed_y_attack, scale: 0.5)
        ]

        base_embs ++ speed_embs
      else
        base_embs
      end

    # Add frame info (hitstun + action frame) if configured
    embs_with_frame_info =
      if config.with_frame_info do
        # Hitstun frames remaining (0-120ish, normalized to 0-1)
        hitstun_frames = Enum.map(players, fn p -> (p && p.hitstun_frames_left) || 0 end)
        # Action frame (how far into the animation, typically 0-100+, normalized)
        action_frames = Enum.map(players, fn p -> (p && p.action_frame) || 0 end)

        frame_embs = [
          # Scale hitstun by max hitstun frames, clamp to 0-1
          Primitives.batch_float_embed(hitstun_frames,
            scale: 1 / Constants.max_hitstun_frames(),
            lower: 0.0,
            upper: 1.0
          ),
          # Scale action_frame by standard animation length, allow overflow for long animations
          Primitives.batch_float_embed(action_frames,
            scale: 1 / Constants.standard_action_frames(),
            lower: 0.0,
            upper: 2.0
          )
        ]

        embs_with_speeds ++ frame_embs
      else
        embs_with_speeds
      end

    # Add stock count if configured
    embs_with_stock =
      if config.with_stock do
        stocks = Enum.map(players, fn p -> (p && p.stock) || 0 end)
        stock_emb = Primitives.batch_float_embed(stocks, scale: 1 / 4, lower: 0.0, upper: 1.0)
        embs_with_frame_info ++ [stock_emb]
      else
        embs_with_frame_info
      end

    # Add distance to ledge if configured
    embs_with_ledge =
      if config.with_ledge_distance do
        # Calculate distance to nearest ledge (simplified: stage_edge - |x|)
        stage_edge = 85.0

        ledge_distances =
          Enum.map(players, fn p ->
            x = (p && p.x) || 0.0
            # Normalize: 0 at edge, ~1 at center
            (stage_edge - abs(x)) / stage_edge
          end)

        ledge_emb = Primitives.batch_float_embed(ledge_distances, scale: 1.0)
        embs_with_stock ++ [ledge_emb]
      else
        embs_with_stock
      end

    # Add Nana (Ice Climbers partner) if configured
    all_embs =
      if config.with_nana do
        batch_size = length(players)
        nana_embs = embed_batch_nana(players, config, batch_size)
        embs_with_ledge ++ [nana_embs]
      else
        embs_with_ledge
      end

    # Concatenate all embeddings: [batch, total_embed_size]
    Nx.concatenate(all_embs, axis: 1)
  end

  # Batch embed Nana for all players
  # Returns tensor of shape [batch_size, nana_embedding_size]
  defp embed_batch_nana(players, config, batch_size) do
    case config.nana_mode do
      :compact -> embed_batch_nana_compact(players, batch_size)
      :enhanced -> embed_batch_nana_enhanced(players, batch_size)
      :full -> embed_batch_nana_full(players, config, batch_size)
    end
  end

  # Compact Nana batch embedding (~39 dims)
  defp embed_batch_nana_compact(players, batch_size) do
    if not NanaEmbed.any_nana_exists?(players) do
      Nx.broadcast(0.0, {batch_size, compact_nana_embedding_size()})
    else
      # Extract values using helper
      values = NanaEmbed.extract_batch_values(players)
      popo_actions = Enum.map(players, fn p -> (p && p.action) || 0 end)

      # Compute IC tech flags (compact uses category-based sync)
      flags = NanaEmbed.compute_batch_flags(values, popo_actions, sync_mode: :category)

      # Build embedding
      Nx.concatenate(
        [
          # [batch, 1]
          Primitives.batch_bool_embed(values.exists),
          # [batch, 1]
          Primitives.batch_float_embed(values.xs, scale: 0.05),
          # [batch, 1]
          Primitives.batch_float_embed(values.ys, scale: 0.05),
          # [batch, 1]
          Primitives.batch_bool_embed(values.facings, on: 1.0, off: -1.0),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.on_ground),
          # [batch, 1]
          Primitives.batch_float_embed(values.percents, scale: 0.01, lower: 0.0, upper: 5.0),
          # [batch, 1]
          Primitives.batch_float_embed(values.stocks, scale: 1 / 4, lower: 0.0, upper: 1.0),
          # hitstun [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # action_frame [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # invulnerable [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # [batch, 25]
          Primitives.batch_one_hot(Nx.tensor(flags.categories, type: :s32),
            size: @nana_action_categories,
            clamp: true
          ),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_attacking),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_grabbing),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.can_act),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_synced)
        ],
        axis: 1
      )
    end
  end

  # Enhanced Nana batch embedding (14 dims - action ID handled separately by GameEmbed)
  defp embed_batch_nana_enhanced(players, batch_size) do
    if not NanaEmbed.any_nana_exists?(players) do
      Nx.broadcast(0.0, {batch_size, enhanced_compact_nana_embedding_size()})
    else
      # Extract values using helper
      values = NanaEmbed.extract_batch_values(players)
      popo_actions = Enum.map(players, fn p -> (p && p.action) || 0 end)

      # Compute IC tech flags (enhanced uses exact action comparison for sync)
      flags = NanaEmbed.compute_batch_flags(values, popo_actions, sync_mode: :exact)

      # Build embedding (14 dims total - no action category one-hot)
      Nx.concatenate(
        [
          # [batch, 1]
          Primitives.batch_bool_embed(values.exists),
          # [batch, 1]
          Primitives.batch_float_embed(values.xs, scale: 0.05),
          # [batch, 1]
          Primitives.batch_float_embed(values.ys, scale: 0.05),
          # [batch, 1]
          Primitives.batch_bool_embed(values.facings, on: 1.0, off: -1.0),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.on_ground),
          # [batch, 1]
          Primitives.batch_float_embed(values.percents, scale: 0.01, lower: 0.0, upper: 5.0),
          # [batch, 1]
          Primitives.batch_float_embed(values.stocks, scale: 1 / 4, lower: 0.0, upper: 1.0),
          # hitstun [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # action_frame [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # invulnerable [batch, 1]
          Nx.broadcast(0.0, {batch_size, 1}),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_attacking),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_grabbing),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.can_act),
          # [batch, 1]
          Primitives.batch_bool_embed(flags.is_synced)
        ],
        axis: 1
      )
    end
  end

  # Full Nana batch embedding (455 dims)
  defp embed_batch_nana_full(players, config, batch_size) do
    nanas = Enum.map(players, fn p -> p && p.nana end)
    has_any_nana = Enum.any?(nanas, & &1)

    # Calculate full Nana embedding size
    nana_base_size = base_embedding_size(config) + 1
    nana_speeds_size = if config.with_speeds, do: 5, else: 0
    nana_frames_size = if config.with_frame_info, do: 2, else: 0
    nana_stock_size = if config.with_stock, do: 1, else: 0
    nana_total_size = nana_base_size + nana_speeds_size + nana_frames_size + nana_stock_size

    if not has_any_nana do
      Nx.broadcast(0.0, {batch_size, nana_total_size})
    else
      # Extract Nana values (defaulting to 0/false for nil Nanas)
      nana_percents = Enum.map(nanas, fn n -> (n && n.percent) || 0.0 end)
      nana_facings = Enum.map(nanas, fn n -> (n && n.facing) || false end)
      nana_xs = Enum.map(nanas, fn n -> (n && n.x) || 0.0 end)
      nana_ys = Enum.map(nanas, fn n -> (n && n.y) || 0.0 end)
      nana_actions = Enum.map(nanas, fn n -> (n && n.action) || 0 end)
      nana_exists = Enum.map(nanas, fn n -> n != nil end)

      # Batch embed Nana values
      nana_percent_emb =
        Primitives.batch_float_embed(nana_percents, scale: 0.01, lower: 0.0, upper: 5.0)

      nana_facing_emb = Primitives.batch_bool_embed(nana_facings, on: 1.0, off: -1.0)
      nana_x_emb = Primitives.batch_float_embed(nana_xs, scale: config.xy_scale)
      nana_y_emb = Primitives.batch_float_embed(nana_ys, scale: config.xy_scale)

      nana_action_emb =
        Primitives.batch_one_hot(Nx.tensor(nana_actions, type: :s32),
          size: Constants.num_actions(),
          clamp: true
        )

      # Nana uses character 0, no jumps/shield/invuln/on_ground data
      nana_char_emb =
        Primitives.batch_one_hot(Nx.tensor(List.duplicate(0, batch_size), type: :s32),
          size: Constants.num_characters(),
          clamp: true
        )

      nana_invuln_emb = Nx.broadcast(0.0, {batch_size, 1})
      # Jumps: normalized (1-dim) or one-hot (7-dim) based on config
      nana_jumps_emb =
        if config.jumps_normalized do
          Nx.broadcast(0.0, {batch_size, 1})
        else
          Primitives.batch_one_hot(Nx.tensor(List.duplicate(0, batch_size), type: :s32),
            size: 7,
            clamp: true
          )
        end

      nana_shield_emb = Nx.broadcast(0.0, {batch_size, 1})
      nana_ground_emb = Nx.broadcast(0.0, {batch_size, 1})
      nana_exists_emb = Primitives.batch_bool_embed(nana_exists)

      base_nana_embs = [
        nana_percent_emb,
        nana_facing_emb,
        nana_x_emb,
        nana_y_emb,
        nana_action_emb,
        nana_char_emb,
        nana_invuln_emb,
        nana_jumps_emb,
        nana_shield_emb,
        nana_ground_emb,
        nana_exists_emb
      ]

      nana_embs_with_speeds =
        if config.with_speeds do
          base_nana_embs ++ [Nx.broadcast(0.0, {batch_size, 5})]
        else
          base_nana_embs
        end

      nana_embs_with_frame_info =
        if config.with_frame_info do
          nana_embs_with_speeds ++ [Nx.broadcast(0.0, {batch_size, 2})]
        else
          nana_embs_with_speeds
        end

      nana_embs_with_stock =
        if config.with_stock do
          nana_stocks = Enum.map(nanas, fn n -> (n && n.stock) || 0 end)

          nana_stock_emb =
            Primitives.batch_float_embed(nana_stocks, scale: 1 / 4, lower: 0.0, upper: 1.0)

          nana_embs_with_frame_info ++ [nana_stock_emb]
        else
          nana_embs_with_frame_info
        end

      Nx.concatenate(nana_embs_with_stock, axis: 1)
    end
  end

  @doc """
  Embed the base player features (no speeds, no nana).
  """
  @spec embed_base(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed_base(%PlayerState{} = player, config) do
    # Choose jumps embedding based on config
    jumps_embed =
      if config.jumps_normalized do
        Primitives.jumps_left_normalized_embed(player.jumps_left || 0)
      else
        Primitives.jumps_left_embed(player.jumps_left || 0)
      end

    # Build base features (without action if using learned embeddings)
    base_features = [
      # Percent - scaled damage
      Primitives.percent_embed(player.percent || 0.0),

      # Facing direction (-1 left, +1 right)
      Primitives.facing_embed(player.facing),

      # Position
      Primitives.xy_embed(player.x || 0.0, scale: config.xy_scale),
      Primitives.xy_embed(player.y || 0.0, scale: config.xy_scale)
    ]

    # Add action one-hot only if not using learned embeddings
    base_features =
      case config.action_mode do
        :one_hot -> base_features ++ [Primitives.action_embed(player.action || 0)]
        # Action ID handled separately by network
        :learned -> base_features
      end

    # Add character one-hot only if not using learned embeddings
    base_features =
      case config.character_mode do
        :one_hot -> base_features ++ [Primitives.character_embed(player.character || 0)]
        # Character ID handled separately by network
        :learned -> base_features
      end

    # Continue with rest of features
    all_features =
      base_features ++
        [
          # Invulnerable flag
          Primitives.bool_embed(player.invulnerable || false),

          # Jumps remaining (1-dim normalized or 7-dim one-hot)
          jumps_embed,

          # Shield strength
          Primitives.shield_embed(player.shield_strength || 0.0),

          # On ground flag
          Primitives.bool_embed(player.on_ground || false)
        ]

    Nx.concatenate(all_features)
  end

  @doc """
  Embed speed/velocity values.
  """
  @spec embed_speeds(PlayerState.t(), config()) :: Nx.Tensor.t()
  def embed_speeds(%PlayerState{} = player, _config) do
    Nx.concatenate([
      Primitives.speed_embed(player.speed_air_x_self || 0.0),
      Primitives.speed_embed(player.speed_ground_x_self || 0.0),
      Primitives.speed_embed(player.speed_y_self || 0.0),
      Primitives.speed_embed(player.speed_x_attack || 0.0),
      Primitives.speed_embed(player.speed_y_attack || 0.0)
    ])
  end

  @doc """
  Embed frame timing info (hitstun remaining, action frame).
  Useful for knowing when player/opponent can act.
  """
  @spec embed_frame_info(PlayerState.t()) :: Nx.Tensor.t()
  def embed_frame_info(%PlayerState{} = player) do
    # Hitstun frames: normalize by max hitstun (120 frames)
    hitstun = min((player.hitstun_frames_left || 0) / Constants.max_hitstun_frames(), 1.0)
    # Action frame: normalize by standard animation length, allow overflow
    action_frame = min((player.action_frame || 0) / Constants.standard_action_frames(), 2.0)

    Nx.tensor([hitstun, action_frame], type: :f32)
  end

  @doc """
  Embed stock count (lives remaining).
  Critical for decision-making: play aggressive when ahead, defensive when behind.
  """
  @spec embed_stock(PlayerState.t()) :: Nx.Tensor.t()
  def embed_stock(%PlayerState{} = player) do
    # Stocks: typically 1-4, normalize to 0-1 (4 stocks = 1.0)
    stock = min((player.stock || 0) / 4, 1.0)
    Nx.tensor([stock], type: :f32)
  end

  @doc """
  Embed distance to nearest ledge (stage edge).
  Useful for recovery and edgeguard awareness.
  Uses simplified estimate based on x position and typical stage width.
  """
  @spec embed_ledge_distance(PlayerState.t()) :: Nx.Tensor.t()
  def embed_ledge_distance(%PlayerState{} = player) do
    # Approximate stage edge at x = ±85 (varies by stage, but good average)
    # Distance to nearest edge = stage_edge - |x|
    # Negative means offstage
    stage_edge = 85.0
    x = player.x || 0.0
    distance_to_edge = stage_edge - abs(x)
    # Normalize: 0 at edge, 1 at center, negative offstage
    # Scale so center of stage (~0) = 0.5, edge = 0, deep offstage = -1
    normalized = distance_to_edge / stage_edge
    Nx.tensor([normalized], type: :f32)
  end

  @doc """
  Embed Nana (Ice Climbers partner).

  Supports three modes:
  - `:full` - Full player embedding (455 dims)
  - `:compact` - Compact embedding preserving IC tech (~39 dims)
  - `:enhanced` - Enhanced compact with action ID for learned embedding (14 dims)

  The `popo_action` parameter is used in compact/enhanced mode for desync detection.

  For `:enhanced` mode, use `get_nana_action_id/1` to get the action ID separately
  for the learned embedding layer.
  """
  @spec embed_nana(Nana.t() | nil, config(), integer() | nil) :: Nx.Tensor.t()
  def embed_nana(nana, config, popo_action \\ nil)

  def embed_nana(nil, config, _popo_action) do
    # Nana doesn't exist - return zeros
    size =
      case config.nana_mode do
        :compact ->
          compact_nana_embedding_size()

        :enhanced ->
          enhanced_compact_nana_embedding_size()

        :full ->
          base_size = base_embedding_size(config) + 1
          speed_size = if config.with_speeds, do: 5, else: 0
          frame_size = if config.with_frame_info, do: 2, else: 0
          stock_size = if config.with_stock, do: 1, else: 0
          base_size + speed_size + frame_size + stock_size
      end

    Nx.broadcast(0.0, {size})
  end

  def embed_nana(%Nana{} = nana, config, popo_action) do
    case config.nana_mode do
      :compact -> embed_nana_compact(nana, popo_action)
      :enhanced -> embed_nana_enhanced(nana, popo_action)
      :full -> embed_nana_full(nana, config)
    end
  end

  # Compact Nana embedding (~39 dims) - preserves IC tech learning
  defp embed_nana_compact(%Nana{} = nana, popo_action) do
    nana_y = nana.y || 0.0
    nana_action = nana.action || 0

    # Compute IC tech flags using helper (compact uses category-based sync)
    flags = NanaEmbed.compute_flags(nana_action, popo_action || 0, nana_y, sync_mode: :category)

    Nx.concatenate([
      # Exists
      Primitives.bool_embed(true),

      # Position
      Primitives.xy_embed(nana.x || 0.0, scale: 0.05),
      Primitives.xy_embed(nana_y, scale: 0.05),

      # Facing
      Primitives.facing_embed(nana.facing),

      # On ground (from flags)
      Primitives.bool_embed(flags.on_ground),

      # Combat state
      Primitives.percent_embed(nana.percent || 0.0),
      # Stock
      Nx.tensor([min((nana.stock || 0) / 4, 1.0)], type: :f32),

      # Frame info (zeros since Nana struct doesn't have these)
      # hitstun_frames
      Nx.tensor([0.0], type: :f32),
      # action_frame
      Nx.tensor([0.0], type: :f32),

      # Invulnerable (unknown, default false)
      Primitives.bool_embed(false),

      # Action category (25-dim one-hot)
      Primitives.one_hot(flags.category, size: @nana_action_categories, clamp: true),

      # IC tech flags
      Primitives.bool_embed(flags.is_attacking),
      Primitives.bool_embed(flags.is_grabbing),
      Primitives.bool_embed(flags.can_act),
      Primitives.bool_embed(flags.is_synced)
    ])
  end

  # Enhanced Compact Nana embedding (14 dims) - uses action ID for learned embedding
  # Provides precise action info via separate action ID, not 25-dim category
  # Critical for IC tech: guarantees network knows exact action (dair vs fair vs ftilt)
  defp embed_nana_enhanced(%Nana{} = nana, popo_action) do
    nana_y = nana.y || 0.0
    nana_action = nana.action || 0

    # Compute IC tech flags using helper (enhanced uses exact action comparison)
    flags = NanaEmbed.compute_flags(nana_action, popo_action || 0, nana_y, sync_mode: :exact)

    Nx.concatenate([
      # Exists
      Primitives.bool_embed(true),

      # Position (scaled for ~[-200, 200] stage range)
      Primitives.xy_embed(nana.x || 0.0, scale: 0.05),
      Primitives.xy_embed(nana_y, scale: 0.05),

      # Facing (-1 left, +1 right)
      Primitives.facing_embed(nana.facing),

      # On ground (from flags)
      Primitives.bool_embed(flags.on_ground),

      # Combat state
      Primitives.percent_embed(nana.percent || 0.0),
      # Stock normalized
      Nx.tensor([min((nana.stock || 0) / 4, 1.0)], type: :f32),

      # Frame info - CRITICAL for IC tech timing
      # Note: Nana struct may not have these, so we approximate
      # TODO: Get real action_frame from libmelee if available
      # hitstun_frames (placeholder)
      Nx.tensor([0.0], type: :f32),
      # action_frame (placeholder)
      Nx.tensor([0.0], type: :f32),

      # Invulnerable (unknown, default false)
      Primitives.bool_embed(false),

      # IC tech flags (from helper)
      Primitives.bool_embed(flags.is_attacking),
      Primitives.bool_embed(flags.is_grabbing),
      Primitives.bool_embed(flags.can_act),
      Primitives.bool_embed(flags.is_synced)
    ])

    # Note: Nana action ID is NOT included here - it's appended by GameEmbed
    # for the learned embedding layer to process
  end

  # Full Nana embedding (455 dims) - original behavior
  defp embed_nana_full(%Nana{} = nana, config) do
    # Convert Nana to a pseudo-player for embedding
    nana_as_player = %PlayerState{
      percent: nana.percent,
      facing: nana.facing,
      x: nana.x,
      y: nana.y,
      action: nana.action,
      # Nana is always same character as Popo
      character: 0,
      invulnerable: false,
      jumps_left: 0,
      shield_strength: 0.0,
      on_ground: false,
      # Speeds not available for Nana
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil,
      stock: nana.stock,
      action_frame: 0,
      hitstun_frames_left: 0
    }

    base = embed_base(nana_as_player, config)
    exists = Primitives.bool_embed(true)

    # Add optional speeds (zeros for Nana since data not available)
    embs = [base, exists]

    embs =
      if config.with_speeds do
        embs ++ [Nx.broadcast(0.0, {5})]
      else
        embs
      end

    # Add optional frame info (zeros for Nana since data not available)
    embs =
      if config.with_frame_info do
        embs ++ [Nx.broadcast(0.0, {2})]
      else
        embs
      end

    # Add stock (Nana has her own stock count)
    embs =
      if config.with_stock do
        stock = min((nana.stock || 0) / 4, 1.0)
        embs ++ [Nx.tensor([stock], type: :f32)]
      else
        embs
      end

    Nx.concatenate(embs)
  end

  # ============================================================================
  # Action/Character ID Extraction (delegated to Ids module)
  # ============================================================================

  @doc """
  Get the action ID from a player state.
  Delegates to `ExPhil.Embeddings.Player.Ids`.
  """
  defdelegate get_action_id(player), to: Ids

  @doc """
  Get action IDs from a list of players as a tensor.
  """
  defdelegate get_action_ids_batch(players), to: Ids

  @doc """
  Get Nana's action ID from a player state.
  """
  defdelegate get_nana_action_id(player), to: Ids

  @doc """
  Get Nana action IDs from a list of players as a tensor.
  """
  defdelegate get_nana_action_ids_batch(players), to: Ids

  @doc """
  Get a player's character ID for learned embedding.
  """
  defdelegate get_character_id(player), to: Ids

  @doc """
  Get character IDs from a list of players as a tensor.
  """
  defdelegate get_character_ids_batch(players), to: Ids

  @doc """
  Check if learned character embeddings are being used.
  """
  @spec uses_learned_characters?(config()) :: boolean()
  def uses_learned_characters?(config), do: config.character_mode == :learned

  @doc """
  Check if enhanced Nana mode is being used.

  Enhanced mode uses action IDs for learned embedding instead of category one-hot.
  """
  @spec uses_enhanced_nana?(config()) :: boolean()
  def uses_enhanced_nana?(config), do: config.nana_mode == :enhanced

  @doc """
  Embed both players from a game state.

  Returns a tensor of shape [2, player_embedding_size] or
  flattened to [2 * player_embedding_size] if `flatten: true`.
  """
  @spec embed_both(PlayerState.t(), PlayerState.t(), keyword()) :: Nx.Tensor.t()
  def embed_both(p0, p1, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    flatten = Keyword.get(opts, :flatten, true)

    p0_embed = embed(p0, config)
    p1_embed = embed(p1, config)

    if flatten do
      Nx.concatenate([p0_embed, p1_embed])
    else
      Nx.stack([p0_embed, p1_embed])
    end
  end

  @doc """
  Embed players with ego-centric view (own player first, opponent second).

  This is the typical format for training - the agent always sees itself
  as "player 0" regardless of which port it's actually on.
  """
  @spec embed_ego(PlayerState.t(), PlayerState.t(), keyword()) :: Nx.Tensor.t()
  def embed_ego(own, opponent, opts \\ []) do
    embed_both(own, opponent, opts)
  end
end
