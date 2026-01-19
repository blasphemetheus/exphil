defmodule ExPhil.Training.Augmentation do
  @moduledoc """
  Data augmentation for Melee training data.

  Provides augmentation functions to improve model generalization:
  - Horizontal flip (mirror): Flips X positions and facing directions
  - Noise injection: Adds small noise to continuous values

  ## Usage

      # Single augmentation
      augmented_frame = Augmentation.mirror(frame)

      # Apply with probability
      frame = Augmentation.maybe_mirror(frame, probability: 0.5)

      # Combined augmentations
      frame = frame
        |> Augmentation.maybe_mirror(probability: 0.5)
        |> Augmentation.maybe_add_noise(probability: 0.3, scale: 0.01)

  ## Why Mirror?

  Most Melee stages are horizontally symmetric. Mirroring a game state:
  - Effectively doubles training data
  - Helps model generalize left/right symmetry
  - Prevents overfitting to one side of the stage
  """

  alias ExPhil.Bridge.{GameState, Player, Nana, ControllerState, Projectile, Item}

  @doc """
  Horizontally flip (mirror) a game state.

  Flips:
  - Player X positions and X velocities
  - Player facing directions (1 ↔ -1)
  - Controller stick X values (for action targets)
  - Projectile and item X positions

  ## Examples

      iex> mirrored = Augmentation.mirror(game_state)
      iex> mirrored.players[1].x == -game_state.players[1].x
      true

  """
  @spec mirror(map()) :: map()
  def mirror(%{game_state: game_state, controller: controller} = frame) do
    %{frame |
      game_state: mirror_game_state(game_state),
      controller: mirror_controller(controller)
    }
  end

  def mirror(%{game_state: game_state} = frame) do
    %{frame | game_state: mirror_game_state(game_state)}
  end

  @doc """
  Apply mirror augmentation with a given probability.

  ## Options
    - `:probability` - Chance to apply augmentation (0.0-1.0, default: 0.5)

  ## Examples

      iex> frame = Augmentation.maybe_mirror(frame, probability: 0.5)

  """
  @spec maybe_mirror(map(), keyword()) :: map()
  def maybe_mirror(frame, opts \\ []) do
    probability = Keyword.get(opts, :probability, 0.5)
    if :rand.uniform() < probability do
      mirror(frame)
    else
      frame
    end
  end

  @doc """
  Add small random noise to continuous values.

  Adds Gaussian noise to:
  - Player positions (X, Y)
  - Player damage percent
  - Velocities

  Does NOT add noise to:
  - Discrete values (stock count, action state, buttons)
  - Binary flags (on_ground, invulnerable)

  ## Options
    - `:scale` - Standard deviation of noise (default: 0.01)

  """
  @spec add_noise(map(), keyword()) :: map()
  def add_noise(%{game_state: game_state} = frame, opts \\ []) do
    scale = Keyword.get(opts, :scale, 0.01)
    %{frame | game_state: add_noise_to_game_state(game_state, scale)}
  end

  @doc """
  Apply noise augmentation with a given probability.

  ## Options
    - `:probability` - Chance to apply augmentation (0.0-1.0, default: 0.3)
    - `:scale` - Standard deviation of noise (default: 0.01)

  """
  @spec maybe_add_noise(map(), keyword()) :: map()
  def maybe_add_noise(frame, opts \\ []) do
    probability = Keyword.get(opts, :probability, 0.3)
    if :rand.uniform() < probability do
      add_noise(frame, opts)
    else
      frame
    end
  end

  @doc """
  Apply all augmentations to a frame with configurable probabilities.

  ## Options
    - `:mirror_prob` - Probability of mirroring (default: 0.5)
    - `:noise_prob` - Probability of noise (default: 0.3)
    - `:noise_scale` - Noise scale (default: 0.01)

  """
  @spec augment(map(), keyword()) :: map()
  def augment(frame, opts \\ []) do
    mirror_prob = Keyword.get(opts, :mirror_prob, 0.5)
    noise_prob = Keyword.get(opts, :noise_prob, 0.3)
    noise_scale = Keyword.get(opts, :noise_scale, 0.01)

    frame
    |> maybe_mirror(probability: mirror_prob)
    |> maybe_add_noise(probability: noise_prob, scale: noise_scale)
  end

  # ===========================================================================
  # Mirror Implementation
  # ===========================================================================

  defp mirror_game_state(%GameState{} = gs) do
    %{gs |
      players: Map.new(gs.players, fn {port, player} ->
        {port, mirror_player(player)}
      end),
      projectiles: Enum.map(gs.projectiles || [], &mirror_projectile/1),
      items: Enum.map(gs.items || [], &mirror_item/1)
    }
  end

  defp mirror_game_state(gs) when is_map(gs) do
    # Handle plain map game states (from parsed replays)
    players = gs[:players] || gs["players"] || %{}
    mirrored_players = Map.new(players, fn {port, player} ->
      {port, mirror_player_map(player)}
    end)

    gs
    |> Map.put(:players, mirrored_players)
    |> Map.put("players", mirrored_players)
  end

  defp mirror_player(%Player{} = p) do
    %{p |
      x: -p.x,
      facing: -p.facing,
      speed_air_x_self: -(p.speed_air_x_self || 0.0),
      speed_ground_x_self: -(p.speed_ground_x_self || 0.0),
      speed_x_attack: -(p.speed_x_attack || 0.0),
      nana: mirror_nana(p.nana)
    }
  end

  defp mirror_player(nil), do: nil

  defp mirror_player_map(p) when is_map(p) do
    p
    |> Map.update(:x, 0.0, &(-&1))
    |> Map.update("x", 0.0, &(-&1))
    |> Map.update(:facing, 1, &(-&1))
    |> Map.update("facing", 1, &(-&1))
    |> Map.update(:speed_air_x_self, 0.0, &(-(&1 || 0.0)))
    |> Map.update(:speed_ground_x_self, 0.0, &(-(&1 || 0.0)))
    |> Map.update(:speed_x_attack, 0.0, &(-(&1 || 0.0)))
  end

  defp mirror_nana(%Nana{} = n) do
    %{n |
      x: -n.x,
      facing: -n.facing
    }
  end

  defp mirror_nana(nil), do: nil

  defp mirror_nana(n) when is_map(n) do
    n
    |> Map.update(:x, 0.0, &(-&1))
    |> Map.update(:facing, 1, &(-&1))
  end

  defp mirror_controller(%ControllerState{} = cs) do
    %{cs |
      main_stick: mirror_stick(cs.main_stick),
      c_stick: mirror_stick(cs.c_stick)
    }
  end

  defp mirror_controller(cs) when is_map(cs) do
    cs
    |> Map.update(:main_stick, %{x: 0.5, y: 0.5}, &mirror_stick/1)
    |> Map.update("main_stick", %{x: 0.5, y: 0.5}, &mirror_stick/1)
    |> Map.update(:c_stick, %{x: 0.5, y: 0.5}, &mirror_stick/1)
    |> Map.update("c_stick", %{x: 0.5, y: 0.5}, &mirror_stick/1)
  end

  defp mirror_controller(nil), do: nil

  # Sticks are in [0, 1] range centered at 0.5
  # Mirror: x → 1.0 - x
  defp mirror_stick(%{x: x, y: y}) do
    %{x: 1.0 - x, y: y}
  end

  defp mirror_stick(nil), do: nil

  defp mirror_projectile(%Projectile{} = p) do
    %{p | x: -p.x, speed_x: -(p.speed_x || 0.0)}
  end

  defp mirror_projectile(p) when is_map(p) do
    p
    |> Map.update(:x, 0.0, &(-&1))
    |> Map.update(:speed_x, 0.0, &(-(&1 || 0.0)))
  end

  defp mirror_projectile(nil), do: nil

  # Items don't have velocity fields in the struct, only position and facing
  defp mirror_item(%Item{} = i) do
    %{i | x: -i.x}
  end

  defp mirror_item(i) when is_map(i) do
    Map.update(i, :x, 0.0, &(-&1))
  end

  defp mirror_item(nil), do: nil

  # ===========================================================================
  # Noise Implementation
  # ===========================================================================

  defp add_noise_to_game_state(%GameState{} = gs, scale) do
    %{gs |
      players: Map.new(gs.players, fn {port, player} ->
        {port, add_noise_to_player(player, scale)}
      end)
    }
  end

  defp add_noise_to_game_state(gs, scale) when is_map(gs) do
    players = gs[:players] || gs["players"] || %{}
    noisy_players = Map.new(players, fn {port, player} ->
      {port, add_noise_to_player_map(player, scale)}
    end)

    gs
    |> Map.put(:players, noisy_players)
    |> Map.put("players", noisy_players)
  end

  defp add_noise_to_player(%Player{} = p, scale) do
    %{p |
      x: p.x + gaussian_noise(scale),
      y: p.y + gaussian_noise(scale),
      percent: max(0.0, p.percent + gaussian_noise(scale * 10)),  # Larger scale for percent
      speed_air_x_self: (p.speed_air_x_self || 0.0) + gaussian_noise(scale),
      speed_ground_x_self: (p.speed_ground_x_self || 0.0) + gaussian_noise(scale),
      speed_y_self: (p.speed_y_self || 0.0) + gaussian_noise(scale)
    }
  end

  defp add_noise_to_player(nil, _scale), do: nil

  defp add_noise_to_player_map(p, scale) when is_map(p) do
    p
    |> Map.update(:x, 0.0, &(&1 + gaussian_noise(scale)))
    |> Map.update(:y, 0.0, &(&1 + gaussian_noise(scale)))
    |> Map.update(:percent, 0.0, &max(0.0, &1 + gaussian_noise(scale * 10)))
  end

  # Box-Muller transform for Gaussian noise
  defp gaussian_noise(scale) do
    u1 = :rand.uniform()
    u2 = :rand.uniform()
    # Avoid log(0)
    u1 = max(u1, 1.0e-10)
    z = :math.sqrt(-2.0 * :math.log(u1)) * :math.cos(2.0 * :math.pi() * u2)
    z * scale
  end
end
