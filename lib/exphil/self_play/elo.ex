defmodule ExPhil.SelfPlay.Elo do
  @moduledoc """
  Elo rating calculation utilities for self-play matchmaking.

  Implements the standard Elo rating system with configurable K-factor
  and support for draws.

  ## Elo Formula

  Expected score:

      E_A = 1 / (1 + 10^((R_B - R_A) / 400))

  Rating update:

      R'_A = R_A + K * (S_A - E_A)

  Where:
  - R_A, R_B: Current ratings
  - E_A: Expected score for player A
  - S_A: Actual score (1 = win, 0.5 = draw, 0 = loss)
  - K: K-factor (determines how much ratings change)

  ## K-Factor Guidelines

  - 40: New players (< 30 games)
  - 32: Standard (most common)
  - 24: Established players (100+ games)
  - 16: Top-tier players (stable ratings)

  ## Usage

      # Calculate new ratings
      {new_rating_a, new_rating_b} = Elo.update(1500, 1600, :win)
      # => {1514.6, 1585.4}

      # With custom K-factor
      {new_a, new_b} = Elo.update(1500, 1600, :win, k_factor: 40)

      # Get expected win probability
      prob = Elo.expected_score(1500, 1600)
      # => 0.36

  """

  @default_k_factor 32
  @default_initial_rating 1000

  @type result :: :win | :loss | :draw

  @doc """
  Calculates the expected score for player A against player B.

  Returns a value between 0 and 1, representing the probability
  of player A winning or drawing.
  """
  @spec expected_score(number(), number()) :: float()
  def expected_score(rating_a, rating_b) do
    1 / (1 + :math.pow(10, (rating_b - rating_a) / 400))
  end

  @doc """
  Updates ratings based on match result.

  Returns `{new_rating_a, new_rating_b}`.

  ## Options
    - `:k_factor` - K-factor for both players (default: 32)
    - `:k_factor_a` - K-factor for player A
    - `:k_factor_b` - K-factor for player B
  """
  @spec update(number(), number(), result(), keyword()) :: {float(), float()}
  def update(rating_a, rating_b, result, opts \\ []) do
    k_a = Keyword.get(opts, :k_factor_a, Keyword.get(opts, :k_factor, @default_k_factor))
    k_b = Keyword.get(opts, :k_factor_b, Keyword.get(opts, :k_factor, @default_k_factor))

    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a

    {actual_a, actual_b} = actual_scores(result)

    new_rating_a = rating_a + k_a * (actual_a - expected_a)
    new_rating_b = rating_b + k_b * (actual_b - expected_b)

    {new_rating_a, new_rating_b}
  end

  @doc """
  Updates a single player's rating given their opponent's rating and result.
  """
  @spec update_single(number(), number(), result(), keyword()) :: float()
  def update_single(player_rating, opponent_rating, result, opts \\ []) do
    k = Keyword.get(opts, :k_factor, @default_k_factor)

    expected = expected_score(player_rating, opponent_rating)

    actual =
      case result do
        :win -> 1.0
        :loss -> 0.0
        :draw -> 0.5
      end

    player_rating + k * (actual - expected)
  end

  @doc """
  Calculates K-factor based on number of games played.

  Uses a sliding scale:
  - < 30 games: 40 (provisional)
  - 30-100 games: 32 (standard)
  - > 100 games: 24 (established)
  """
  @spec dynamic_k_factor(non_neg_integer()) :: number()
  def dynamic_k_factor(games_played) do
    cond do
      games_played < 30 -> 40
      games_played < 100 -> 32
      true -> 24
    end
  end

  @doc """
  Returns the default initial rating.
  """
  @spec initial_rating() :: number()
  def initial_rating, do: @default_initial_rating

  @doc """
  Returns the default K-factor.
  """
  @spec default_k_factor() :: number()
  def default_k_factor, do: @default_k_factor

  @doc """
  Calculates the rating difference needed for a given win probability.

  Inverse of expected_score.
  """
  @spec rating_difference_for_probability(float()) :: float()
  def rating_difference_for_probability(prob) when prob > 0 and prob < 1 do
    # E = 1 / (1 + 10^(d/400))
    # Solving for d: d = -400 * log10(1/E - 1)
    -400 * :math.log10(1 / prob - 1)
  end

  @doc """
  Estimates the number of games needed to reach target rating.

  Assumes consistent performance against opponents of average_opponent_rating.
  """
  @spec games_to_reach(number(), number(), number(), float()) :: non_neg_integer()
  def games_to_reach(current_rating, target_rating, average_opponent_rating, win_rate) do
    if current_rating >= target_rating do
      0
    else
      # Iteratively calculate
      simulate_games(current_rating, target_rating, average_opponent_rating, win_rate, 0)
    end
  end

  defp simulate_games(current, target, _opp_rating, _win_rate, games) when current >= target do
    games
  end

  defp simulate_games(_current, _target, _opp_rating, _win_rate, games) when games > 10000 do
    # Prevent infinite loop
    games
  end

  defp simulate_games(current, target, opp_rating, win_rate, games) do
    # Weighted average of win/loss rating changes
    {rating_win, _} = update(current, opp_rating, :win)
    {rating_loss, _} = update(current, opp_rating, :loss)

    new_rating = win_rate * rating_win + (1 - win_rate) * rating_loss

    simulate_games(new_rating, target, opp_rating, win_rate, games + 1)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp actual_scores(:win), do: {1.0, 0.0}
  defp actual_scores(:loss), do: {0.0, 1.0}
  defp actual_scores(:draw), do: {0.5, 0.5}
end
