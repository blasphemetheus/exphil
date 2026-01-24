defmodule ExPhil.Training.CharacterBalance do
  @moduledoc """
  Character-balanced sampling for multi-character training.

  When training on multiple characters (e.g., Mewtwo, Ganondorf, Link), the dataset
  is often imbalanced - some characters have many more replays than others. This
  module provides weighted sampling to ensure all characters are represented fairly.

  ## Why Balance Characters?

  Eric Gu's research on Melee AI found that all-character training outperforms
  single-character training. However, if Fox has 100K frames and Link has 1K frames,
  the model will primarily learn Fox patterns.

  Character balancing uses inverse frequency weighting:
  - Rare characters (Link, G&W) get higher weights
  - Common characters (Fox, Falco) get lower weights
  - All characters contribute equally to training

  ## Usage

      # Compute character frequencies from frames
      character_counts = CharacterBalance.count_characters(frames)
      #=> %{mewtwo: 50000, ganondorf: 30000, link: 10000}

      # Compute sampling weights (inverse frequency)
      weights = CharacterBalance.compute_weights(character_counts)
      #=> %{mewtwo: 0.6, ganondorf: 1.0, link: 3.0}

      # Get per-frame weights for weighted sampling
      frame_weights = CharacterBalance.frame_weights(frames, weights)
      #=> [0.6, 0.6, 1.0, 3.0, ...]  # One weight per frame

  ## Integration with Training

  Use `--balance-characters` flag:

      mix run scripts/train_from_replays.exs \\
        --character mewtwo,ganondorf,link,gameandwatch,zelda \\
        --balance-characters

  """

  require Logger

  @doc """
  Extract character from a training frame.

  Frames have different structures depending on source:
  - From peppi: frame.game_state.players[port].character
  - Sequence frames: frame.sequence[0].game_state.players[port].character

  Returns the character atom or nil if not found.
  """
  @spec extract_character(map()) :: atom() | nil
  def extract_character(frame) do
    cond do
      # Sequence frame (temporal training)
      Map.has_key?(frame, :sequence) and is_list(frame.sequence) ->
        first_frame = List.first(frame.sequence)
        extract_character_from_game_state(first_frame[:game_state])

      # Regular frame
      Map.has_key?(frame, :game_state) ->
        extract_character_from_game_state(frame.game_state)

      true ->
        nil
    end
  end

  defp extract_character_from_game_state(nil), do: nil
  defp extract_character_from_game_state(game_state) do
    # game_state.players is a map like %{1 => %Player{character: :mewtwo}, 2 => %Player{...}}
    # We want player 0/1 (the agent's character)
    players = Map.get(game_state, :players) || %{}

    # Try port 0 first (common), then port 1
    player = Map.get(players, 0) || Map.get(players, 1) || Map.values(players) |> List.first()

    case player do
      %{character: char} when is_atom(char) -> char
      %{character: char} when is_integer(char) -> character_id_to_atom(char)
      _ -> nil
    end
  end

  # Convert Melee internal character ID to atom
  # Based on libmelee character IDs
  defp character_id_to_atom(id) when is_integer(id) do
    case id do
      0 -> :captain_falcon
      1 -> :donkey_kong
      2 -> :fox
      3 -> :game_and_watch
      4 -> :kirby
      5 -> :bowser
      6 -> :link
      7 -> :luigi
      8 -> :mario
      9 -> :marth
      10 -> :mewtwo
      11 -> :ness
      12 -> :peach
      13 -> :pikachu
      14 -> :ice_climbers
      15 -> :jigglypuff
      16 -> :samus
      17 -> :yoshi
      18 -> :zelda
      19 -> :sheik
      20 -> :falco
      21 -> :young_link
      22 -> :dr_mario
      23 -> :roy
      24 -> :pichu
      25 -> :ganondorf
      _ -> :unknown
    end
  end

  @doc """
  Count frames per character.

  Returns a map of character => count.

  ## Example

      counts = CharacterBalance.count_characters(frames)
      #=> %{mewtwo: 50000, ganondorf: 30000, link: 10000}

  """
  @spec count_characters([map()]) :: %{atom() => non_neg_integer()}
  def count_characters(frames) do
    frames
    |> Enum.map(&extract_character/1)
    |> Enum.reject(&is_nil/1)
    |> Enum.frequencies()
  end

  @doc """
  Compute sampling weights from character counts.

  Uses inverse frequency weighting so rare characters are sampled more often.
  Weights are normalized so the most common character has weight 1.0.

  ## Options

    - `:min_weight` - Minimum weight cap (default: 0.1)
    - `:max_weight` - Maximum weight cap (default: 10.0)
    - `:smoothing` - Add to counts before computing weights (default: 0)

  ## Example

      counts = %{mewtwo: 50000, ganondorf: 30000, link: 10000}
      weights = CharacterBalance.compute_weights(counts)
      #=> %{mewtwo: 0.6, ganondorf: 1.0, link: 3.0}

  """
  @spec compute_weights(%{atom() => non_neg_integer()}, keyword()) :: %{atom() => float()}
  def compute_weights(character_counts, opts \\ []) do
    min_weight = Keyword.get(opts, :min_weight, 0.1)
    max_weight = Keyword.get(opts, :max_weight, 10.0)
    smoothing = Keyword.get(opts, :smoothing, 0)

    # Find the minimum count (rarest character)
    min_count = character_counts
    |> Map.values()
    |> Enum.min(fn -> 1 end)

    # Compute inverse frequency weights
    # Rare characters get higher weights
    character_counts
    |> Enum.map(fn {char, count} ->
      # Weight = min_count / count
      # This makes the rarest character have weight 1.0
      # and more common characters have lower weights
      weight = (min_count + smoothing) / (count + smoothing)

      # Clamp to [min_weight, max_weight]
      weight = weight |> max(min_weight) |> min(max_weight)

      {char, weight}
    end)
    |> Map.new()
  end

  @doc """
  Get sampling weight for each frame based on its character.

  Returns a list of weights, one per frame, suitable for weighted sampling.

  ## Example

      weights = %{mewtwo: 0.6, ganondorf: 1.0, link: 3.0}
      frame_weights = CharacterBalance.frame_weights(frames, weights)
      #=> [0.6, 0.6, 1.0, 3.0, ...]

  """
  @spec frame_weights([map()], %{atom() => float()}) :: [float()]
  def frame_weights(frames, character_weights) do
    default_weight = 1.0

    Enum.map(frames, fn frame ->
      char = extract_character(frame)
      Map.get(character_weights, char, default_weight)
    end)
  end

  @doc """
  Sample indices using weighted random sampling.

  This implements weighted sampling without replacement, suitable for
  creating balanced batches during training.

  ## Parameters

    - `weights` - List of weights, one per item
    - `n` - Number of indices to sample

  ## Returns

    List of sampled indices.

  ## Example

      weights = [0.6, 0.6, 1.0, 3.0, 3.0]
      indices = CharacterBalance.weighted_sample(weights, 3)
      #=> [3, 4, 2]  # Indices of items with higher weights more likely

  """
  @spec weighted_sample([float()], non_neg_integer()) :: [non_neg_integer()]
  def weighted_sample(weights, n) when is_list(weights) and n > 0 do
    total_weight = Enum.sum(weights)

    if total_weight == 0 do
      # Fallback: uniform random if all weights are 0
      0..(length(weights) - 1)
      |> Enum.to_list()
      |> Enum.shuffle()
      |> Enum.take(n)
    else
      # Normalize weights to probabilities
      probs = Enum.map(weights, fn w -> w / total_weight end)

      # Build cumulative distribution
      cumulative = Enum.scan(probs, &(&1 + &2))

      # Sample n indices
      Enum.map(1..n, fn _ ->
        r = :rand.uniform()
        Enum.find_index(cumulative, fn c -> r <= c end) || length(weights) - 1
      end)
    end
  end

  @doc """
  Create a shuffled index list with character-balanced sampling.

  Instead of uniform shuffling, this biases towards underrepresented characters.
  Each index can appear multiple times (oversampling rare characters).

  ## Parameters

    - `frame_weights` - Weights per frame from `frame_weights/2`
    - `target_size` - Total number of indices to generate (typically dataset size)

  ## Example

      # With 1000 Mewtwo frames (weight 0.6) and 100 Link frames (weight 3.0)
      indices = CharacterBalance.balanced_indices(weights, 1100)
      # Link frames will appear ~5x more often than Mewtwo frames

  """
  @spec balanced_indices([float()], non_neg_integer()) :: [non_neg_integer()]
  def balanced_indices(frame_weights, target_size) do
    weighted_sample(frame_weights, target_size)
  end

  @doc """
  Log character distribution statistics.

  Useful for understanding dataset balance before and after weighting.
  """
  @spec log_distribution(%{atom() => non_neg_integer()}, %{atom() => float()}) :: :ok
  def log_distribution(counts, weights) do
    total = counts |> Map.values() |> Enum.sum()

    Logger.info("Character distribution:")

    counts
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.each(fn {char, count} ->
      pct = Float.round(count / total * 100, 1)
      weight = Map.get(weights, char, 1.0) |> Float.round(2)
      Logger.info("  #{char}: #{count} frames (#{pct}%) - weight: #{weight}x")
    end)

    :ok
  end

  @doc """
  Format character distribution for display.

  Returns a list of formatted strings for console output.
  """
  @spec format_distribution(%{atom() => non_neg_integer()}, %{atom() => float()}) :: [String.t()]
  def format_distribution(counts, weights) do
    total = counts |> Map.values() |> Enum.sum()

    counts
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.map(fn {char, count} ->
      pct = Float.round(count / total * 100, 1)
      weight = Map.get(weights, char, 1.0) |> Float.round(2)

      # Visual bar based on percentage
      bar_len = round(pct / 2)
      bar = String.duplicate("█", bar_len)

      "  #{char |> to_string() |> String.pad_trailing(15)}: #{bar} #{pct}% (#{count}) - weight: #{weight}x"
    end)
  end
end
