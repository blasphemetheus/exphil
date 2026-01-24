defmodule ExPhil.Training.PlayerRegistry do
  @moduledoc """
  Maps player tags to numeric IDs for style-conditional training.

  Player tags from replay files (e.g., "Plup", "Jmook", "Mango") are mapped to
  integer IDs (0 to max_players-1) for use in the player name embedding.

  ## Usage

      # Build registry from replay files
      {:ok, registry} = PlayerRegistry.from_replays(replay_files)

      # Look up a tag
      PlayerRegistry.get_id(registry, "Plup")  # => 0
      PlayerRegistry.get_id(registry, "Jmook") # => 1
      PlayerRegistry.get_id(registry, "unknown") # => nil (or 0 if :default_id set)

      # Use with training
      name_id = PlayerRegistry.get_id(registry, player_tag) || 0
      embedding = GameEmbed.embed(game_state, nil, port, name_id: name_id)

  ## Handling Unknown Players

  Players not in the registry return `nil` by default. Options:
  - Use a default ID (typically 0) for unknown players
  - Use a hash-based fallback to spread unknown players across IDs
  - Reject unknown players (strict mode)

  ## Persistence

  Registries can be serialized to JSON for reproducibility:

      PlayerRegistry.to_json(registry, "players.json")
      {:ok, registry} = PlayerRegistry.from_json("players.json")
  """

  alias ExPhil.Data.Peppi

  defstruct [
    :tag_to_id,
    :id_to_tag,
    :max_players,
    :unknown_strategy
  ]

  @type t :: %__MODULE__{
          tag_to_id: %{String.t() => non_neg_integer()},
          id_to_tag: %{non_neg_integer() => String.t()},
          max_players: pos_integer(),
          unknown_strategy: :nil | :default | :hash
        }

  @doc """
  Create a new empty registry.

  ## Options
    - `:max_players` - Maximum number of unique players (default: 112)
    - `:unknown_strategy` - How to handle unknown players:
      - `:nil` - Return nil for unknown (default)
      - `:default` - Return 0 for unknown
      - `:hash` - Hash unknown tags to an ID in range
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      tag_to_id: %{},
      id_to_tag: %{},
      max_players: Keyword.get(opts, :max_players, 112),
      unknown_strategy: Keyword.get(opts, :unknown_strategy, :nil)
    }
  end

  @doc """
  Build a registry from a list of replay files.

  Scans replay metadata to extract unique player tags and assigns IDs.
  Tags are sorted alphabetically for deterministic ID assignment.

  ## Options
    - `:max_players` - Maximum players (extra players get hash-based IDs)
    - `:min_games` - Minimum games a player must appear in (default: 1)
    - `:unknown_strategy` - Strategy for overflow players
  """
  @spec from_replays([Path.t()], keyword()) :: {:ok, t()} | {:error, term()}
  def from_replays(replay_files, opts \\ []) do
    max_players = Keyword.get(opts, :max_players, 112)
    min_games = Keyword.get(opts, :min_games, 1)
    unknown_strategy = Keyword.get(opts, :unknown_strategy, :hash)

    # Count games per player tag
    tag_counts =
      replay_files
      |> Task.async_stream(
        fn path ->
          case Peppi.parse(path) do
            {:ok, replay} ->
              replay.metadata.players
              |> Enum.map(& &1.tag)
              |> Enum.reject(&is_nil/1)
              |> Enum.reject(&(&1 == ""))

            {:error, _} ->
              []
          end
        end,
        max_concurrency: System.schedulers_online() * 2,
        timeout: 30_000,
        on_timeout: :kill_task
      )
      |> Enum.reduce(%{}, fn
        {:ok, tags}, acc ->
          Enum.reduce(tags, acc, fn tag, inner_acc ->
            Map.update(inner_acc, tag, 1, &(&1 + 1))
          end)

        {:exit, _}, acc ->
          acc
      end)

    # Filter by min_games and sort for deterministic ordering
    qualified_tags =
      tag_counts
      |> Enum.filter(fn {_tag, count} -> count >= min_games end)
      |> Enum.sort_by(fn {tag, count} -> {-count, tag} end)
      |> Enum.map(fn {tag, _count} -> tag end)
      |> Enum.take(max_players)

    # Build the registry
    {tag_to_id, id_to_tag} =
      qualified_tags
      |> Enum.with_index()
      |> Enum.reduce({%{}, %{}}, fn {tag, id}, {t2i, i2t} ->
        {Map.put(t2i, tag, id), Map.put(i2t, id, tag)}
      end)

    registry = %__MODULE__{
      tag_to_id: tag_to_id,
      id_to_tag: id_to_tag,
      max_players: max_players,
      unknown_strategy: unknown_strategy
    }

    {:ok, registry}
  end

  @doc """
  Build a registry from a pre-defined list of player tags.

  Useful for using a fixed player list across training runs.
  """
  @spec from_tags([String.t()], keyword()) :: t()
  def from_tags(tags, opts \\ []) do
    max_players = Keyword.get(opts, :max_players, 112)
    unknown_strategy = Keyword.get(opts, :unknown_strategy, :nil)

    tags = Enum.take(tags, max_players)

    {tag_to_id, id_to_tag} =
      tags
      |> Enum.with_index()
      |> Enum.reduce({%{}, %{}}, fn {tag, id}, {t2i, i2t} ->
        {Map.put(t2i, tag, id), Map.put(i2t, id, tag)}
      end)

    %__MODULE__{
      tag_to_id: tag_to_id,
      id_to_tag: id_to_tag,
      max_players: max_players,
      unknown_strategy: unknown_strategy
    }
  end

  @doc """
  Get the numeric ID for a player tag.

  Returns `nil` if the tag is not in the registry (unless unknown_strategy is set).
  """
  @spec get_id(t(), String.t() | nil) :: non_neg_integer() | nil
  def get_id(_registry, nil), do: nil
  def get_id(_registry, ""), do: nil

  def get_id(%__MODULE__{} = registry, tag) when is_binary(tag) do
    case Map.get(registry.tag_to_id, tag) do
      nil -> handle_unknown(registry, tag)
      id -> id
    end
  end

  @doc """
  Get the player tag for a numeric ID.
  """
  @spec get_tag(t(), non_neg_integer()) :: String.t() | nil
  def get_tag(%__MODULE__{} = registry, id) when is_integer(id) do
    Map.get(registry.id_to_tag, id)
  end

  @doc """
  Add a player tag to the registry, returning the assigned ID.

  If the tag already exists, returns its existing ID.
  If the registry is full, uses the unknown strategy.
  """
  @spec add_tag(t(), String.t()) :: {t(), non_neg_integer()}
  def add_tag(%__MODULE__{} = registry, tag) when is_binary(tag) do
    case Map.get(registry.tag_to_id, tag) do
      nil ->
        next_id = map_size(registry.tag_to_id)

        if next_id < registry.max_players do
          new_registry = %{
            registry
            | tag_to_id: Map.put(registry.tag_to_id, tag, next_id),
              id_to_tag: Map.put(registry.id_to_tag, next_id, tag)
          }

          {new_registry, next_id}
        else
          # Registry full, use unknown strategy
          id = handle_unknown(registry, tag) || 0
          {registry, id}
        end

      existing_id ->
        {registry, existing_id}
    end
  end

  @doc """
  Get the number of registered players.
  """
  @spec size(t()) :: non_neg_integer()
  def size(%__MODULE__{} = registry) do
    map_size(registry.tag_to_id)
  end

  @doc """
  List all registered tags sorted by ID.
  """
  @spec list_tags(t()) :: [String.t()]
  def list_tags(%__MODULE__{} = registry) do
    registry.id_to_tag
    |> Enum.sort_by(fn {id, _tag} -> id end)
    |> Enum.map(fn {_id, tag} -> tag end)
  end

  @doc """
  Check if a tag is in the registry.
  """
  @spec has_tag?(t(), String.t()) :: boolean()
  def has_tag?(%__MODULE__{} = registry, tag) do
    Map.has_key?(registry.tag_to_id, tag)
  end

  @doc """
  Serialize registry to JSON.
  """
  @spec to_json(t(), Path.t()) :: :ok | {:error, term()}
  def to_json(%__MODULE__{} = registry, path) do
    data = %{
      "version" => 1,
      "max_players" => registry.max_players,
      "unknown_strategy" => Atom.to_string(registry.unknown_strategy),
      "players" => list_tags(registry)
    }

    case Jason.encode(data, pretty: true) do
      {:ok, json} -> File.write(path, json)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Load registry from JSON.
  """
  @spec from_json(Path.t()) :: {:ok, t()} | {:error, term()}
  def from_json(path) do
    with {:ok, content} <- File.read(path),
         {:ok, data} <- Jason.decode(content) do
      unknown_strategy =
        case data["unknown_strategy"] do
          "nil" -> :nil
          "default" -> :default
          "hash" -> :hash
          _ -> :nil
        end

      registry =
        from_tags(
          data["players"],
          max_players: data["max_players"],
          unknown_strategy: unknown_strategy
        )

      {:ok, registry}
    end
  end

  # Private functions

  defp handle_unknown(%__MODULE__{unknown_strategy: :nil}, _tag), do: nil
  defp handle_unknown(%__MODULE__{unknown_strategy: :default}, _tag), do: 0

  defp handle_unknown(%__MODULE__{unknown_strategy: :hash, max_players: max}, tag) do
    # Use erlang phash2 for consistent hashing
    :erlang.phash2(tag, max)
  end
end
