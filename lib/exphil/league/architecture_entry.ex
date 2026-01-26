defmodule ExPhil.League.ArchitectureEntry do
  @moduledoc """
  Data structure representing an architecture entry in the league.

  Each entry tracks a specific neural network architecture competing in the
  architecture league, including its model definition, trained parameters,
  performance metrics, and lineage history.

  ## Fields

  - `:id` - Unique identifier (e.g., `:mamba_mewtwo`)
  - `:architecture` - Architecture type (`:mlp`, `:lstm`, `:gru`, `:mamba`, `:attention`, `:jamba`)
  - `:character` - Character this architecture is trained for (default: `:mewtwo`)
  - `:model` - Compiled Axon model
  - `:params` - Current trained parameters
  - `:generation` - Training generation/iteration count
  - `:elo` - Current Elo rating
  - `:config` - Architecture-specific configuration
  - `:lineage` - List of previous version IDs
  - `:stats` - Win/loss/draw statistics

  ## Example

      %ArchitectureEntry{
        id: :mamba_mewtwo,
        architecture: :mamba,
        character: :mewtwo,
        model: %Axon{},
        params: %{},
        generation: 5,
        elo: 1450.0,
        config: %{
          hidden_size: 256,
          window_size: 30,
          num_layers: 2,
          state_size: 16
        },
        lineage: ["mamba_mewtwo_v0", "mamba_mewtwo_v1", "mamba_mewtwo_v2"],
        stats: %{wins: 42, losses: 38, draws: 5, total_frames: 450000}
      }

  """

  alias ExPhil.SelfPlay.Elo

  defstruct [
    :id,
    :architecture,
    :character,
    :model,
    :params,
    :generation,
    :elo,
    :config,
    :lineage,
    :stats,
    :created_at,
    :updated_at
  ]

  @type architecture :: :mlp | :lstm | :gru | :mamba | :attention | :jamba
  @type character :: :mewtwo | :ganondorf | :link | :game_and_watch | :zelda | atom()

  @type t :: %__MODULE__{
          id: atom(),
          architecture: architecture(),
          character: character(),
          model: Axon.t() | nil,
          params: map(),
          generation: non_neg_integer(),
          elo: float(),
          config: map(),
          lineage: [String.t()],
          stats: map(),
          created_at: integer(),
          updated_at: integer()
        }

  @supported_architectures [:mlp, :lstm, :gru, :mamba, :attention, :jamba]

  # ============================================================================
  # Constructor
  # ============================================================================

  @doc """
  Create a new architecture entry.

  ## Options

  - `:id` - Unique identifier (required)
  - `:architecture` - Architecture type (required)
  - `:character` - Target character (default: `:mewtwo`)
  - `:config` - Architecture-specific config (default: architecture defaults)
  - `:model` - Pre-built Axon model (optional, built from config if not provided)
  - `:params` - Initial parameters (optional)
  - `:elo` - Initial Elo rating (default: 1000.0)

  ## Example

      {:ok, entry} = ArchitectureEntry.new(
        id: :mamba_mewtwo,
        architecture: :mamba,
        config: %{hidden_size: 256, num_layers: 2}
      )

  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) do
    id = Keyword.fetch!(opts, :id)
    architecture = Keyword.fetch!(opts, :architecture)

    unless architecture in @supported_architectures do
      {:error, {:unsupported_architecture, architecture, @supported_architectures}}
    else
      character = Keyword.get(opts, :character, :mewtwo)
      config = Keyword.get(opts, :config, default_config(architecture))
      model = Keyword.get(opts, :model)
      params = Keyword.get(opts, :params, %{})
      elo = Keyword.get(opts, :elo, Elo.initial_rating()) |> ensure_float()

      now = System.system_time(:second)

      entry = %__MODULE__{
        id: id,
        architecture: architecture,
        character: character,
        model: model,
        params: params,
        generation: 0,
        elo: elo,
        config: Map.merge(default_config(architecture), config),
        lineage: [],
        stats: init_stats(),
        created_at: now,
        updated_at: now
      }

      {:ok, entry}
    end
  end

  @doc """
  Create a new architecture entry, raising on error.
  """
  @spec new!(keyword()) :: t()
  def new!(opts) do
    case new(opts) do
      {:ok, entry} ->
        entry

      {:error, reason} ->
        raise ArgumentError, "Failed to create ArchitectureEntry: #{inspect(reason)}"
    end
  end

  # ============================================================================
  # Updates
  # ============================================================================

  @doc """
  Update the entry after training (new params, increment generation).
  """
  @spec update_from_training(t(), map()) :: t()
  def update_from_training(%__MODULE__{} = entry, new_params) do
    version_id = "#{entry.id}_v#{entry.generation}"

    %{
      entry
      | params: new_params,
        generation: entry.generation + 1,
        lineage: entry.lineage ++ [version_id],
        updated_at: System.system_time(:second)
    }
  end

  @doc """
  Update Elo rating after a match.
  """
  @spec update_elo(t(), float()) :: t()
  def update_elo(%__MODULE__{} = entry, new_elo) do
    %{entry | elo: ensure_float(new_elo), updated_at: System.system_time(:second)}
  end

  @doc """
  Record a match result in stats.
  """
  @spec record_result(t(), :win | :loss | :draw, non_neg_integer()) :: t()
  def record_result(%__MODULE__{} = entry, result, frames \\ 0) do
    stats =
      case result do
        :win -> %{entry.stats | wins: entry.stats.wins + 1}
        :loss -> %{entry.stats | losses: entry.stats.losses + 1}
        :draw -> %{entry.stats | draws: entry.stats.draws + 1}
      end

    stats = %{stats | total_frames: stats.total_frames + frames}

    %{entry | stats: stats, updated_at: System.system_time(:second)}
  end

  @doc """
  Set the model and optionally initialize params.
  """
  @spec set_model(t(), Axon.t(), map() | nil) :: t()
  def set_model(%__MODULE__{} = entry, model, params \\ nil) do
    %{
      entry
      | model: model,
        params: params || entry.params,
        updated_at: System.system_time(:second)
    }
  end

  # ============================================================================
  # Queries
  # ============================================================================

  @doc """
  Get the win rate for this architecture.
  """
  @spec win_rate(t()) :: float()
  def win_rate(%__MODULE__{stats: stats}) do
    total = stats.wins + stats.losses + stats.draws
    if total > 0, do: stats.wins / total, else: 0.5
  end

  @doc """
  Get total games played.
  """
  @spec games_played(t()) :: non_neg_integer()
  def games_played(%__MODULE__{stats: stats}) do
    stats.wins + stats.losses + stats.draws
  end

  @doc """
  Check if the architecture has been trained (has params).
  """
  @spec trained?(t()) :: boolean()
  def trained?(%__MODULE__{params: params}) do
    params != nil and map_size(params) > 0
  end

  @doc """
  Check if the architecture has a model built.
  """
  @spec has_model?(t()) :: boolean()
  def has_model?(%__MODULE__{model: model}) do
    model != nil
  end

  @doc """
  Get supported architecture types.
  """
  @spec supported_architectures() :: [architecture()]
  def supported_architectures, do: @supported_architectures

  # ============================================================================
  # Serialization
  # ============================================================================

  @doc """
  Convert to map for serialization (excluding model and params).
  """
  @spec to_metadata(t()) :: map()
  def to_metadata(%__MODULE__{} = entry) do
    %{
      id: entry.id,
      architecture: entry.architecture,
      character: entry.character,
      generation: entry.generation,
      elo: entry.elo,
      config: entry.config,
      lineage: entry.lineage,
      stats: entry.stats,
      created_at: entry.created_at,
      updated_at: entry.updated_at
    }
  end

  @doc """
  Create entry from metadata (model and params must be loaded separately).
  """
  @spec from_metadata(map()) :: {:ok, t()} | {:error, term()}
  def from_metadata(metadata) when is_map(metadata) do
    entry = %__MODULE__{
      id: to_atom(metadata[:id] || metadata["id"]),
      architecture: to_atom(metadata[:architecture] || metadata["architecture"]),
      character: to_atom(metadata[:character] || metadata["character"]),
      model: nil,
      params: %{},
      generation: metadata[:generation] || metadata["generation"] || 0,
      elo: ensure_float(metadata[:elo] || metadata["elo"] || Elo.initial_rating()),
      config: to_atom_keys(metadata[:config] || metadata["config"] || %{}),
      lineage: metadata[:lineage] || metadata["lineage"] || [],
      stats: to_atom_keys(metadata[:stats] || metadata["stats"] || init_stats()),
      created_at: metadata[:created_at] || metadata["created_at"] || System.system_time(:second),
      updated_at: metadata[:updated_at] || metadata["updated_at"] || System.system_time(:second)
    }

    {:ok, entry}
  end

  # ============================================================================
  # Default Configurations
  # ============================================================================

  @doc """
  Get default configuration for an architecture type.
  """
  @spec default_config(architecture()) :: map()
  def default_config(:mlp) do
    %{
      hidden_sizes: [256, 256],
      dropout: 0.1,
      activation: :relu,
      layer_norm: false
    }
  end

  def default_config(:lstm) do
    %{
      hidden_size: 256,
      num_layers: 2,
      dropout: 0.1,
      window_size: 30
    }
  end

  def default_config(:gru) do
    %{
      hidden_size: 256,
      num_layers: 2,
      dropout: 0.1,
      window_size: 30
    }
  end

  def default_config(:mamba) do
    %{
      hidden_size: 256,
      num_layers: 2,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      window_size: 30
    }
  end

  def default_config(:attention) do
    %{
      hidden_size: 256,
      num_layers: 2,
      num_heads: 4,
      head_dim: 64,
      dropout: 0.1,
      window_size: 30
    }
  end

  def default_config(:jamba) do
    %{
      hidden_size: 256,
      num_layers: 4,
      state_size: 16,
      num_heads: 4,
      attention_every: 3,
      dropout: 0.1,
      window_size: 30
    }
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp init_stats do
    %{
      wins: 0,
      losses: 0,
      draws: 0,
      total_frames: 0
    }
  end

  defp ensure_float(value) when is_float(value), do: value
  defp ensure_float(value) when is_integer(value), do: value * 1.0
  defp ensure_float(value), do: value

  defp to_atom(value) when is_atom(value), do: value
  defp to_atom(value) when is_binary(value), do: String.to_atom(value)

  defp to_atom_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {String.to_atom(k), v}
      {k, v} -> {k, v}
    end)
  end
end
