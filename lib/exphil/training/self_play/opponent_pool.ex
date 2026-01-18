defmodule ExPhil.Training.SelfPlay.OpponentPool do
  @moduledoc """
  Manages a pool of opponents for self-play training.

  The pool contains:
  - Current policy (self-play)
  - Historical checkpoints (prevent forgetting)
  - CPU opponents (diverse play styles)

  ## Opponent Selection

  Opponents are sampled according to configurable weights:

      config = OpponentPool.default_config()
      # %{current: 0.4, historical: 0.3, cpu: 0.2, random: 0.1}

      {:ok, pool} = OpponentPool.new(config)
      opponent = OpponentPool.sample(pool)

  ## Usage

      # Create pool
      {:ok, pool} = OpponentPool.new()

      # Add current policy as potential opponent
      pool = OpponentPool.set_current(pool, policy_params)

      # Snapshot current to history (after training iteration)
      pool = OpponentPool.snapshot(pool, "v1")

      # Sample an opponent
      {opponent_type, opponent} = OpponentPool.sample(pool)
      # => {:historical, %{params: ..., version: "v1"}}
      # => {:cpu, %{level: 7}}
      # => {:current, %{params: ...}}

  """

  use GenServer
  require Logger

  defstruct [
    :current_params,       # Current policy parameters
    :historical,           # List of {version, params} tuples
    :cpu_levels,           # List of CPU levels to use
    :config,               # Sampling configuration
    :win_rates,            # Map of opponent_id => win_rate (for prioritized sampling)
    :max_historical        # Max historical checkpoints to keep
  ]

  @type opponent_type :: :current | :historical | :cpu | :random
  @type opponent :: %{type: opponent_type, params: map() | nil, level: integer() | nil, version: String.t() | nil}
  @type t :: %__MODULE__{}

  # Default sampling weights
  @default_config %{
    current: 0.4,      # 40% current self-play
    historical: 0.3,   # 30% recent historical
    cpu: 0.2,          # 20% CPU opponents
    random: 0.1        # 10% random from full history
  }

  @default_cpu_levels [5, 6, 7, 8, 9]
  @default_max_historical 10

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Default configuration for opponent sampling.
  """
  @spec default_config() :: map()
  def default_config, do: @default_config

  @doc """
  Create a new opponent pool.

  ## Options
    - `:config` - Sampling weights (default: 40% current, 30% historical, 20% CPU, 10% random)
    - `:cpu_levels` - CPU levels to include (default: [5, 6, 7, 8, 9])
    - `:max_historical` - Max historical checkpoints to keep (default: 10)
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts \\ []) do
    config = Keyword.get(opts, :config, @default_config)
    cpu_levels = Keyword.get(opts, :cpu_levels, @default_cpu_levels)
    max_historical = Keyword.get(opts, :max_historical, @default_max_historical)

    pool = %__MODULE__{
      current_params: nil,
      historical: [],
      cpu_levels: cpu_levels,
      config: config,
      win_rates: %{},
      max_historical: max_historical
    }

    {:ok, pool}
  end

  @doc """
  Start the opponent pool as a GenServer for concurrent access.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Set the current policy parameters.
  """
  @spec set_current(t(), map()) :: t()
  def set_current(%__MODULE__{} = pool, params) do
    %{pool | current_params: params}
  end

  @doc """
  Snapshot current policy to historical pool.

  Called after each training iteration to save a copy for future opponents.
  """
  @spec snapshot(t(), String.t()) :: t()
  def snapshot(%__MODULE__{current_params: nil} = pool, _version) do
    Logger.warning("Cannot snapshot: no current params set")
    pool
  end

  def snapshot(%__MODULE__{} = pool, version) do
    # Deep copy params to avoid mutation issues
    params_copy = deep_copy(pool.current_params)
    entry = {version, params_copy, System.system_time(:second)}

    # Add to front, trim if over limit
    historical = [entry | pool.historical]
    |> Enum.take(pool.max_historical)

    Logger.info("Snapshotted policy version #{version} (#{length(historical)} in pool)")

    %{pool | historical: historical}
  end

  @doc """
  Sample an opponent according to the configured weights.

  Returns `{type, opponent_data}` where:
  - `:current` -> `%{params: current_params}`
  - `:historical` -> `%{params: params, version: version}`
  - `:cpu` -> `%{level: cpu_level}`
  - `:random` -> Same as historical but from full history
  """
  @spec sample(t()) :: {opponent_type(), opponent()}
  def sample(%__MODULE__{} = pool) do
    # Normalize weights for available opponent types
    weights = normalize_weights(pool)

    # Sample category
    category = weighted_sample(weights)

    # Get opponent from category
    case category do
      :current ->
        {:current, %{type: :current, params: pool.current_params, level: nil, version: nil}}

      :historical ->
        sample_historical(pool, :recent)

      :cpu ->
        level = Enum.random(pool.cpu_levels)
        {:cpu, %{type: :cpu, params: nil, level: level, version: nil}}

      :random ->
        sample_historical(pool, :any)
    end
  end

  @doc """
  Record a game result for win rate tracking.

  Used for prioritized opponent sampling.
  """
  @spec record_result(t(), String.t(), :win | :loss | :draw) :: t()
  def record_result(%__MODULE__{} = pool, opponent_id, result) do
    current = Map.get(pool.win_rates, opponent_id, %{wins: 0, losses: 0, draws: 0})

    updated = case result do
      :win -> %{current | wins: current.wins + 1}
      :loss -> %{current | losses: current.losses + 1}
      :draw -> %{current | draws: current.draws + 1}
    end

    %{pool | win_rates: Map.put(pool.win_rates, opponent_id, updated)}
  end

  @doc """
  Get win rate against a specific opponent.
  """
  @spec get_win_rate(t(), String.t()) :: float()
  def get_win_rate(%__MODULE__{} = pool, opponent_id) do
    case Map.get(pool.win_rates, opponent_id) do
      nil -> 0.5  # Unknown opponent, assume 50%
      %{wins: w, losses: l, draws: d} ->
        total = w + l + d
        if total == 0, do: 0.5, else: (w + d * 0.5) / total
    end
  end

  @doc """
  List all available opponents with their types.
  """
  @spec list_opponents(t()) :: [%{type: opponent_type(), id: String.t()}]
  def list_opponents(%__MODULE__{} = pool) do
    current = if pool.current_params, do: [%{type: :current, id: "current"}], else: []

    historical = Enum.map(pool.historical, fn {version, _params, _time} ->
      %{type: :historical, id: version}
    end)

    cpu = Enum.map(pool.cpu_levels, fn level ->
      %{type: :cpu, id: "cpu_#{level}"}
    end)

    current ++ historical ++ cpu
  end

  @doc """
  Load historical checkpoints from a directory.
  """
  @spec load_from_directory(t(), String.t()) :: t()
  def load_from_directory(%__MODULE__{} = pool, dir) do
    checkpoints = Path.wildcard(Path.join(dir, "*.bin"))
    |> Enum.sort_by(&File.stat!(&1).mtime, :desc)
    |> Enum.take(pool.max_historical)

    historical = Enum.map(checkpoints, fn path ->
      version = Path.basename(path, ".bin")
      {:ok, policy} = ExPhil.Training.load_policy(path)
      {version, policy.params, File.stat!(path).mtime |> DateTime.to_unix()}
    end)

    Logger.info("Loaded #{length(historical)} checkpoints from #{dir}")

    %{pool | historical: historical}
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    {:ok, pool} = new(opts)
    {:ok, pool}
  end

  @impl true
  def handle_call(:get, _from, pool) do
    {:reply, pool, pool}
  end

  @impl true
  def handle_call({:set_current, params}, _from, pool) do
    new_pool = set_current(pool, params)
    {:reply, :ok, new_pool}
  end

  @impl true
  def handle_call({:snapshot, version}, _from, pool) do
    new_pool = snapshot(pool, version)
    {:reply, :ok, new_pool}
  end

  @impl true
  def handle_call(:sample, _from, pool) do
    opponent = sample(pool)
    {:reply, opponent, pool}
  end

  @impl true
  def handle_call({:record_result, opponent_id, result}, _from, pool) do
    new_pool = record_result(pool, opponent_id, result)
    {:reply, :ok, new_pool}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp normalize_weights(%__MODULE__{} = pool) do
    config = pool.config

    # Disable categories that have no opponents
    weights = %{
      current: if(pool.current_params, do: config.current, else: 0),
      historical: if(length(pool.historical) > 0, do: config.historical, else: 0),
      cpu: if(length(pool.cpu_levels) > 0, do: config.cpu, else: 0),
      random: if(length(pool.historical) > 0, do: config.random, else: 0)
    }

    # Normalize to sum to 1
    total = Enum.sum(Map.values(weights))

    if total == 0 do
      # Fallback: if nothing available, use CPU
      %{current: 0, historical: 0, cpu: 1.0, random: 0}
    else
      Map.new(weights, fn {k, v} -> {k, v / total} end)
    end
  end

  defp weighted_sample(weights) do
    r = :rand.uniform()

    weights
    |> Enum.reduce_while({0.0, nil}, fn {category, weight}, {cumsum, _} ->
      new_cumsum = cumsum + weight
      if r <= new_cumsum do
        {:halt, {new_cumsum, category}}
      else
        {:cont, {new_cumsum, category}}
      end
    end)
    |> elem(1)
    |> case do
      nil -> :cpu  # Fallback
      category -> category
    end
  end

  defp sample_historical(%__MODULE__{historical: []}, _mode) do
    # No historical, fallback to CPU
    {:cpu, %{type: :cpu, params: nil, level: 7, version: nil}}
  end

  defp sample_historical(%__MODULE__{historical: historical}, :recent) do
    # Sample from recent half
    recent_count = max(1, div(length(historical), 2))
    recent = Enum.take(historical, recent_count)
    {version, params, _time} = Enum.random(recent)
    {:historical, %{type: :historical, params: params, level: nil, version: version}}
  end

  defp sample_historical(%__MODULE__{historical: historical}, :any) do
    # Sample from full history uniformly
    {version, params, _time} = Enum.random(historical)
    {:historical, %{type: :historical, params: params, level: nil, version: version}}
  end

  defp deep_copy(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp deep_copy(%Axon.ModelState{} = state) do
    %{state |
      data: deep_copy(state.data),
      state: deep_copy(state.state)
    }
  end
  defp deep_copy(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_copy(v)} end)
  end
  defp deep_copy(other), do: other
end
