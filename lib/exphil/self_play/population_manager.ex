defmodule ExPhil.SelfPlay.PopulationManager do
  @moduledoc """
  GenServer managing the population of policies for self-play training.

  Tracks the current policy and maintains a history of past policies
  for opponent sampling. Supports different sampling strategies to
  ensure diverse training.

  ## Architecture

      ┌────────────────────────────────────────────────────────────────────┐
      │                     PopulationManager                               │
      │                                                                     │
      │  ┌──────────────────────────────────────────────────────────────┐  │
      │  │                  Current Policy                               │  │
      │  │  {model, params, generation: 42}                             │  │
      │  └──────────────────────────────────────────────────────────────┘  │
      │                            │                                        │
      │                     snapshot()                                      │
      │                            ▼                                        │
      │  ┌──────────────────────────────────────────────────────────────┐  │
      │  │                Historical Policies                            │  │
      │  │  [                                                            │  │
      │  │    {v42, params, timestamp},                                  │  │
      │  │    {v41, params, timestamp},                                  │  │
      │  │    ...                                                        │  │
      │  │  ]                                                            │  │
      │  └──────────────────────────────────────────────────────────────┘  │
      │                            │                                        │
      │                    sample_opponent()                                │
      │                            ▼                                        │
      │  ┌──────────────────────────────────────────────────────────────┐  │
      │  │               Sampling Strategies                             │  │
      │  │  - :current (self-play)                                       │  │
      │  │  - :historical (past versions)                                │  │
      │  │  - :cpu (builtin CPU)                                         │  │
      │  │  - :uniform (random from all)                                 │  │
      │  └──────────────────────────────────────────────────────────────┘  │
      │                                                                     │
      └────────────────────────────────────────────────────────────────────┘

  ## Usage

      # Start the manager
      {:ok, manager} = PopulationManager.start_link(
        max_history_size: 20,
        history_sample_prob: 0.3
      )

      # Set the current policy
      :ok = PopulationManager.set_current(manager, model, params)

      # Snapshot current to history
      :ok = PopulationManager.snapshot(manager)

      # Sample an opponent
      {:ok, {policy_id, policy}} = PopulationManager.sample_opponent(manager)

      # Get specific policy
      {:ok, policy} = PopulationManager.get_policy(manager, :current)

  """

  use GenServer

  require Logger

  defstruct [
    # Current model architecture (Axon.t())
    :model,
    # Current policy parameters
    :current_params,
    # Current generation number
    :current_generation,
    # List of {version_id, params, timestamp}
    :historical_policies,
    :max_history_size,
    :history_sample_prob,
    :cpu_levels,
    :sampling_weights,
    :stats
  ]

  @type policy_id :: :current | :cpu | {:historical, term()}
  @type policy :: {Axon.t(), map()}

  @default_opts %{
    max_history_size: 20,
    history_sample_prob: 0.3,
    cpu_levels: [5, 6, 7, 8, 9],
    sampling_weights: %{
      current: 0.4,
      historical: 0.3,
      cpu: 0.2,
      uniform: 0.1
    }
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the PopulationManager.

  ## Options
    - `:max_history_size` - Maximum historical policies to keep (default: 20)
    - `:history_sample_prob` - Probability of sampling from history (default: 0.3)
    - `:cpu_levels` - CPU levels to use (default: [5, 6, 7, 8, 9])
    - `:sampling_weights` - Weights for different opponent types
    - `:name` - Name for the GenServer
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Sets the current policy.
  """
  def set_current(manager, model, params) do
    GenServer.call(manager, {:set_current, model, params})
  end

  @doc """
  Updates only the current params (model stays the same).
  """
  def update_params(manager, params) do
    GenServer.call(manager, {:update_params, params})
  end

  @doc """
  Gets the current policy.
  """
  def get_current(manager) do
    GenServer.call(manager, :get_current)
  end

  @doc """
  Snapshots the current policy to history.

  The current policy is copied and added to the historical pool.
  """
  def snapshot(manager) do
    GenServer.call(manager, :snapshot)
  end

  @doc """
  Samples an opponent according to configured weights.

  Returns `{:ok, {policy_id, {model, params}}}` or `{:ok, {:cpu, level}}`.
  """
  def sample_opponent(manager, opts \\ []) do
    GenServer.call(manager, {:sample_opponent, opts})
  end

  @doc """
  Gets a specific policy by ID.

  Policy IDs:
  - `:current` - Current policy
  - `{:historical, version}` - Historical policy by version
  - `:cpu` - Returns nil (indicates CPU opponent)
  """
  def get_policy(manager, policy_id) do
    GenServer.call(manager, {:get_policy, policy_id})
  end

  @doc """
  Lists all available policies.
  """
  def list_policies(manager) do
    GenServer.call(manager, :list_policies)
  end

  @doc """
  Gets population statistics.
  """
  def get_stats(manager) do
    GenServer.call(manager, :get_stats)
  end

  @doc """
  Adds an external policy to the historical pool.

  Used for loading pretrained checkpoints.
  """
  def add_policy(manager, version_id, model, params) do
    GenServer.call(manager, {:add_policy, version_id, model, params})
  end

  @doc """
  Loads historical policies from a checkpoint directory.
  """
  def load_from_directory(manager, directory) do
    GenServer.call(manager, {:load_from_directory, directory}, 60_000)
  end

  @doc """
  Saves the current policy to a file.
  """
  def save_current(manager, path) do
    GenServer.call(manager, {:save_current, path})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    state = %__MODULE__{
      model: nil,
      current_params: nil,
      current_generation: 0,
      historical_policies: [],
      max_history_size: Keyword.get(opts, :max_history_size, @default_opts.max_history_size),
      history_sample_prob:
        Keyword.get(opts, :history_sample_prob, @default_opts.history_sample_prob),
      cpu_levels: Keyword.get(opts, :cpu_levels, @default_opts.cpu_levels),
      sampling_weights: Keyword.get(opts, :sampling_weights, @default_opts.sampling_weights),
      stats: init_stats()
    }

    Logger.debug("[PopulationManager] Started with max_history=#{state.max_history_size}")

    {:ok, state}
  end

  @impl true
  def handle_call({:set_current, model, params}, _from, state) do
    new_state = %{
      state
      | model: model,
        current_params: params,
        current_generation: state.current_generation + 1
    }

    Logger.debug(
      "[PopulationManager] Set current policy (generation #{new_state.current_generation})"
    )

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:update_params, params}, _from, state) do
    new_state = %{
      state
      | current_params: params,
        current_generation: state.current_generation + 1
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:get_current, _from, state) do
    if state.model && state.current_params do
      {:reply, {:ok, {state.model, state.current_params}}, state}
    else
      {:reply, {:error, :no_current_policy}, state}
    end
  end

  @impl true
  def handle_call(:snapshot, _from, %{current_params: nil} = state) do
    Logger.warning("[PopulationManager] Cannot snapshot: no current policy")
    {:reply, {:error, :no_current_policy}, state}
  end

  @impl true
  def handle_call(:snapshot, _from, state) do
    version_id = "v#{state.current_generation}"
    timestamp = System.system_time(:second)

    # Deep copy params to avoid mutation
    params_copy = deep_copy_params(state.current_params)

    entry = {version_id, state.model, params_copy, timestamp}

    # Add to front, trim if over limit
    new_history =
      [entry | state.historical_policies]
      |> Enum.take(state.max_history_size)

    new_stats = update_stats(state.stats, :snapshot)

    Logger.info(
      "[PopulationManager] Snapshotted #{version_id} (#{length(new_history)} in history)"
    )

    {:reply, :ok, %{state | historical_policies: new_history, stats: new_stats}}
  end

  @impl true
  def handle_call({:sample_opponent, opts}, _from, state) do
    strategy = Keyword.get(opts, :strategy, :weighted)

    {policy_id, result} =
      case strategy do
        :weighted -> weighted_sample(state)
        :current_only -> sample_current(state)
        :historical_only -> sample_historical(state)
        :cpu_only -> sample_cpu(state)
        _ -> weighted_sample(state)
      end

    new_stats = update_stats(state.stats, {:sample, policy_id})

    {:reply, {:ok, {policy_id, result}}, %{state | stats: new_stats}}
  end

  @impl true
  def handle_call({:get_policy, :current}, _from, state) do
    if state.model && state.current_params do
      {:reply, {:ok, {state.model, state.current_params}}, state}
    else
      {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:get_policy, :cpu}, _from, state) do
    level = Enum.random(state.cpu_levels)
    {:reply, {:ok, {:cpu, level}}, state}
  end

  @impl true
  def handle_call({:get_policy, {:historical, version}}, _from, state) do
    case Enum.find(state.historical_policies, fn {v, _, _, _} -> v == version end) do
      {_v, model, params, _ts} -> {:reply, {:ok, {model, params}}, state}
      nil -> {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:get_policy, version_id}, _from, state) when is_binary(version_id) do
    case Enum.find(state.historical_policies, fn {v, _, _, _} -> v == version_id end) do
      {_v, model, params, _ts} -> {:reply, {:ok, {model, params}}, state}
      nil -> {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:list_policies, _from, state) do
    current =
      if state.current_params do
        [%{id: :current, generation: state.current_generation, type: :current}]
      else
        []
      end

    historical =
      Enum.map(state.historical_policies, fn {version, _model, _params, ts} ->
        %{id: {:historical, version}, version: version, type: :historical, timestamp: ts}
      end)

    cpu =
      Enum.map(state.cpu_levels, fn level ->
        %{id: {:cpu, level}, level: level, type: :cpu}
      end)

    {:reply, current ++ historical ++ cpu, state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats =
      Map.merge(state.stats, %{
        current_generation: state.current_generation,
        history_size: length(state.historical_policies),
        has_current: state.current_params != nil
      })

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:add_policy, version_id, model, params}, _from, state) do
    timestamp = System.system_time(:second)
    params_copy = deep_copy_params(params)

    entry = {version_id, model, params_copy, timestamp}

    new_history =
      [entry | state.historical_policies]
      |> Enum.take(state.max_history_size)

    Logger.info("[PopulationManager] Added external policy #{version_id}")

    {:reply, :ok, %{state | historical_policies: new_history}}
  end

  @impl true
  def handle_call({:load_from_directory, directory}, _from, state) do
    {:ok, loaded_policies} = load_checkpoints_from_dir(directory, state)

    new_history =
      (loaded_policies ++ state.historical_policies)
      |> Enum.take(state.max_history_size)

    Logger.info(
      "[PopulationManager] Loaded #{length(loaded_policies)} policies from #{directory}"
    )

    {:reply, {:ok, length(loaded_policies)}, %{state | historical_policies: new_history}}
  end

  @impl true
  def handle_call({:save_current, path}, _from, state) do
    if state.current_params do
      export = %{
        model: state.model,
        params: to_binary_backend(state.current_params),
        generation: state.current_generation
      }

      dir = Path.dirname(path)
      File.mkdir_p!(dir)

      case File.write(path, :erlang.term_to_binary(export)) do
        :ok ->
          Logger.info("[PopulationManager] Saved current policy to #{path}")
          {:reply, :ok, state}

        {:error, _reason} = error ->
          {:reply, error, state}
      end
    else
      {:reply, {:error, :no_current_policy}, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp weighted_sample(state) do
    weights = normalize_weights(state)
    category = sample_category(weights)

    case category do
      :current -> sample_current(state)
      :historical -> sample_historical(state)
      :cpu -> sample_cpu(state)
      :uniform -> sample_uniform(state)
    end
  end

  defp sample_current(%{current_params: nil} = state), do: sample_cpu(state)

  defp sample_current(state) do
    {:current, {state.model, state.current_params}}
  end

  defp sample_historical(%{historical_policies: []} = state), do: sample_cpu(state)

  defp sample_historical(state) do
    # Sample from recent half with higher probability
    recent_count = max(1, div(length(state.historical_policies), 2))

    policy =
      if :rand.uniform() < 0.7 do
        # 70% recent
        state.historical_policies
        |> Enum.take(recent_count)
        |> Enum.random()
      else
        # 30% any
        Enum.random(state.historical_policies)
      end

    {version, model, params, _ts} = policy
    {{:historical, version}, {model, params}}
  end

  defp sample_cpu(state) do
    level = Enum.random(state.cpu_levels)
    {{:cpu, level}, {:cpu, level}}
  end

  defp sample_uniform(state) do
    all_options =
      [
        if(state.current_params, do: {:current, {state.model, state.current_params}}),
        Enum.map(state.historical_policies, fn {v, m, p, _ts} ->
          {{:historical, v}, {m, p}}
        end),
        Enum.map(state.cpu_levels, fn l -> {{:cpu, l}, {:cpu, l}} end)
      ]
      |> List.flatten()
      |> Enum.reject(&is_nil/1)

    if length(all_options) > 0 do
      Enum.random(all_options)
    else
      sample_cpu(state)
    end
  end

  defp normalize_weights(state) do
    weights = state.sampling_weights

    # Disable categories that aren't available
    adjusted = %{
      current: if(state.current_params, do: weights[:current] || 0, else: 0),
      historical:
        if(length(state.historical_policies) > 0, do: weights[:historical] || 0, else: 0),
      cpu: if(length(state.cpu_levels) > 0, do: weights[:cpu] || 0, else: 0),
      uniform: weights[:uniform] || 0
    }

    total = Enum.sum(Map.values(adjusted))

    if total == 0 do
      %{current: 0, historical: 0, cpu: 1.0, uniform: 0}
    else
      Map.new(adjusted, fn {k, v} -> {k, v / total} end)
    end
  end

  defp sample_category(weights) do
    r = :rand.uniform()

    weights
    |> Enum.reduce_while({0.0, :cpu}, fn {category, weight}, {cumsum, _} ->
      new_cumsum = cumsum + weight

      if r <= new_cumsum do
        {:halt, {new_cumsum, category}}
      else
        {:cont, {new_cumsum, category}}
      end
    end)
    |> elem(1)
  end

  defp deep_copy_params(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp deep_copy_params(%Axon.ModelState{} = state) do
    %{state | data: deep_copy_params(state.data), state: deep_copy_params(state.state)}
  end

  defp deep_copy_params(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_copy_params(v)} end)
  end

  defp deep_copy_params(other), do: other

  defp to_binary_backend(%Nx.Tensor{} = tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defp to_binary_backend(%Axon.ModelState{data: data, state: st} = ms) do
    %{ms | data: to_binary_backend(data), state: to_binary_backend(st)}
  end

  defp to_binary_backend(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
  end

  defp to_binary_backend(other), do: other

  defp load_checkpoints_from_dir(directory, _state) do
    pattern = Path.join(directory, "*.bin")

    checkpoints =
      Path.wildcard(pattern)
      |> Enum.sort_by(
        fn path ->
          case File.stat(path) do
            {:ok, stat} -> stat.mtime
            _ -> {{0, 0, 0}, {0, 0, 0}}
          end
        end,
        :desc
      )

    loaded =
      Enum.flat_map(checkpoints, fn path ->
        case File.read(path) do
          {:ok, binary} ->
            case :erlang.binary_to_term(binary) do
              %{model: model, params: params} ->
                version = Path.basename(path, ".bin")

                timestamp =
                  case File.stat(path) do
                    {:ok, stat} ->
                      {{y, mo, d}, {h, mi, s}} = stat.mtime

                      NaiveDateTime.new!(y, mo, d, h, mi, s)
                      |> DateTime.from_naive!("Etc/UTC")
                      |> DateTime.to_unix()

                    _ ->
                      System.system_time(:second)
                  end

                [{version, model, params, timestamp}]

              %{params: _params} ->
                # No model saved, skip
                Logger.warning("Checkpoint #{path} missing model, skipping")
                []

              _ ->
                Logger.warning("Invalid checkpoint format: #{path}")
                []
            end

          {:error, reason} ->
            Logger.warning("Failed to read #{path}: #{inspect(reason)}")
            []
        end
      end)

    {:ok, loaded}
  end

  defp init_stats do
    %{
      total_samples: 0,
      samples_by_type: %{current: 0, historical: 0, cpu: 0, uniform: 0},
      total_snapshots: 0
    }
  end

  defp update_stats(stats, :snapshot) do
    %{stats | total_snapshots: stats.total_snapshots + 1}
  end

  defp update_stats(stats, {:sample, policy_id}) do
    type =
      case policy_id do
        :current -> :current
        {:historical, _} -> :historical
        {:cpu, _} -> :cpu
        _ -> :uniform
      end

    samples_by_type = Map.update(stats.samples_by_type, type, 1, &(&1 + 1))

    %{stats | total_samples: stats.total_samples + 1, samples_by_type: samples_by_type}
  end
end
