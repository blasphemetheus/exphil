defmodule ExPhil.Agents.Agent do
  @moduledoc """
  GenServer that holds a trained policy and performs inference.

  An Agent encapsulates a neural network policy and provides a simple
  interface for getting actions from game states. Agents can be hot-swapped
  with new policies during runtime.

  ## Usage

      # Start an agent with a trained policy
      {:ok, agent} = ExPhil.Agents.Agent.start_link(
        name: :mewtwo_agent,
        policy_path: "checkpoints/mewtwo_policy.axon"
      )

      # Get action for a game state
      {:ok, action} = ExPhil.Agents.Agent.get_action(agent, game_state)

      # Get controller state directly
      {:ok, controller} = ExPhil.Agents.Agent.get_controller(agent, game_state)

      # Hot-swap policy
      :ok = ExPhil.Agents.Agent.load_policy(agent, "checkpoints/new_policy.axon")

  ## Frame Delay Handling

  For online play with frame delay, the agent can maintain a frame buffer:

      {:ok, agent} = ExPhil.Agents.Agent.start_link(
        name: :online_agent,
        policy_path: "checkpoints/policy.axon",
        frame_delay: 4
      )

  """

  use GenServer

  alias ExPhil.{Embeddings, Networks, Training}
  alias ExPhil.Bridge.{GameState, ControllerState}

  require Logger

  @registry ExPhil.Registry

  defstruct [
    :name,
    :policy_params,
    :predict_fn,
    :embed_config,
    :frame_delay,
    :frame_buffer,
    :deterministic,
    :temperature
  ]

  @type t :: %__MODULE__{}

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start an Agent GenServer.

  ## Options
    - `:name` - Atom name for registry lookup
    - `:policy_path` - Path to exported policy file
    - `:policy` - Pre-loaded policy map (alternative to path)
    - `:frame_delay` - Frame delay for online play (default: 0)
    - `:deterministic` - Use deterministic action selection (default: false)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:embed_config` - Embedding configuration (auto-detected from policy)
  """
  def start_link(opts) do
    name = Keyword.get(opts, :name)

    gen_opts = if name do
      [name: {:via, Registry, {@registry, {:agent, name}}}]
    else
      []
    end

    GenServer.start_link(__MODULE__, opts, gen_opts)
  end

  @doc """
  Get the action for a game state.

  Returns a map with button presses and stick positions.
  """
  @spec get_action(GenServer.server(), GameState.t(), keyword()) ::
    {:ok, map()} | {:error, term()}
  def get_action(agent, game_state, opts \\ []) do
    GenServer.call(agent, {:get_action, game_state, opts})
  end

  @doc """
  Get controller state for a game state.

  Returns a ControllerState struct ready to send to the bridge.
  """
  @spec get_controller(GenServer.server(), GameState.t(), keyword()) ::
    {:ok, ControllerState.t()} | {:error, term()}
  def get_controller(agent, game_state, opts \\ []) do
    GenServer.call(agent, {:get_controller, game_state, opts})
  end

  @doc """
  Load a new policy, hot-swapping the current one.
  """
  @spec load_policy(GenServer.server(), Path.t() | map()) :: :ok | {:error, term()}
  def load_policy(agent, policy_or_path) do
    GenServer.call(agent, {:load_policy, policy_or_path})
  end

  @doc """
  Get current agent configuration.
  """
  @spec get_config(GenServer.server()) :: map()
  def get_config(agent) do
    GenServer.call(agent, :get_config)
  end

  @doc """
  Update agent settings.
  """
  @spec configure(GenServer.server(), keyword()) :: :ok
  def configure(agent, opts) do
    GenServer.call(agent, {:configure, opts})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    frame_delay = Keyword.get(opts, :frame_delay, 0)
    deterministic = Keyword.get(opts, :deterministic, false)
    temperature = Keyword.get(opts, :temperature, 1.0)

    state = %__MODULE__{
      name: Keyword.get(opts, :name),
      frame_delay: frame_delay,
      frame_buffer: :queue.new(),
      deterministic: deterministic,
      temperature: temperature
    }

    # Load policy if provided
    state = cond do
      Keyword.has_key?(opts, :policy_path) ->
        case load_policy_internal(state, Keyword.fetch!(opts, :policy_path)) do
          {:ok, new_state} -> new_state
          {:error, reason} ->
            Logger.warning("[Agent] Failed to load policy: #{inspect(reason)}")
            state
        end

      Keyword.has_key?(opts, :policy) ->
        case load_policy_internal(state, Keyword.fetch!(opts, :policy)) do
          {:ok, new_state} -> new_state
          {:error, reason} ->
            Logger.warning("[Agent] Failed to load policy: #{inspect(reason)}")
            state
        end

      true ->
        state
    end

    {:ok, state}
  end

  @impl true
  def handle_call({:get_action, game_state, opts}, _from, state) do
    result = compute_action(state, game_state, opts)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_controller, game_state, opts}, _from, state) do
    case compute_action(state, game_state, opts) do
      {:ok, action} ->
        controller = action_to_controller(action, state)
        {:reply, {:ok, controller}, state}

      error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:load_policy, policy_or_path}, _from, state) do
    case load_policy_internal(state, policy_or_path) do
      {:ok, new_state} ->
        Logger.info("[Agent] Policy loaded successfully")
        {:reply, :ok, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:get_config, _from, state) do
    config = %{
      name: state.name,
      frame_delay: state.frame_delay,
      deterministic: state.deterministic,
      temperature: state.temperature,
      has_policy: state.policy_params != nil
    }
    {:reply, config, state}
  end

  @impl true
  def handle_call({:configure, opts}, _from, state) do
    state = state
    |> maybe_update(:deterministic, opts)
    |> maybe_update(:temperature, opts)
    |> maybe_update(:frame_delay, opts)

    {:reply, :ok, state}
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp compute_action(state, _game_state, _opts) when state.policy_params == nil do
    {:error, :no_policy_loaded}
  end

  defp compute_action(state, game_state, opts) do
    try do
      # Embed game state
      player_port = Keyword.get(opts, :player_port, 1)
      embedded = embed_game_state(game_state, player_port, state.embed_config)

      # Add batch dimension
      embedded_batch = Nx.reshape(embedded, {1, Nx.size(embedded)})

      # Get action from policy
      deterministic = Keyword.get(opts, :deterministic, state.deterministic)
      temperature = Keyword.get(opts, :temperature, state.temperature)

      action = Networks.sample(
        state.policy_params,
        state.predict_fn,
        embedded_batch,
        deterministic: deterministic,
        temperature: temperature
      )

      {:ok, action}
    rescue
      e ->
        Logger.error("[Agent] Error computing action: #{inspect(e)}")
        {:error, e}
    end
  end

  defp embed_game_state(game_state, player_port, embed_config) do
    Embeddings.Game.embed(game_state, player_port, embed_config)
  end

  defp action_to_controller(action, state) do
    Networks.to_controller_state(action, axis_buckets: state.embed_config[:axis_buckets] || 16)
  end

  defp load_policy_internal(state, path) when is_binary(path) do
    case Training.load_policy(path) do
      {:ok, policy} -> load_policy_internal(state, policy)
      error -> error
    end
  end

  defp load_policy_internal(state, %{params: params, config: config} = _policy) do
    # Build prediction function
    embed_config = Map.get(config, :embed_config, %{})
    embed_size = Map.get(config, :embed_size) || Map.get(embed_config, :embed_size, 1991)
    axis_buckets = Map.get(config, :axis_buckets, 16)

    model = Networks.build_policy(embed_size: embed_size, axis_buckets: axis_buckets)
    {_init_fn, predict_fn} = Axon.build(model)

    new_state = %{state |
      policy_params: params,
      predict_fn: predict_fn,
      embed_config: Map.merge(%{embed_size: embed_size, axis_buckets: axis_buckets}, embed_config)
    }

    {:ok, new_state}
  end

  defp load_policy_internal(_state, invalid) do
    {:error, {:invalid_policy, invalid}}
  end

  defp maybe_update(state, key, opts) do
    if Keyword.has_key?(opts, key) do
      Map.put(state, key, Keyword.fetch!(opts, key))
    else
      state
    end
  end
end
