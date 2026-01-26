defmodule ExPhil.Agents do
  @moduledoc """
  High-level interface for managing inference agents.

  This module provides convenient functions for starting agents,
  loading policies, and running inference.

  ## Example

      # Start an agent with a trained policy
      {:ok, agent} = ExPhil.Agents.start(:mewtwo, "checkpoints/mewtwo.axon")

      # Get action for a game state
      {:ok, action} = ExPhil.Agents.get_action(:mewtwo, game_state)

      # Get controller input directly
      {:ok, controller} = ExPhil.Agents.get_controller(:mewtwo, game_state)

      # Run agent with a bridge
      ExPhil.Agents.run(:mewtwo, bridge_pid)

      # Stop agent
      :ok = ExPhil.Agents.stop(:mewtwo)

  """

  alias ExPhil.Agents.{Agent, Supervisor}
  alias ExPhil.Bridge.{GameState, ControllerState}

  require Logger

  # ============================================================================
  # Agent Lifecycle
  # ============================================================================

  @doc """
  Start a new agent with a trained policy.

  ## Arguments
    - `name` - Atom name for the agent
    - `policy_path` - Path to the exported policy file

  ## Options
    - `:frame_delay` - Frame delay for online play (default: 0)
    - `:deterministic` - Use deterministic actions (default: false)
    - `:temperature` - Sampling temperature (default: 1.0)

  ## Returns
    `{:ok, pid}` on success, `{:error, reason}` on failure.
  """
  @spec start(atom(), Path.t(), keyword()) :: {:ok, pid()} | {:error, term()}
  def start(name, policy_path, opts \\ []) do
    Supervisor.start_agent([
      {:name, name},
      {:policy_path, policy_path}
      | opts
    ])
  end

  @doc """
  Start an agent without a policy (load later).
  """
  @spec start_empty(atom(), keyword()) :: {:ok, pid()} | {:error, term()}
  def start_empty(name, opts \\ []) do
    Supervisor.start_agent([{:name, name} | opts])
  end

  @doc """
  Stop an agent by name.
  """
  @spec stop(atom()) :: :ok | {:error, term()}
  def stop(name) do
    Supervisor.stop_agent(name)
  end

  @doc """
  Get an agent process by name.
  """
  @spec get(atom()) :: {:ok, pid()} | {:error, :not_found}
  def get(name) do
    Supervisor.get_agent(name)
  end

  @doc """
  List all active agents.
  """
  @spec list() :: [{atom() | nil, pid()}]
  def list do
    Supervisor.list_agents()
  end

  # ============================================================================
  # Inference
  # ============================================================================

  @doc """
  Get an action for a game state.

  ## Arguments
    - `name_or_pid` - Agent name or pid
    - `game_state` - Current game state

  ## Options
    - `:player_port` - Player port for embedding (default: 1)
    - `:deterministic` - Override agent's deterministic setting
    - `:temperature` - Override agent's temperature

  ## Returns
    `{:ok, action}` with action map, or `{:error, reason}`.
  """
  @spec get_action(atom() | pid(), GameState.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def get_action(name_or_pid, game_state, opts \\ [])

  def get_action(name, game_state, opts) when is_atom(name) do
    case get(name) do
      {:ok, pid} -> Agent.get_action(pid, game_state, opts)
      error -> error
    end
  end

  def get_action(pid, game_state, opts) when is_pid(pid) do
    Agent.get_action(pid, game_state, opts)
  end

  @doc """
  Get a controller state for a game state.

  Returns a ControllerState struct ready to send to the bridge.
  """
  @spec get_controller(atom() | pid(), GameState.t(), keyword()) ::
          {:ok, ControllerState.t()} | {:error, term()}
  def get_controller(name_or_pid, game_state, opts \\ [])

  def get_controller(name, game_state, opts) when is_atom(name) do
    case get(name) do
      {:ok, pid} -> Agent.get_controller(pid, game_state, opts)
      error -> error
    end
  end

  def get_controller(pid, game_state, opts) when is_pid(pid) do
    Agent.get_controller(pid, game_state, opts)
  end

  @doc """
  Load a new policy into an agent.
  """
  @spec load_policy(atom() | pid(), Path.t()) :: :ok | {:error, term()}
  def load_policy(name, policy_path) when is_atom(name) do
    case get(name) do
      {:ok, pid} -> Agent.load_policy(pid, policy_path)
      error -> error
    end
  end

  def load_policy(pid, policy_path) when is_pid(pid) do
    Agent.load_policy(pid, policy_path)
  end

  # ============================================================================
  # Running with Bridge
  # ============================================================================

  @doc """
  Run an agent with a bridge, using the agent for action decisions.

  ## Arguments
    - `agent_name` - Name of the agent
    - `bridge` - Bridge pid or name

  ## Options
    - `:max_frames` - Maximum frames to run (default: infinity)
    - `:player_port` - Player port for the agent (default: 1)

  ## Returns
    `{:ok, reason}` when game ends, or `{:error, reason}` on failure.
  """
  @spec run(atom(), pid() | atom(), keyword()) :: {:ok, term()} | {:error, term()}
  def run(agent_name, bridge, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)

    case get(agent_name) do
      {:ok, agent_pid} ->
        ExPhil.Bridge.run_game(
          bridge,
          fn game_state ->
            case Agent.get_controller(agent_pid, game_state, player_port: player_port) do
              {:ok, input} ->
                {:continue, controller_to_input(input)}

              {:error, reason} ->
                Logger.warning("[Agents] Failed to get action: #{inspect(reason)}")
                {:continue, ExPhil.Bridge.ControllerInput.neutral()}
            end
          end,
          opts
        )

      error ->
        error
    end
  end

  # Convert ControllerState to bridge input format
  defp controller_to_input(%ControllerState{} = cs) do
    %{
      main_stick: cs.main_stick,
      c_stick: cs.c_stick,
      shoulder: cs.l_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end
end
