defmodule ExPhil.Agents.Supervisor do
  @moduledoc """
  Dynamic supervisor for Agent processes.

  Manages policy inference agents that can be started and stopped
  dynamically. Each agent holds a trained policy and performs inference
  for game states.

  ## Usage

      # Start a new agent
      {:ok, pid} = ExPhil.Agents.Supervisor.start_agent(
        name: :mewtwo_agent,
        policy_path: "checkpoints/mewtwo.axon"
      )

      # Get agent by name
      {:ok, pid} = ExPhil.Agents.Supervisor.get_agent(:mewtwo_agent)

      # Stop an agent
      :ok = ExPhil.Agents.Supervisor.stop_agent(:mewtwo_agent)

      # List all agents
      agents = ExPhil.Agents.Supervisor.list_agents()

  """

  use DynamicSupervisor

  alias ExPhil.Agents.Agent

  @registry ExPhil.Registry

  # ============================================================================
  # Supervisor API
  # ============================================================================

  @doc """
  Starts the Agent supervisor.
  """
  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  # ============================================================================
  # Agent Management
  # ============================================================================

  @doc """
  Start a new supervised Agent.

  ## Options
    - `:name` - Atom name for lookup (required)
    - `:policy_path` - Path to policy file
    - `:policy` - Pre-loaded policy map
    - `:frame_delay` - Frame delay (default: 0)
    - `:deterministic` - Use deterministic actions (default: false)
    - `:temperature` - Sampling temperature (default: 1.0)

  ## Returns
    `{:ok, pid}` on success, `{:error, reason}` on failure.
  """
  @spec start_agent(keyword()) :: {:ok, pid()} | {:error, term()}
  def start_agent(opts) do
    DynamicSupervisor.start_child(__MODULE__, {Agent, opts})
  end

  @doc """
  Get an agent by its registered name.
  """
  @spec get_agent(atom()) :: {:ok, pid()} | {:error, :not_found}
  def get_agent(name) do
    case Registry.lookup(@registry, {:agent, name}) do
      [{pid, _}] -> {:ok, pid}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  Stop an agent by name or pid.
  """
  @spec stop_agent(atom() | pid()) :: :ok | {:error, term()}
  def stop_agent(name) when is_atom(name) do
    case get_agent(name) do
      {:ok, pid} -> stop_agent(pid)
      error -> error
    end
  end

  def stop_agent(pid) when is_pid(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end

  @doc """
  List all active agents.

  Returns a list of `{name, pid}` tuples.
  """
  @spec list_agents() :: [{atom() | nil, pid()}]
  def list_agents do
    DynamicSupervisor.which_children(__MODULE__)
    |> Enum.map(fn {_, pid, _, _} ->
      name = find_agent_name(pid)
      {name, pid}
    end)
  end

  @doc """
  Count active agents.
  """
  @spec count_agents() :: non_neg_integer()
  def count_agents do
    DynamicSupervisor.count_children(__MODULE__).active
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp find_agent_name(pid) do
    @registry
    |> Registry.keys(pid)
    |> Enum.find_value(fn
      {:agent, name} -> name
      _ -> nil
    end)
  end
end
