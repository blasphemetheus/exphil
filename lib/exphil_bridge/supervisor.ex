defmodule ExPhil.Bridge.Supervisor do
  @moduledoc """
  Dynamic supervisor for Melee bridge processes.

  Each MeleePort process manages communication with one Dolphin instance.
  This supervisor allows starting multiple game instances for parallel
  self-play or evaluation.

  ## Usage

      # Start a new bridge (supervised)
      {:ok, pid} = ExPhil.Bridge.Supervisor.start_bridge(
        name: :game1,
        dolphin_path: "/path/to/slippi",
        iso_path: "/path/to/melee.iso"
      )

      # Get bridge by name
      {:ok, pid} = ExPhil.Bridge.Supervisor.get_bridge(:game1)

      # Stop a bridge
      :ok = ExPhil.Bridge.Supervisor.stop_bridge(:game1)

      # List all active bridges
      bridges = ExPhil.Bridge.Supervisor.list_bridges()

  """

  use DynamicSupervisor

  alias ExPhil.Bridge.MeleePort

  @registry ExPhil.Registry

  # ============================================================================
  # Supervisor API
  # ============================================================================

  @doc """
  Starts the Bridge supervisor.
  """
  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  # ============================================================================
  # Bridge Management
  # ============================================================================

  @doc """
  Start a new supervised MeleePort bridge.

  ## Options
    - `:name` - Atom name for the bridge (required for lookup)
    - `:dolphin_path` - Path to Slippi/Dolphin (required for init)
    - `:iso_path` - Path to Melee ISO (required for init)
    - `:auto_init` - Whether to initialize console on start (default: false)
    - All other options passed to `MeleePort.start_link/1`

  ## Returns
    `{:ok, pid}` on success, `{:error, reason}` on failure.
  """
  @spec start_bridge(keyword()) :: {:ok, pid()} | {:error, term()}
  def start_bridge(opts) do
    name = Keyword.get(opts, :name)
    auto_init = Keyword.get(opts, :auto_init, false)

    # If name provided, register with Registry
    child_opts =
      if name do
        Keyword.put(opts, :name, {:via, Registry, {@registry, {:bridge, name}}})
      else
        opts
      end

    case DynamicSupervisor.start_child(__MODULE__, {MeleePort, child_opts}) do
      {:ok, pid} = result ->
        # Auto-initialize if requested
        if auto_init do
          config = %{
            dolphin_path: Keyword.fetch!(opts, :dolphin_path),
            iso_path: Keyword.fetch!(opts, :iso_path),
            character: Keyword.get(opts, :character, :fox),
            stage: Keyword.get(opts, :stage, :final_destination),
            controller_port: Keyword.get(opts, :controller_port, 1),
            opponent_port: Keyword.get(opts, :opponent_port, 2)
          }

          case MeleePort.init_console(pid, config) do
            {:ok, _} -> result
            {:error, _} = err -> err
          end
        else
          result
        end

      error ->
        error
    end
  end

  @doc """
  Get a bridge by its registered name.
  """
  @spec get_bridge(atom()) :: {:ok, pid()} | {:error, :not_found}
  def get_bridge(name) do
    case Registry.lookup(@registry, {:bridge, name}) do
      [{pid, _}] -> {:ok, pid}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  Stop a bridge by name or pid.
  """
  @spec stop_bridge(atom() | pid()) :: :ok | {:error, term()}
  def stop_bridge(name) when is_atom(name) do
    case get_bridge(name) do
      {:ok, pid} -> stop_bridge(pid)
      error -> error
    end
  end

  def stop_bridge(pid) when is_pid(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end

  @doc """
  List all active bridge processes.

  Returns a list of `{name, pid}` tuples for named bridges,
  or just `{nil, pid}` for unnamed ones.
  """
  @spec list_bridges() :: [{atom() | nil, pid()}]
  def list_bridges do
    DynamicSupervisor.which_children(__MODULE__)
    |> Enum.map(fn {_, pid, _, _} ->
      name = find_bridge_name(pid)
      {name, pid}
    end)
  end

  @doc """
  Count active bridges.
  """
  @spec count_bridges() :: non_neg_integer()
  def count_bridges do
    DynamicSupervisor.count_children(__MODULE__).active
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp find_bridge_name(pid) do
    # Look up name in registry
    @registry
    |> Registry.keys(pid)
    |> Enum.find_value(fn
      {:bridge, name} -> name
      _ -> nil
    end)
  end
end
