defmodule ExPhil.Application do
  @moduledoc """
  ExPhil OTP Application.

  Supervises the core components:
  - Registry for named process lookup
  - Bridge supervisor for Dolphin/game instances
  - Agent supervisor for policy inference processes
  - Telemetry metrics collector

  ## Supervision Tree

  ```
  ExPhil.Supervisor (one_for_one)
  ├── ExPhil.Registry (process registry)
  ├── ExPhil.Bridge.Supervisor (dynamic, game instances)
  ├── ExPhil.Agents.Supervisor (dynamic, inference agents)
  └── ExPhil.Telemetry.Collector (metrics)
  ```

  ## Usage

  The application starts automatically. To start agents or bridges:

      # Start a bridge for a game instance
      {:ok, bridge} = ExPhil.Bridge.Supervisor.start_bridge(
        name: :game1,
        dolphin_path: "/path/to/slippi",
        iso_path: "/path/to/melee.iso"
      )

      # Start an agent with a trained policy
      {:ok, agent} = ExPhil.Agents.Supervisor.start_agent(
        name: :mewtwo,
        policy_path: "checkpoints/mewtwo.axon"
      )

  """

  use Application

  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      # Registry for named process lookup (bridges and agents)
      {Registry, keys: :unique, name: ExPhil.Registry},

      # Bridge supervisor for game instances
      {ExPhil.Bridge.Supervisor, []},

      # Agent supervisor for inference
      {ExPhil.Agents.Supervisor, []},

      # Telemetry metrics collector (optional, can be disabled)
      {ExPhil.Telemetry.Collector, []},

      # Async checkpoint saver (non-blocking checkpoint saves)
      {ExPhil.Training.AsyncCheckpoint, []}
    ]

    opts = [strategy: :one_for_one, name: ExPhil.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} = result ->
        Logger.info("[ExPhil] Application started")
        Logger.debug("[ExPhil] Supervision tree: #{inspect(Supervisor.which_children(pid))}")
        result

      error ->
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("[ExPhil] Application stopping")
    :ok
  end
end
