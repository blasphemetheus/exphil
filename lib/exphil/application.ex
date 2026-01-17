defmodule ExPhil.Application do
  @moduledoc """
  ExPhil OTP Application.

  Supervises the core components:
  - Python bridge for libmelee communication
  - Agent processes for inference
  - Training coordinator (when training)
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Python bridge supervisor (start when needed)
      # {ExPhil.Bridge.Supervisor, []},

      # Telemetry metrics
      # {ExPhil.Telemetry, []}
    ]

    opts = [strategy: :one_for_one, name: ExPhil.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
