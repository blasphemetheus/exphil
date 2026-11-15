defmodule ExPhil.Stomped.Server do
  @moduledoc """
  The web server a stomped.dev bot is: Bandit serving `ExPhil.Stomped.Router`.

  It is not part of the application's supervision tree — a training run must
  never open a port. Start it from `scripts/stomped_serve.exs` or add it to a
  supervisor:

      {ExPhil.Stomped.Server,
       port: 3000,
       path: "/7f3a9c2e-unguessable",
       bot: {ExPhil.Stomped.Bots.Walker, character: 2}}

  Options:

    * `:bot` — `{module, opts}` implementing `ExPhil.Stomped.Bot` (required)
    * `:path` — the secret path of the registered URL (required). The URL is
      the bot's password: anyone who knows it can open a match socket unless
      handshake verification is on.
    * `:port` — TCP port (default 3000; 0 picks a free one)
    * `:ip` — bind address (default loopback, which is what a tunnel needs)
    * `:lead_extra` — extra frames to aim ahead (see `ExPhil.Stomped.Match`)
    * `:hold_ahead` — frames each pad is repeated ahead as cover for a late
      decision (see `ExPhil.Stomped.Match`)
    * `:timing_log` — file that gets one JSON line per closed match with its
      timing summary (see `ExPhil.Stomped.Match.summary/1`)
    * `:url` and `:public_key` — the registered URL and the platform key;
      setting both turns on handshake verification
  """

  alias ExPhil.Stomped.Router

  @default_port 3000

  @spec child_spec(keyword()) :: Supervisor.child_spec()
  def child_spec(opts) do
    %{
      id: Keyword.get(opts, :id, __MODULE__),
      start: {__MODULE__, :start_link, [opts]},
      type: :supervisor
    }
  end

  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    router_opts = %{
      path: Keyword.fetch!(opts, :path),
      url: Keyword.get(opts, :url),
      public_key: Keyword.get(opts, :public_key),
      match_opts: [
        bot: Keyword.fetch!(opts, :bot),
        lead_extra: Keyword.get(opts, :lead_extra, 0),
        hold_ahead: Keyword.get(opts, :hold_ahead, 0),
        timing_log: Keyword.get(opts, :timing_log)
      ]
    }

    Bandit.start_link(
      plug: {Router, router_opts},
      scheme: :http,
      port: Keyword.get(opts, :port, @default_port),
      ip: Keyword.get(opts, :ip, :loopback),
      startup_log: false
    )
  end

  @doc "The port a running server is bound to (useful with `port: 0`)."
  @spec port(Supervisor.supervisor()) :: :inet.port_number()
  def port(server) do
    {:ok, {_ip, port}} = ThousandIsland.listener_info(server)
    port
  end
end
