defmodule ExPhil.Bridge.SimPort do
  @moduledoc """
  Elixir side of the melee-sim-light worker (`priv/python/sim_worker.py`,
  SIM_INTEGRATION.md step 2).

  A GenServer owning one Erlang Port to the Python worker. Requests are
  synchronous line-delimited JSON (same shape as the libmelee bridge); every
  frame comes back as the sim's gamestate row and is mapped through
  `ExPhil.Bridge.SimState` into a `GameState`, so callers see exactly what
  the Dolphin bridge would hand them.

  ## Environment (all optional; defaults = the sim `main` clone)

    * `EXPHIL_SIM_PYTHON` — interpreter with numpy (default:
      `~/git/melee-sim-light/.venv/bin/python`).
    * `EXPHIL_SIM_ROOT` — sim checkout used for `PYTHONPATH`,
      `MSL_CORE_LIBRARY` (`build/melee_core/python/libmelee_core.so`) and
      `MSL_DATA_DIR` (`data/`) (default: `~/git/msl-main`).

  ## Example

      {:ok, sim} = SimPort.start_link(stage: "final_destination",
                                      players: [%{character: "fox"}, %{character: "fox", costume: 1}])
      {:ok, [gs]} = SimPort.frames(sim)
      {:ok, [gs], [term]} = SimPort.step(sim, [[controller_state, nil]])
  """

  use GenServer
  require Logger

  alias ExPhil.Bridge.{ControllerState, SimRows, SimState}

  @default_timeout 30_000

  # ---------------------------------------------------------------------------
  # Client
  # ---------------------------------------------------------------------------

  @doc """
  Start a worker and initialize a match.

  Options: `:stage` (name or external id), `:players` (list of maps with
  `:character` and optional `:costume`/`:start_percent`/`:facing`/`:team`),
  `:batch_size` (default 1), `:length` (buffer window, default 256), `:seed`,
  `:stocks`, `:max_frame`, `:own_port` (default 1), `:name`.
  """
  def start_link(opts \\ []) do
    {name_opts, opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, opts, name_opts)
  end

  @doc "Current frames (one `GameState` per env)."
  def frames(server), do: GenServer.call(server, :frames, @default_timeout)

  @doc """
  Advance one frame. `controllers` is a list (per env) of lists (per player)
  of `ControllerState` structs, sim controller rows, or `nil` (neutral).
  Returns `{:ok, [GameState], [terminal_row]}`.
  """
  def step(server, controllers \\ nil), do: GenServer.call(server, {:step, controllers}, @default_timeout)

  @doc "Reset all envs (or `env_ids`)."
  def reset(server, env_ids \\ nil), do: GenServer.call(server, {:reset, env_ids}, @default_timeout)

  @doc """
  Serialize env `i`'s full state. Returns `{:ok, binary}`; with `keep: true`
  returns `{:ok, binary, state_id}` and the worker also keeps the blob so
  `restore/3` with `{:id, state_id}` skips the ~1 MB transfer (a search
  restores the same start 64 times).
  """
  def save(server, env \\ 0, opts \\ []), do: GenServer.call(server, {:save, env, opts}, @default_timeout)

  @doc "Cache a blob in the worker; returns `{:ok, state_id}`."
  def upload(server, blob) when is_binary(blob), do: GenServer.call(server, {:upload, blob}, @default_timeout)

  @doc "Drop cached states (`nil` = all)."
  def forget(server, state_ids \\ nil), do: GenServer.call(server, {:forget, state_ids}, @default_timeout)

  @doc "Restore env `i` from a `save` binary or a cached `{:id, state_id}`."
  def restore(server, env, state), do: GenServer.call(server, {:restore, env, state}, @default_timeout)

  @doc "Re-initialize the match (same keys as the init request: stage, players, batch_size, length, seed...). Updates the cached frames and batch size."
  def reinit(server, req) when is_map(req), do: GenServer.call(server, {:reinit, req}, @default_timeout)

  @doc "Raw request passthrough (`%{cmd: ...}`), for probes."
  def request(server, req), do: GenServer.call(server, {:raw, req}, @default_timeout)

  def stop(server), do: GenServer.stop(server, :normal)

  # ---------------------------------------------------------------------------
  # Server
  # ---------------------------------------------------------------------------

  @impl true
  def init(opts) do
    python = System.get_env("EXPHIL_SIM_PYTHON") || Path.expand("~/git/melee-sim-light/.venv/bin/python")
    root = System.get_env("EXPHIL_SIM_ROOT") || Path.expand("~/git/msl-main")
    script = Path.join(:code.priv_dir(:exphil) |> to_string(), "python/sim_worker.py")

    cond do
      not File.exists?(python) -> {:stop, {:sim_python_missing, python}}
      not File.exists?(Path.join(root, "build/melee_core/python/libmelee_core.so")) -> {:stop, {:sim_library_missing, root}}
      true ->
        port =
          Port.open({:spawn_executable, python}, [
            :binary,
            :exit_status,
            # 4-byte length-prefixed frames both ways; JSON control messages
            # and binary step frames share the channel (SimRows).
            {:packet, 4},
            {:args, [script]},
            {:cd, root},
            {:env,
             [
               {~c"PYTHONPATH", String.to_charlist(root)},
               {~c"MSL_CORE_LIBRARY", String.to_charlist(Path.join(root, "build/melee_core/python/libmelee_core.so"))},
               {~c"MSL_DATA_DIR", String.to_charlist(Path.join(root, "data"))},
               {~c"PYTHONUNBUFFERED", ~c"1"}
             ]}
          ])

        state = %{
          port: port,
          own_port: Keyword.get(opts, :own_port, 1),
          frames: [],
          batch_size: 1,
          # dtype layouts from the worker (init/ping); binary steps need them
          layout: nil,
          # :binary false forces the JSON step path (A/B and fallback)
          binary: Keyword.get(opts, :binary, true)
        }

        init_req =
          %{cmd: "init"}
          |> put_opt(opts, :stage, "final_destination")
          |> put_opt(opts, :players, [%{character: "fox"}, %{character: "fox", costume: 1}])
          |> put_opt(opts, :batch_size, 1)
          |> put_opt(opts, :length, 256)
          |> put_opt(opts, :seed)
          |> put_opt(opts, :stocks)
          |> put_opt(opts, :max_frame)

        case send_request(port, init_req) do
          {:ok, %{"frames" => rows} = resp} ->
            {:ok, %{state | frames: map_rows(rows, state.own_port), batch_size: resp["batch_size"] || 1, layout: resp["layout"]}}

          {:error, reason} ->
            Port.close(port)
            {:stop, {:sim_init_failed, reason}}
        end
    end
  end

  @impl true
  def handle_call(:frames, _from, state), do: {:reply, {:ok, state.frames}, state}

  def handle_call({:step, controllers}, _from, %{binary: true, layout: %{} = layout} = state) do
    # Binary step: <<1, controller rows>> -> <<1, gamestate rows, terminal rows>>.
    body =
      if controllers do
        cl = layout["controller_input"]
        rows = encode_controllers(controllers)
        for per_env <- rows, into: <<>>, do: SimRows.encode(cl, %{"players" => per_env})
      else
        <<>>
      end

    Port.command(state.port, <<1>> <> body)

    case receive_packet(state.port) do
      {:ok, {:binary, rest}} ->
        n = state.batch_size
        gl = layout["gamestate"]
        tl = layout["terminal"]
        gsize = SimRows.itemsize(gl) * n
        <<g::binary-size(gsize), t::binary>> = rest
        frames = gl |> SimRows.decode_rows(g, n) |> map_rows(state.own_port)
        terminal = SimRows.decode_rows(tl, t, n)
        {:reply, {:ok, frames, terminal}, %{state | frames: frames}}

      {:ok, other} ->
        {:reply, {:error, {:sim_bad_response, other}}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:step, controllers}, _from, state) do
    req = %{cmd: "step"}
    req = if controllers, do: Map.put(req, :controllers, encode_controllers(controllers)), else: req

    case send_request(state.port, req) do
      {:ok, %{"frames" => rows, "terminal" => terminal}} ->
        frames = map_rows(rows, state.own_port)
        {:reply, {:ok, frames, terminal}, %{state | frames: frames}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:reset, env_ids}, _from, state) do
    req = if env_ids, do: %{cmd: "reset", env_ids: env_ids}, else: %{cmd: "reset"}
    reply_frames(send_request(state.port, req), state)
  end

  def handle_call({:save, env, opts}, _from, state) do
    keep = Keyword.get(opts, :keep, false)

    case send_request(state.port, %{cmd: "save", env: env, keep: keep}) do
      {:ok, %{"state" => b64, "state_id" => id}} when keep -> {:reply, {:ok, Base.decode64!(b64), id}, state}
      {:ok, %{"state" => b64}} -> {:reply, {:ok, Base.decode64!(b64)}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:upload, blob}, _from, state) do
    case send_request(state.port, %{cmd: "upload", state: Base.encode64(blob)}) do
      {:ok, %{"state_id" => id}} -> {:reply, {:ok, id}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:forget, ids}, _from, state) do
    req = if ids, do: %{cmd: "forget", state_ids: ids}, else: %{cmd: "forget"}
    {:reply, send_request(state.port, req), state}
  end

  def handle_call({:restore, env, {:id, id}}, _from, state) do
    reply_frames(send_request(state.port, %{cmd: "restore", env: env, state_id: id}), state)
  end

  def handle_call({:restore, env, bin}, _from, state) when is_binary(bin) do
    reply_frames(send_request(state.port, %{cmd: "restore", env: env, state: Base.encode64(bin)}), state)
  end

  def handle_call({:reinit, req}, _from, state) do
    case send_request(state.port, Map.put(req, :cmd, "init")) do
      {:ok, %{"frames" => rows} = resp} ->
        frames = map_rows(rows, state.own_port)
        {:reply, {:ok, frames}, %{state | frames: frames, batch_size: resp["batch_size"] || length(rows), layout: resp["layout"] || state.layout}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:raw, req}, _from, state), do: {:reply, send_request(state.port, req), state}

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[SimPort] worker exited with status #{status}")
    {:stop, {:sim_worker_exited, status}, state}
  end

  def handle_info({port, {:data, line}}, %{port: port} = state) do
    Logger.warning("[SimPort] unsolicited line: #{String.slice(line, 0, 200)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, %{port: port}) do
    try do
      # Wait for the stop ack so the worker never writes into a closed pipe.
      _ = send_request(port, %{cmd: "stop"})
      Port.close(port)
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end

    :ok
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp reply_frames({:ok, %{"frames" => rows}}, state) do
    frames = map_rows(rows, state.own_port)
    {:reply, {:ok, frames}, %{state | frames: frames}}
  end

  defp reply_frames({:error, reason}, state), do: {:reply, {:error, reason}, state}

  defp map_rows(rows, own_port), do: Enum.map(rows, &SimState.to_game_state(&1, own_port: own_port))

  defp encode_controllers(controllers) do
    for per_env <- controllers do
      for c <- per_env do
        case c do
          nil -> nil
          %ControllerState{} = cs -> SimState.controller_to_row(cs)
          %{} = row -> row
        end
      end
    end
  end

  defp put_opt(req, opts, key, default \\ nil) do
    case Keyword.get(opts, key, default) do
      nil -> req
      v -> Map.put(req, key, v)
    end
  end

  # One request, one response frame. The worker answers strictly in order,
  # so the first frame after a request is its reply.
  defp send_request(port, req) do
    Port.command(port, Jason.encode!(req))

    case receive_packet(port) do
      {:ok, {:binary, _}} -> {:error, :sim_unexpected_binary}
      other -> other
    end
  end

  defp receive_packet(port) do
    receive do
      {^port, {:data, <<1, rest::binary>>}} ->
        {:ok, {:binary, rest}}

      {^port, {:data, payload}} ->
        case Jason.decode(payload) do
          {:ok, %{"ok" => true} = resp} -> {:ok, resp}
          {:ok, %{"ok" => false, "error" => err}} -> {:error, {:sim_error, err}}
          {:ok, other} -> {:error, {:sim_bad_response, other}}
          {:error, _} -> {:error, {:sim_bad_json, String.slice(payload, 0, 200)}}
        end

      {^port, {:exit_status, status}} ->
        {:error, {:sim_worker_exited, status}}
    after
      @default_timeout -> {:error, :sim_timeout}
    end
  end
end
