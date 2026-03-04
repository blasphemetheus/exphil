defmodule ExPhil.Bridge.MojoPort do
  @moduledoc """
  Elixir Port interface to Mojo for GPU linear scan operations.

  Mojo aims for Python-level ergonomics with C-level performance.
  This port wraps the Mojo kernel via a Python server (Mojo Python interop).

  Falls back to NumPy if Mojo SDK is not installed — useful for testing
  the integration protocol without Mojo.

  ## Usage

      {:ok, pid} = ExPhil.Bridge.MojoPort.start_link()
      ExPhil.Bridge.MojoPort.available?()
      {:ok, result} = ExPhil.Bridge.MojoPort.linear_scan(a, b, h0)

  ## Architecture

      ┌─────────────┐     msgpack      ┌──────────────┐
      │   Elixir    │◄───────────────►│   Python     │
      │  GenServer  │   Port (stdio)   │  Mojo/NumPy  │
      └─────────────┘                  └──────────────┘
  """

  use GenServer
  require Logger

  alias ExPhil.Error.BridgeError

  @timeout 30_000

  # ==========================================================================
  # Client API
  # ==========================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @spec available?() :: boolean()
  def available? do
    case ping() do
      {:ok, _} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @spec ping() :: {:ok, map()} | {:error, BridgeError.t()}
  def ping do
    GenServer.call(__MODULE__, :ping, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :mojo_port)}
  end

  @spec info() :: {:ok, map()} | {:error, BridgeError.t()}
  def info do
    GenServer.call(__MODULE__, :info, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :mojo_port)}
  end

  @doc """
  Perform fused linear scan: h = a * h + b over timesteps.

  ## Arguments

  * `a` - Decay coefficients `[batch, seq_len, hidden]`, f32
  * `b` - Additive terms `[batch, seq_len, hidden]`, f32
  * `h0` - Initial hidden state `[batch, hidden]`, f32

  ## Options

  * `:mode` - `"mojo"` (Mojo kernel), `"numpy_vec"` (NumPy vectorized), `"auto"` (default)
  """
  @spec linear_scan(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, BridgeError.t()}
  def linear_scan(a, b, h0, opts \\ []) do
    GenServer.call(__MODULE__, {:scan, a, b, h0, opts}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :mojo_port, context: %{operation: :linear_scan})}
  end

  @spec linear_scan!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def linear_scan!(a, b, h0, opts \\ []) do
    case linear_scan(a, b, h0, opts) do
      {:ok, result} -> result
      {:error, reason} -> raise "Mojo scan failed: #{inspect(reason)}"
    end
  end

  @doc """
  Run benchmark on server side (avoids serialization overhead in timing).
  """
  @spec benchmark(pos_integer(), pos_integer(), pos_integer(), keyword()) ::
          {:ok, map()} | {:error, BridgeError.t()}
  def benchmark(batch, seq_len, hidden, opts \\ []) do
    GenServer.call(__MODULE__, {:benchmark, batch, seq_len, hidden, opts}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :mojo_port)}
  end

  # ==========================================================================
  # GenServer Implementation
  # ==========================================================================

  @impl true
  def init(_opts) do
    server_script = Path.join([File.cwd!(), "native", "mojo_scan", "server.py"])

    python_exe = System.find_executable("python3")

    if python_exe == nil do
      Logger.error("[MojoPort] python3 not found in PATH")
      {:stop, :python_not_found}
    else
      port =
        Port.open({:spawn_executable, python_exe}, [
          :binary,
          :exit_status,
          {:packet, 4},
          {:args, [server_script]},
          {:cd, Path.dirname(server_script)}
        ])

      Process.sleep(500)
      Logger.info("[MojoPort] Started Mojo/NumPy scan server")
      {:ok, %{port: port}}
    end
  end

  @impl true
  def handle_call(:ping, _from, state) do
    response = send_request(state.port, %{op: "ping"})
    {:reply, response, state}
  end

  @impl true
  def handle_call(:info, _from, state) do
    response = send_request(state.port, %{op: "info"})
    {:reply, response, state}
  end

  @impl true
  def handle_call({:scan, a, b, h0, opts}, _from, state) do
    {batch, seq_len, hidden} = Nx.shape(a)
    mode = Keyword.get(opts, :mode, "auto")

    request = %{
      op: "scan",
      batch: batch,
      seq_len: seq_len,
      hidden: hidden,
      mode: mode,
      a: tensor_to_binary(a),
      b: tensor_to_binary(b),
      h0: tensor_to_binary(h0)
    }

    case send_request(state.port, request) do
      {:ok, %{"status" => "ok", "result" => result_bin, "shape" => shape}} ->
        result =
          result_bin
          |> Nx.from_binary(:f32)
          |> Nx.reshape(List.to_tuple(shape))

        {:reply, {:ok, result}, state}

      {:ok, %{"status" => "error", "message" => msg}} ->
        {:reply, {:error, BridgeError.new(:protocol_error, bridge: :mojo_port, context: %{message: msg})}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:benchmark, batch, seq_len, hidden, opts}, _from, state) do
    warmup = Keyword.get(opts, :warmup, 5)
    iterations = Keyword.get(opts, :iterations, 30)

    request = %{
      op: "benchmark",
      batch: batch,
      seq_len: seq_len,
      hidden: hidden,
      warmup: warmup,
      iterations: iterations
    }

    case send_request(state.port, request) do
      {:ok, %{"status" => "ok"} = result} ->
        {:reply, {:ok, result}, state}

      {:ok, %{"status" => "error", "message" => msg}} ->
        {:reply, {:error, msg}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_info({port, {:data, _data}}, %{port: port} = state) do
    Logger.warning("[MojoPort] Unexpected data from port")
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[MojoPort] Python process exited with status #{status}")
    {:stop, {:python_exit, status}, state}
  end

  @impl true
  def terminate(_reason, %{port: port}) do
    Port.close(port)
    :ok
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp tensor_to_binary(tensor) do
    tensor
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> Nx.as_type(:f32)
    |> Nx.to_binary()
  end

  defp send_request(port, request) do
    encoded = Msgpax.pack!(request, iodata: false)
    Port.command(port, encoded)

    receive do
      {^port, {:data, data}} ->
        case Msgpax.unpack(data) do
          {:ok, response} -> {:ok, response}
          {:error, reason} -> {:error, BridgeError.new(:protocol_error, bridge: :mojo_port, context: %{details: "decode error: #{inspect(reason)}"})}
        end
    after
      @timeout -> {:error, BridgeError.new(:timeout, bridge: :mojo_port, context: %{timeout_ms: @timeout})}
    end
  end
end
