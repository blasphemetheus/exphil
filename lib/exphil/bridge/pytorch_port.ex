defmodule ExPhil.Bridge.PyTorchPort do
  @moduledoc """
  Elixir Port interface to PyTorch for fast GPU operations.

  This module provides a 10x speedup for selective scan operations by using
  PyTorch instead of Nx/XLA. PyTorch achieves ~5ms vs XLA's ~55ms.

  ## Usage

      # Start the server (usually done by application supervisor)
      {:ok, pid} = ExPhil.Bridge.PyTorchPort.start_link()

      # Check if available
      ExPhil.Bridge.PyTorchPort.available?()

      # Run selective scan
      result = ExPhil.Bridge.PyTorchPort.selective_scan(x, dt, a, b, c)

  ## Architecture

      ┌─────────────┐     msgpack      ┌──────────────┐
      │   Elixir    │◄───────────────►│    Python    │
      │  GenServer  │   Port (stdio)   │   PyTorch    │
      └─────────────┘                  └──────────────┘

  ## Performance

      | Implementation    | Time   | 60 FPS? |
      |-------------------|--------|---------|
      | PyTorch (this)    | ~5ms   | YES     |
      | Nx/XLA Blelloch   | ~55ms  | NO      |
  """

  use GenServer
  require Logger

  alias ExPhil.Error.BridgeError

  @timeout 30_000

  # ==========================================================================
  # Client API
  # ==========================================================================

  @doc """
  Start the PyTorch Port server.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if the PyTorch server is available and responding.
  """
  @spec available?() :: boolean()
  def available? do
    case ping() do
      {:ok, _} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @doc """
  Ping the server to check connectivity.
  """
  @spec ping() :: {:ok, map()} | {:error, BridgeError.t()}
  def ping do
    GenServer.call(__MODULE__, :ping, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :pytorch_port)}
  end

  @doc """
  Get server info (device, CUDA availability, etc).
  """
  @spec info() :: {:ok, map()} | {:error, BridgeError.t()}
  def info do
    GenServer.call(__MODULE__, :info, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :pytorch_port)}
  end

  @doc """
  Perform selective scan on GPU using PyTorch.

  ## Arguments

  * `x` - Input tensor `[batch, seq_len, hidden]`, f32
  * `dt` - Delta tensor `[batch, seq_len, hidden]`, f32
  * `a` - State transition `[hidden, state]`, f32
  * `b` - Input projection `[batch, seq_len, state]`, f32
  * `c` - Output projection `[batch, seq_len, state]`, f32

  ## Returns

  Output tensor `[batch, seq_len, hidden]`, f32
  """
  @spec selective_scan(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {:ok, Nx.Tensor.t()} | {:error, BridgeError.t()}
  def selective_scan(x, dt, a, b, c) do
    GenServer.call(__MODULE__, {:scan, x, dt, a, b, c}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :pytorch_port, context: %{operation: :selective_scan})}
  end

  @doc """
  Perform selective scan, raising on error.
  """
  @spec selective_scan!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def selective_scan!(x, dt, a, b, c) do
    case selective_scan(x, dt, a, b, c) do
      {:ok, result} -> result
      {:error, reason} -> raise "PyTorch scan failed: #{inspect(reason)}"
    end
  end

  # ==========================================================================
  # GenServer Implementation
  # ==========================================================================

  @impl true
  def init(_opts) do
    python_script = Application.app_dir(:exphil, "priv/python/pytorch_scan_server.py")

    port =
      Port.open({:spawn_executable, System.find_executable("python3")}, [
        :binary,
        :exit_status,
        {:packet, 4},
        {:args, [python_script]}
      ])

    # Wait for server to initialize
    Process.sleep(500)

    Logger.info("[PyTorchPort] Started PyTorch scan server")
    {:ok, %{port: port}}
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
  def handle_call({:scan, x, dt, a, b, c}, _from, state) do
    # Get shapes
    {batch, seq_len, hidden} = Nx.shape(x)
    {^hidden, state_size} = Nx.shape(a)

    # Convert tensors to binary (ensure f32 and on CPU)
    x_bin = tensor_to_binary(x)
    dt_bin = tensor_to_binary(dt)
    a_bin = tensor_to_binary(a)
    b_bin = tensor_to_binary(b)
    c_bin = tensor_to_binary(c)

    # Build request
    request = %{
      op: "scan",
      batch: batch,
      seq_len: seq_len,
      hidden: hidden,
      state: state_size,
      x: x_bin,
      dt: dt_bin,
      A: a_bin,
      B: b_bin,
      C: c_bin
    }

    # Send and receive
    case send_request(state.port, request) do
      {:ok, %{"status" => "ok", "result" => result_bin, "shape" => shape}} ->
        result =
          result_bin
          |> Nx.from_binary(:f32)
          |> Nx.reshape(List.to_tuple(shape))

        {:reply, {:ok, result}, state}

      {:ok, %{"status" => "error", "message" => msg}} ->
        {:reply, {:error, msg}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_info({port, {:data, _data}}, %{port: port} = state) do
    # Unexpected data - log and ignore
    Logger.warning("[PyTorchPort] Unexpected data from port")
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[PyTorchPort] Python process exited with status #{status}")
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
    # Force evaluation and copy to avoid "donated buffer" errors
    # when EXLA tensors get garbage collected
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
          {:error, reason} -> {:error, BridgeError.new(:protocol_error, bridge: :pytorch_port, context: %{details: "decode error: #{inspect(reason)}"})}
        end
    after
      @timeout -> {:error, BridgeError.new(:timeout, bridge: :pytorch_port, context: %{timeout_ms: @timeout})}
    end
  end
end
