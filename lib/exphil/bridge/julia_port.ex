defmodule ExPhil.Bridge.JuliaPort do
  @moduledoc """
  Elixir Port interface to Julia for GPU linear scan operations.

  Julia's CUDA.jl provides near-CUDA-C performance with significantly less code.
  A long-lived server process amortizes Julia's ~30s JIT cold-start.

  ## Usage

      {:ok, pid} = ExPhil.Bridge.JuliaPort.start_link()
      ExPhil.Bridge.JuliaPort.available?()
      {:ok, result} = ExPhil.Bridge.JuliaPort.linear_scan(a, b, h0)

  ## Architecture

      ┌─────────────┐     msgpack      ┌──────────────┐
      │   Elixir    │◄───────────────►│    Julia     │
      │  GenServer  │   Port (stdio)   │   CUDA.jl    │
      └─────────────┘                  └──────────────┘
  """

  use GenServer
  require Logger

  alias ExPhil.Error.BridgeError

  @timeout 60_000  # 60s — Julia JIT warmup can be slow

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
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :julia_port)}
  end

  @spec info() :: {:ok, map()} | {:error, BridgeError.t()}
  def info do
    GenServer.call(__MODULE__, :info, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :julia_port)}
  end

  @doc """
  Perform fused linear scan: h = a * h + b over timesteps.

  ## Arguments

  * `a` - Decay coefficients `[batch, seq_len, hidden]`, f32
  * `b` - Additive terms `[batch, seq_len, hidden]`, f32
  * `h0` - Initial hidden state `[batch, hidden]`, f32

  ## Options

  * `:mode` - `"cuda"` (CUDA.jl kernel, default), `"ka"` (KernelAbstractions), `"cpu"`

  ## Returns

  Output tensor `[batch, seq_len, hidden]`, f32
  """
  @spec linear_scan(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, BridgeError.t()}
  def linear_scan(a, b, h0, opts \\ []) do
    GenServer.call(__MODULE__, {:scan, a, b, h0, opts}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :julia_port, context: %{operation: :linear_scan})}
  end

  @spec linear_scan!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def linear_scan!(a, b, h0, opts \\ []) do
    case linear_scan(a, b, h0, opts) do
      {:ok, result} -> result
      {:error, reason} -> raise "Julia scan failed: #{inspect(reason)}"
    end
  end

  @doc """
  Run benchmark on Julia side (avoids serialization overhead in timing).
  Returns timing stats in microseconds.
  """
  @spec benchmark(pos_integer(), pos_integer(), pos_integer(), keyword()) ::
          {:ok, map()} | {:error, BridgeError.t()}
  def benchmark(batch, seq_len, hidden, opts \\ []) do
    GenServer.call(__MODULE__, {:benchmark, batch, seq_len, hidden, opts}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :julia_port)}
  end

  # ==========================================================================
  # GenServer Implementation
  # ==========================================================================

  @impl true
  def init(_opts) do
    julia_script = Path.join([File.cwd!(), "native", "julia_scan", "linear_scan_server.jl"])
    project_dir = Path.join([File.cwd!(), "native", "julia_scan"])

    julia_exe = System.find_executable("julia")

    if julia_exe == nil do
      Logger.error("[JuliaPort] julia not found in PATH")
      {:stop, :julia_not_found}
    else
      port =
        Port.open({:spawn_executable, julia_exe}, [
          :binary,
          :exit_status,
          {:packet, 4},
          {:args, ["--project=#{project_dir}", julia_script]}
        ])

      Logger.info("[JuliaPort] Starting Julia scan server (JIT warmup may take 30-60s)")
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
    mode = Keyword.get(opts, :mode, "cuda")

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
      {:ok, %{"status" => "ok", "result" => result_data, "shape" => shape}} ->
        result_bin = if is_list(result_data), do: :erlang.list_to_binary(result_data), else: result_data

        result =
          result_bin
          |> Nx.from_binary(:f32)
          |> Nx.reshape(List.to_tuple(shape))

        {:reply, {:ok, result}, state}

      {:ok, %{"status" => "error", "message" => msg}} ->
        {:reply, {:error, BridgeError.new(:protocol_error, bridge: :julia_port, context: %{message: msg})}, state}

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
    Logger.warning("[JuliaPort] Unexpected data from port")
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[JuliaPort] Julia process exited with status #{status}")
    {:stop, {:julia_exit, status}, state}
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
          {:error, reason} -> {:error, BridgeError.new(:protocol_error, bridge: :julia_port, context: %{details: "decode error: #{inspect(reason)}"})}
        end
    after
      @timeout -> {:error, BridgeError.new(:timeout, bridge: :julia_port, context: %{timeout_ms: @timeout})}
    end
  end
end
