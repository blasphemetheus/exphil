defmodule ExPhil.Bridge.FlashAttentionPort do
  @moduledoc """
  Elixir Port interface to FlashAttention-2 via Python/PyTorch.

  This module provides GPU-accelerated flash attention by calling out to
  Python's flash_attn library. This is a prototype for experimentation -
  for production, consider the NIF approach.

  ## Usage

      # Start the server (usually done by application supervisor)
      {:ok, pid} = ExPhil.Bridge.FlashAttentionPort.start_link()

      # Check if available (requires flash_attn Python package)
      ExPhil.Bridge.FlashAttentionPort.available?()

      # Run flash attention
      {:ok, output} = ExPhil.Bridge.FlashAttentionPort.forward(query, key, value, causal: true)

  ## Architecture

      ┌─────────────┐     msgpack      ┌──────────────┐
      │   Elixir    │◄───────────────►│    Python    │
      │  GenServer  │   Port (stdio)   │ FlashAttn-2  │
      └─────────────┘                  └──────────────┘

  ## Performance

  The Python bridge adds overhead (~1-2ms) for tensor serialization. This is
  suitable for experimentation and batch processing, but not real-time inference.
  For production, use the NIF implementation or wait for EXLA flash attention.

  ## Requirements

      # Python dependencies
      pip install torch flash-attn --no-build-isolation msgpack

  """

  use GenServer
  require Logger

  alias ExPhil.Error.BridgeError

  @timeout 30_000

  # ==========================================================================
  # Client API
  # ==========================================================================

  @doc """
  Start the FlashAttention Port server.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if the FlashAttention server is available and responding.
  """
  @spec available?() :: boolean()
  def available? do
    case ping() do
      {:ok, %{"flash_attn" => true}} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @doc """
  Check if server is running (may not have flash_attn installed).
  """
  @spec running?() :: boolean()
  def running? do
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
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :flash_attention)}
  end

  @doc """
  Get server info (device, CUDA availability, flash_attn status).
  """
  @spec info() :: {:ok, map()} | {:error, BridgeError.t()}
  def info do
    GenServer.call(__MODULE__, :info, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :flash_attention)}
  end

  @doc """
  Run FlashAttention-2 forward pass.

  ## Arguments

  * `query` - Query tensor `[batch, seq_len, dim]`, f32
  * `key` - Key tensor `[batch, seq_len, dim]`, f32
  * `value` - Value tensor `[batch, seq_len, dim]`, f32
  * `opts` - Options:
    * `:causal` - Apply causal masking (default: true)
    * `:num_heads` - Number of attention heads (default: 1)
    * `:use_flash` - Use flash attention if available (default: true)

  ## Returns

  Output tensor `[batch, seq_len, dim]`, f32
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, BridgeError.t()}
  def forward(query, key, value, opts \\ []) do
    GenServer.call(__MODULE__, {:forward, query, key, value, opts}, @timeout)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :flash_attention, context: %{operation: :forward})}
  end

  @doc """
  Run FlashAttention forward pass, raising on error.
  """
  @spec forward!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward!(query, key, value, opts \\ []) do
    case forward(query, key, value, opts) do
      {:ok, result} -> result
      {:error, reason} -> raise "FlashAttention failed: #{inspect(reason)}"
    end
  end

  @doc """
  Run a benchmark comparing flash attention vs standard attention.

  ## Options

  * `:batch` - Batch size (default: 4)
  * `:seq_len` - Sequence length (default: 128)
  * `:dim` - Hidden dimension (default: 256)
  * `:num_iters` - Number of iterations (default: 10)

  ## Returns

  Map with timing results in milliseconds.
  """
  @spec benchmark(keyword()) :: {:ok, map()} | {:error, BridgeError.t()}
  def benchmark(opts \\ []) do
    GenServer.call(__MODULE__, {:benchmark, opts}, @timeout * 2)
  catch
    :exit, _ -> {:error, BridgeError.new(:not_running, bridge: :flash_attention, context: %{operation: :benchmark})}
  end

  # ==========================================================================
  # GenServer Implementation
  # ==========================================================================

  @impl true
  def init(_opts) do
    python_script = Application.app_dir(:exphil, "priv/python/flash_attention_server.py")

    port =
      Port.open({:spawn_executable, System.find_executable("python3")}, [
        :binary,
        :exit_status,
        {:packet, 4},
        {:args, [python_script]}
      ])

    # Wait for server to initialize
    Process.sleep(1000)

    Logger.info("[FlashAttentionPort] Started FlashAttention server")
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
  def handle_call({:forward, query, key, value, opts}, _from, state) do
    # Get shapes
    {batch, seq_len, dim} = Nx.shape(query)
    causal = Keyword.get(opts, :causal, true)
    num_heads = Keyword.get(opts, :num_heads, 1)
    use_flash = Keyword.get(opts, :use_flash, true)

    # Convert tensors to binary (ensure f32 and on CPU)
    q_bin = tensor_to_binary(query)
    k_bin = tensor_to_binary(key)
    v_bin = tensor_to_binary(value)

    # Build request
    request = %{
      op: "forward",
      batch: batch,
      seq_len: seq_len,
      dim: dim,
      num_heads: num_heads,
      causal: causal,
      use_flash: use_flash,
      q: q_bin,
      k: k_bin,
      v: v_bin
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
  def handle_call({:benchmark, opts}, _from, state) do
    request = %{
      op: "benchmark",
      batch: Keyword.get(opts, :batch, 4),
      seq_len: Keyword.get(opts, :seq_len, 128),
      dim: Keyword.get(opts, :dim, 256),
      num_iters: Keyword.get(opts, :num_iters, 10)
    }

    response = send_request(state.port, request)
    {:reply, response, state}
  end

  @impl true
  def handle_info({port, {:data, _data}}, %{port: port} = state) do
    # Unexpected data - log and ignore
    Logger.warning("[FlashAttentionPort] Unexpected data from port")
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[FlashAttentionPort] Python process exited with status #{status}")
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
          {:error, reason} -> {:error, BridgeError.new(:protocol_error, bridge: :flash_attention, context: %{details: "decode error: #{inspect(reason)}"})}
        end
    after
      @timeout -> {:error, BridgeError.new(:timeout, bridge: :flash_attention, context: %{timeout_ms: @timeout})}
    end
  end
end
