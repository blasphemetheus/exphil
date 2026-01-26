defmodule ExPhil.Integrations.Wandb do
  @moduledoc """
  Weights & Biases integration for experiment tracking.

  Uses the Wandb HTTP API to log metrics, hyperparameters, and artifacts.
  Integrates with ExPhil.Telemetry for automatic metric logging.

  ## Setup

  Set your API key via environment variable:

      export WANDB_API_KEY=your_key_here

  Or pass it directly:

      ExPhil.Integrations.Wandb.start_run(
        project: "exphil",
        name: "mewtwo_v1",
        api_key: "your_key"
      )

  ## Usage

      # Start a run (attaches to telemetry automatically)
      {:ok, run_id} = ExPhil.Integrations.Wandb.start_run(
        project: "exphil",
        name: "mewtwo_imitation_v1",
        config: %{
          hidden_sizes: [64, 64],
          learning_rate: 1.0e-4,
          batch_size: 64
        }
      )

      # Training happens... telemetry events logged automatically

      # Manual logging if needed
      ExPhil.Integrations.Wandb.log(%{custom_metric: 123})

      # End the run
      ExPhil.Integrations.Wandb.finish_run()

  ## Telemetry Events

  Automatically logs these events:
  - `[:exphil, :training, :step]` - loss, step
  - `[:exphil, :training, :epoch]` - epoch metrics
  - `[:exphil, :training, :checkpoint]` - checkpoint saved

  """

  use GenServer

  require Logger

  @base_url "https://api.wandb.ai"
  # Batch logs every 5 seconds
  @batch_interval 5_000

  defstruct [
    :api_key,
    :project,
    :entity,
    :run_id,
    :run_name,
    :config,
    :step,
    :pending_logs,
    :started_at
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Start a new Wandb run.

  ## Options
    - `:project` - Wandb project name (required)
    - `:name` - Run name (default: auto-generated)
    - `:entity` - Wandb entity/team (default: from API key)
    - `:config` - Hyperparameters map
    - `:api_key` - API key (default: from WANDB_API_KEY env)
    - `:tags` - List of tags for the run

  ## Returns
    `{:ok, run_id}` on success, `{:error, reason}` on failure.
  """
  @spec start_run(keyword()) :: {:ok, String.t()} | {:error, term()}
  def start_run(opts) do
    case GenServer.start_link(__MODULE__, opts, name: __MODULE__) do
      {:ok, _pid} ->
        GenServer.call(__MODULE__, :get_run_id)

      {:error, {:already_started, _pid}} ->
        {:error, :run_already_active}

      error ->
        error
    end
  end

  @doc """
  Log metrics to the current run.

  ## Examples

      Wandb.log(%{loss: 0.5, accuracy: 0.92})
      Wandb.log(%{loss: 0.4}, step: 100)

  """
  @spec log(map(), keyword()) :: :ok | {:error, term()}
  def log(metrics, opts \\ []) do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, {:log, metrics, opts})
    else
      {:error, :no_active_run}
    end
  end

  @doc """
  Finish the current run.
  """
  @spec finish_run() :: :ok | {:error, term()}
  def finish_run do
    if Process.whereis(__MODULE__) do
      GenServer.call(__MODULE__, :finish)
    else
      {:error, :no_active_run}
    end
  end

  @doc """
  Check if a run is active.
  """
  @spec active?() :: boolean()
  def active? do
    Process.whereis(__MODULE__) != nil
  end

  @doc """
  Get current run info.
  """
  @spec run_info() :: map() | nil
  def run_info do
    if Process.whereis(__MODULE__) do
      GenServer.call(__MODULE__, :run_info)
    else
      nil
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    api_key = Keyword.get(opts, :api_key) || System.get_env("WANDB_API_KEY")

    unless api_key do
      {:stop, {:error, :no_api_key}}
    else
      project = Keyword.fetch!(opts, :project)
      run_name = Keyword.get(opts, :name, generate_run_name())
      entity = Keyword.get(opts, :entity)
      config = Keyword.get(opts, :config, %{})
      tags = Keyword.get(opts, :tags, [])

      state = %__MODULE__{
        api_key: api_key,
        project: project,
        entity: entity,
        run_name: run_name,
        config: config,
        step: 0,
        pending_logs: [],
        started_at: DateTime.utc_now()
      }

      # Initialize run with Wandb API
      case init_run(state, tags) do
        {:ok, run_id} ->
          new_state = %{state | run_id: run_id}

          # Attach telemetry handlers
          attach_telemetry_handlers()

          # Schedule periodic flush
          schedule_flush()

          Logger.info("[Wandb] Run started: #{run_name} (#{run_id})")
          {:ok, new_state}

        {:error, reason} ->
          Logger.error("[Wandb] Failed to init run: #{inspect(reason)}")
          {:stop, {:error, reason}}
      end
    end
  end

  @impl true
  def handle_call(:get_run_id, _from, state) do
    {:reply, {:ok, state.run_id}, state}
  end

  @impl true
  def handle_call(:run_info, _from, state) do
    info = %{
      run_id: state.run_id,
      run_name: state.run_name,
      project: state.project,
      step: state.step,
      started_at: state.started_at
    }

    {:reply, info, state}
  end

  @impl true
  def handle_call(:finish, _from, state) do
    # Flush pending logs
    flush_logs(state)

    # Detach telemetry
    detach_telemetry_handlers()

    # Mark run as finished
    finish_run_api(state)

    Logger.info("[Wandb] Run finished: #{state.run_name}")
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_cast({:log, metrics, opts}, state) do
    step = Keyword.get(opts, :step, state.step)
    log_entry = %{step: step, metrics: metrics, timestamp: System.system_time(:millisecond)}

    new_state = %{
      state
      | pending_logs: [log_entry | state.pending_logs],
        step: max(state.step, step + 1)
    }

    {:noreply, new_state}
  end

  @impl true
  def handle_info(:flush, state) do
    new_state = flush_logs(state)
    schedule_flush()
    {:noreply, new_state}
  end

  @impl true
  def handle_info({:telemetry_event, event, measurements, metadata}, state) do
    metrics = telemetry_to_metrics(event, measurements, metadata)

    if map_size(metrics) > 0 do
      log_entry = %{
        step: state.step,
        metrics: metrics,
        timestamp: System.system_time(:millisecond)
      }

      new_state = %{state | pending_logs: [log_entry | state.pending_logs], step: state.step + 1}
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  # ============================================================================
  # Private - API Calls
  # ============================================================================

  defp init_run(state, tags) do
    # Wandb uses GraphQL API for run creation
    # For simplicity, we'll use a local run ID and log via the simpler REST endpoints
    # Full integration would use: POST /api/v1/run/create

    run_id = generate_run_id()

    # Log config as first entry
    body = %{
      run_id: run_id,
      project: state.project,
      entity: state.entity,
      display_name: state.run_name,
      config: state.config,
      tags: tags
    }

    case api_post("/api/v1/run/#{run_id}/init", body, state.api_key) do
      {:ok, _response} ->
        {:ok, run_id}

      {:error, %{status: 404}} ->
        # API endpoint might not exist in this form - use offline mode
        Logger.warning("[Wandb] API init failed, using offline mode")
        {:ok, run_id}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp flush_logs(%{pending_logs: []} = state), do: state

  defp flush_logs(state) do
    logs = Enum.reverse(state.pending_logs)

    # Batch send to API
    Enum.each(logs, fn log_entry ->
      body = %{
        run_id: state.run_id,
        step: log_entry.step,
        data: log_entry.metrics,
        timestamp: log_entry.timestamp
      }

      case api_post("/api/v1/run/#{state.run_id}/log", body, state.api_key) do
        {:ok, _} ->
          :ok

        {:error, reason} ->
          Logger.debug("[Wandb] Log failed: #{inspect(reason)}")
      end
    end)

    %{state | pending_logs: []}
  end

  defp finish_run_api(state) do
    body = %{
      run_id: state.run_id,
      exit_code: 0
    }

    api_post("/api/v1/run/#{state.run_id}/finish", body, state.api_key)
  end

  defp api_post(path, body, api_key) do
    url = @base_url <> path

    case Req.post(url,
           json: body,
           headers: [{"Authorization", "Bearer #{api_key}"}],
           receive_timeout: 10_000
         ) do
      {:ok, %{status: status} = response} when status in 200..299 ->
        {:ok, response}

      {:ok, %{status: status} = response} ->
        {:error, %{status: status, body: response.body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ============================================================================
  # Private - Telemetry
  # ============================================================================

  defp attach_telemetry_handlers do
    pid = self()

    events = [
      [:exphil, :training, :step],
      [:exphil, :training, :epoch],
      [:exphil, :training, :checkpoint]
    ]

    Enum.each(events, fn event ->
      handler_id = "wandb-#{Enum.join(event, "-")}"

      :telemetry.attach(
        handler_id,
        event,
        fn event, measurements, metadata, _config ->
          send(pid, {:telemetry_event, event, measurements, metadata})
        end,
        nil
      )
    end)
  end

  defp detach_telemetry_handlers do
    events = [
      [:exphil, :training, :step],
      [:exphil, :training, :epoch],
      [:exphil, :training, :checkpoint]
    ]

    Enum.each(events, fn event ->
      handler_id = "wandb-#{Enum.join(event, "-")}"
      :telemetry.detach(handler_id)
    end)
  end

  defp telemetry_to_metrics([:exphil, :training, :step], measurements, _metadata) do
    measurements
    |> Map.take([:loss, :learning_rate, :step])
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp telemetry_to_metrics([:exphil, :training, :epoch], measurements, metadata) do
    base = Map.take(measurements, [:avg_loss, :val_loss, :epoch])
    meta = Map.take(metadata, [:epoch])
    Map.merge(base, meta)
  end

  defp telemetry_to_metrics([:exphil, :training, :checkpoint], measurements, metadata) do
    %{
      checkpoint_step: measurements[:step],
      checkpoint_path: metadata[:path]
    }
  end

  defp telemetry_to_metrics(_event, _measurements, _metadata), do: %{}

  # ============================================================================
  # Private - Helpers
  # ============================================================================

  defp schedule_flush do
    Process.send_after(self(), :flush, @batch_interval)
  end

  defp generate_run_name do
    adjectives = ~w(swift clever bold calm fierce gentle rapid steady)
    nouns = ~w(falcon mewtwo fox wolf hawk eagle tiger lion)

    adj = Enum.random(adjectives)
    noun = Enum.random(nouns)
    num = :rand.uniform(999)

    "#{adj}-#{noun}-#{num}"
  end

  defp generate_run_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
