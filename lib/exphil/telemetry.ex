defmodule ExPhil.Telemetry do
  @moduledoc """
  Telemetry event definitions and handlers for ExPhil.

  This module defines telemetry events emitted during training, inference,
  and gameplay. Attach handlers to these events for logging, metrics
  collection, or external integrations (e.g., Wandb).

  ## Events

  ### Training Events
  - `[:exphil, :training, :step]` - Emitted after each training step
  - `[:exphil, :training, :epoch]` - Emitted after each epoch
  - `[:exphil, :training, :checkpoint]` - Emitted when checkpoint saved

  ### Inference Events
  - `[:exphil, :agent, :action]` - Emitted when agent produces action
  - `[:exphil, :agent, :inference]` - Inference timing

  ### Game Events
  - `[:exphil, :bridge, :step]` - Game state received
  - `[:exphil, :bridge, :game_end]` - Game ended

  ## Usage

      # Attach a handler
      :telemetry.attach(
        "my-handler",
        [:exphil, :training, :step],
        &MyModule.handle_event/4,
        nil
      )

      # Or use the built-in console logger
      ExPhil.Telemetry.attach_console_logger()

  """

  require Logger

  # ============================================================================
  # Event Emission Helpers
  # ============================================================================

  @doc """
  Emit a training step event.
  """
  def training_step(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :training, :step],
      measurements,
      metadata
    )
  end

  @doc """
  Emit a training epoch event.
  """
  def training_epoch(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :training, :epoch],
      measurements,
      metadata
    )
  end

  @doc """
  Emit a checkpoint saved event.
  """
  def checkpoint_saved(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :training, :checkpoint],
      measurements,
      metadata
    )
  end

  @doc """
  Emit an agent action event.
  """
  def agent_action(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :agent, :action],
      measurements,
      metadata
    )
  end

  @doc """
  Emit an inference timing event.
  """
  def agent_inference(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :agent, :inference],
      measurements,
      metadata
    )
  end

  @doc """
  Emit a game step event.
  """
  def bridge_step(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :bridge, :step],
      measurements,
      metadata
    )
  end

  @doc """
  Emit a game end event.
  """
  def game_end(measurements, metadata \\ %{}) do
    :telemetry.execute(
      [:exphil, :bridge, :game_end],
      measurements,
      metadata
    )
  end

  # ============================================================================
  # Built-in Handlers
  # ============================================================================

  @doc """
  Attach a console logger that logs all ExPhil telemetry events.
  """
  def attach_console_logger(opts \\ []) do
    events = [
      [:exphil, :training, :step],
      [:exphil, :training, :epoch],
      [:exphil, :training, :checkpoint],
      [:exphil, :agent, :action],
      [:exphil, :agent, :inference],
      [:exphil, :bridge, :step],
      [:exphil, :bridge, :game_end]
    ]

    level = Keyword.get(opts, :level, :debug)

    Enum.each(events, fn event ->
      :telemetry.attach(
        "exphil-console-#{inspect(event)}",
        event,
        &handle_console_log/4,
        %{level: level}
      )
    end)
  end

  @doc """
  Detach the console logger.
  """
  def detach_console_logger do
    events = [
      [:exphil, :training, :step],
      [:exphil, :training, :epoch],
      [:exphil, :training, :checkpoint],
      [:exphil, :agent, :action],
      [:exphil, :agent, :inference],
      [:exphil, :bridge, :step],
      [:exphil, :bridge, :game_end]
    ]

    Enum.each(events, fn event ->
      :telemetry.detach("exphil-console-#{inspect(event)}")
    end)
  end

  defp handle_console_log(event, measurements, metadata, config) do
    level = Map.get(config, :level, :debug)
    event_name = Enum.join(event, ".")

    case level do
      :debug -> Logger.debug("[Telemetry] #{event_name}: #{inspect(measurements)}, #{inspect(metadata)}")
      :info -> Logger.info("[Telemetry] #{event_name}: #{inspect(measurements)}")
      _ -> :ok
    end
  end

  # ============================================================================
  # Metrics Collection
  # ============================================================================

  @doc """
  Start a metrics collector process.

  The collector accumulates metrics in memory and can be periodically
  flushed to an external system.
  """
  def start_collector(opts \\ []) do
    GenServer.start_link(__MODULE__.Collector, opts, name: __MODULE__.Collector)
  end

  @doc """
  Get accumulated metrics from the collector.
  """
  def get_metrics do
    if Process.whereis(__MODULE__.Collector) do
      GenServer.call(__MODULE__.Collector, :get_metrics)
    else
      {:error, :collector_not_started}
    end
  end

  @doc """
  Reset/flush collected metrics.
  """
  def flush_metrics do
    if Process.whereis(__MODULE__.Collector) do
      GenServer.call(__MODULE__.Collector, :flush)
    else
      {:error, :collector_not_started}
    end
  end
end

defmodule ExPhil.Telemetry.Collector do
  @moduledoc false
  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    # Attach to all ExPhil telemetry events
    events = [
      [:exphil, :training, :step],
      [:exphil, :training, :epoch],
      [:exphil, :training, :checkpoint],
      [:exphil, :agent, :inference],
      [:exphil, :bridge, :game_end]
    ]

    Enum.each(events, fn event ->
      :telemetry.attach(
        "exphil-collector-#{inspect(event)}",
        event,
        &__MODULE__.handle_event/4,
        %{collector: self()}
      )
    end)

    {:ok, %{metrics: %{}, counts: %{}}}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:flush, _from, _state) do
    {:reply, :ok, %{metrics: %{}, counts: %{}}}
  end

  @impl true
  def handle_cast({:event, event, measurements}, state) do
    event_key = Enum.join(event, ".")

    # Accumulate measurements
    metrics = Map.update(
      state.metrics,
      event_key,
      measurements,
      fn existing ->
        Map.merge(existing, measurements, fn _k, v1, v2 ->
          (v1 + v2) / 2  # Simple running average
        end)
      end
    )

    counts = Map.update(state.counts, event_key, 1, &(&1 + 1))

    {:noreply, %{state | metrics: metrics, counts: counts}}
  end

  def handle_event(event, measurements, _metadata, %{collector: pid}) do
    GenServer.cast(pid, {:event, event, measurements})
  end
end
