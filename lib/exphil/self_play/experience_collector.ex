defmodule ExPhil.SelfPlay.ExperienceCollector do
  @moduledoc """
  GenServer that collects and batches experiences from multiple game runners.

  Aggregates experiences into batches suitable for PPO training. Supports
  configurable batch sizes, buffer limits, and callbacks for when batches
  are ready.

  ## Architecture

      ┌────────────────────────────────────────────────────────────────────┐
      │                     ExperienceCollector                             │
      │                                                                     │
      │  ┌────────────────────────────────────────────────────────────┐    │
      │  │                    Experience Buffer                        │    │
      │  │  [exp_1, exp_2, exp_3, ..., exp_N]                         │    │
      │  └────────────────────────────────────────────────────────────┘    │
      │                            │                                        │
      │                            ▼                                        │
      │  ┌────────────────────────────────────────────────────────────┐    │
      │  │                   Batch Aggregator                          │    │
      │  │  Converts list of experiences to tensors for PPO           │    │
      │  └────────────────────────────────────────────────────────────┘    │
      │                            │                                        │
      │                            ▼                                        │
      │  ┌────────────────────────────────────────────────────────────┐    │
      │  │                   Ready Callback                            │    │
      │  │  Notifies trainer when batch_size experiences collected    │    │
      │  └────────────────────────────────────────────────────────────┘    │
      │                                                                     │
      └────────────────────────────────────────────────────────────────────┘

  ## Usage

      # Start the collector
      {:ok, collector} = ExperienceCollector.start_link(
        batch_size: 2048,
        max_buffer_size: 10_000,
        ready_callback: fn experiences ->
          # Train PPO with experiences
          PPO.update(trainer, experiences)
        end
      )

      # Submit experiences from game runners
      :ok = ExperienceCollector.submit(collector, experience)

      # Get a batch manually (blocking)
      {:ok, batch} = ExperienceCollector.get_batch(collector, 2048)

      # Get current buffer stats
      stats = ExperienceCollector.get_stats(collector)
      # => %{buffer_size: 1500, total_submitted: 50000, batches_produced: 24}

      # Flush the buffer
      {:ok, remaining} = ExperienceCollector.flush(collector)

  """

  use GenServer

  require Logger

  defstruct [
    :buffer,
    :batch_size,
    :max_buffer_size,
    :ready_callback,
    :total_submitted,
    :batches_produced,
    # List of {from, size} for get_batch callers
    :pending_requests,
    # Whether to auto-call callback when batch ready
    :auto_batch
  ]

  @type experience :: %{
          state: Nx.Tensor.t(),
          action: map(),
          log_prob: Nx.Tensor.t(),
          value: Nx.Tensor.t(),
          reward: float(),
          done: boolean()
        }

  @type batch :: %{
          states: Nx.Tensor.t(),
          actions: map(),
          log_probs: Nx.Tensor.t(),
          values: Nx.Tensor.t(),
          rewards: Nx.Tensor.t(),
          dones: Nx.Tensor.t()
        }

  @default_opts %{
    batch_size: 2048,
    max_buffer_size: 100_000,
    auto_batch: false
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the ExperienceCollector.

  ## Options
    - `:batch_size` - Number of experiences per batch (default: 2048)
    - `:max_buffer_size` - Maximum buffer size before dropping (default: 100_000)
    - `:ready_callback` - Function called when batch is ready (optional)
    - `:auto_batch` - Auto-produce batches when buffer reaches batch_size (default: false)
    - `:name` - Name for the GenServer
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Submits an experience to the collector.

  This is an asynchronous cast for high-throughput submission.
  """
  def submit(collector, experience) do
    GenServer.cast(collector, {:submit, experience})
  end

  @doc """
  Submits multiple experiences at once.
  """
  def submit_batch(collector, experiences) when is_list(experiences) do
    GenServer.cast(collector, {:submit_batch, experiences})
  end

  @doc """
  Gets a batch of experiences.

  Blocks until `size` experiences are available or timeout.
  """
  def get_batch(collector, size, timeout \\ 60_000) do
    GenServer.call(collector, {:get_batch, size}, timeout)
  end

  @doc """
  Gets all available experiences without waiting.
  """
  def get_all(collector) do
    GenServer.call(collector, :get_all)
  end

  @doc """
  Flushes the buffer and returns all remaining experiences.
  """
  def flush(collector) do
    GenServer.call(collector, :flush)
  end

  @doc """
  Gets the current buffer size.
  """
  def buffer_size(collector) do
    GenServer.call(collector, :buffer_size)
  end

  @doc """
  Gets collector statistics.
  """
  def get_stats(collector) do
    GenServer.call(collector, :get_stats)
  end

  @doc """
  Resets the collector state.
  """
  def reset(collector) do
    GenServer.call(collector, :reset)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    batch_size = Keyword.get(opts, :batch_size, @default_opts.batch_size)
    max_buffer_size = Keyword.get(opts, :max_buffer_size, @default_opts.max_buffer_size)
    ready_callback = Keyword.get(opts, :ready_callback)
    auto_batch = Keyword.get(opts, :auto_batch, @default_opts.auto_batch)

    state = %__MODULE__{
      buffer: [],
      batch_size: batch_size,
      max_buffer_size: max_buffer_size,
      ready_callback: ready_callback,
      total_submitted: 0,
      batches_produced: 0,
      pending_requests: [],
      auto_batch: auto_batch
    }

    Logger.debug(
      "[ExperienceCollector] Started with batch_size=#{batch_size}, max_buffer=#{max_buffer_size}"
    )

    {:ok, state}
  end

  @impl true
  def handle_cast({:submit, experience}, state) do
    new_state = add_to_buffer(state, [experience])
    {:noreply, maybe_process_batch(new_state)}
  end

  @impl true
  def handle_cast({:submit_batch, experiences}, state) do
    new_state = add_to_buffer(state, experiences)
    {:noreply, maybe_process_batch(new_state)}
  end

  @impl true
  def handle_call({:get_batch, size}, from, state) do
    buffer_len = length(state.buffer)

    if buffer_len >= size do
      # Have enough, return immediately
      {batch_exps, remaining} = Enum.split(state.buffer, size)
      batch = experiences_to_batch(batch_exps)

      new_state = %{state | buffer: remaining, batches_produced: state.batches_produced + 1}

      {:reply, {:ok, batch}, new_state}
    else
      # Not enough, add to pending requests
      new_state = %{state | pending_requests: [{from, size} | state.pending_requests]}
      {:noreply, new_state}
    end
  end

  @impl true
  def handle_call(:get_all, _from, state) do
    if length(state.buffer) > 0 do
      batch = experiences_to_batch(state.buffer)
      new_state = %{state | buffer: [], batches_produced: state.batches_produced + 1}
      {:reply, {:ok, batch}, new_state}
    else
      {:reply, {:ok, empty_batch()}, state}
    end
  end

  @impl true
  def handle_call(:flush, _from, state) do
    remaining = state.buffer

    batch =
      if length(remaining) > 0 do
        experiences_to_batch(remaining)
      else
        empty_batch()
      end

    new_state = %{
      state
      | buffer: [],
        batches_produced: state.batches_produced + if(length(remaining) > 0, do: 1, else: 0)
    }

    {:reply, {:ok, batch}, new_state}
  end

  @impl true
  def handle_call(:buffer_size, _from, state) do
    {:reply, length(state.buffer), state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = %{
      buffer_size: length(state.buffer),
      batch_size: state.batch_size,
      max_buffer_size: state.max_buffer_size,
      total_submitted: state.total_submitted,
      batches_produced: state.batches_produced,
      pending_requests: length(state.pending_requests)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    new_state = %{
      state
      | buffer: [],
        total_submitted: 0,
        batches_produced: 0,
        pending_requests: []
    }

    {:reply, :ok, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp add_to_buffer(state, experiences) do
    new_buffer = state.buffer ++ experiences
    new_total = state.total_submitted + length(experiences)

    # Trim if over max size (drop oldest)
    trimmed_buffer =
      if length(new_buffer) > state.max_buffer_size do
        excess = length(new_buffer) - state.max_buffer_size

        Logger.warning(
          "[ExperienceCollector] Buffer overflow, dropping #{excess} oldest experiences"
        )

        Enum.drop(new_buffer, excess)
      else
        new_buffer
      end

    %{state | buffer: trimmed_buffer, total_submitted: new_total}
  end

  defp maybe_process_batch(state) do
    state
    |> maybe_serve_pending_requests()
    |> maybe_auto_batch()
  end

  defp maybe_serve_pending_requests(state) do
    buffer_len = length(state.buffer)

    {fulfilled, remaining_requests} =
      Enum.split_with(state.pending_requests, fn {_from, size} ->
        buffer_len >= size
      end)

    if length(fulfilled) > 0 do
      # Serve the first fulfilled request (FIFO order)
      [{from, size} | rest_fulfilled] = Enum.reverse(fulfilled)

      {batch_exps, remaining_buffer} = Enum.split(state.buffer, size)
      batch = experiences_to_batch(batch_exps)

      GenServer.reply(from, {:ok, batch})

      new_state = %{
        state
        | buffer: remaining_buffer,
          batches_produced: state.batches_produced + 1,
          pending_requests: Enum.reverse(rest_fulfilled) ++ remaining_requests
      }

      # Recursively check if more requests can be served
      maybe_serve_pending_requests(new_state)
    else
      state
    end
  end

  defp maybe_auto_batch(%{auto_batch: false} = state), do: state

  defp maybe_auto_batch(state) do
    buffer_len = length(state.buffer)

    if buffer_len >= state.batch_size && state.ready_callback do
      {batch_exps, remaining} = Enum.split(state.buffer, state.batch_size)
      batch = experiences_to_batch(batch_exps)

      # Call the callback asynchronously to not block
      callback = state.ready_callback
      Task.start(fn -> callback.(batch) end)

      new_state = %{state | buffer: remaining, batches_produced: state.batches_produced + 1}

      # Check if another batch can be produced
      maybe_auto_batch(new_state)
    else
      state
    end
  end

  defp experiences_to_batch(experiences) when length(experiences) == 0 do
    empty_batch()
  end

  defp experiences_to_batch(experiences) do
    # Stack states
    states =
      experiences
      |> Enum.map(& &1.state)
      |> stack_tensors()

    # Stack scalar values
    rewards =
      experiences
      |> Enum.map(& &1.reward)
      |> Nx.tensor(type: :f32)

    dones =
      experiences
      |> Enum.map(&if(&1.done, do: 1.0, else: 0.0))
      |> Nx.tensor(type: :f32)

    values =
      experiences
      |> Enum.map(fn exp ->
        case exp.value do
          %Nx.Tensor{} = t -> Nx.to_number(t)
          n when is_number(n) -> n
        end
      end)
      |> Nx.tensor(type: :f32)

    log_probs =
      experiences
      |> Enum.map(fn exp ->
        case exp.log_prob do
          %Nx.Tensor{} = t -> Nx.to_number(t)
          n when is_number(n) -> n
        end
      end)
      |> Nx.tensor(type: :f32)

    # Stack actions (map of tensors)
    actions = stack_actions(Enum.map(experiences, & &1.action))

    %{
      states: states,
      actions: actions,
      rewards: rewards,
      values: values,
      log_probs: log_probs,
      dones: dones
    }
  end

  defp stack_tensors([]), do: Nx.tensor([], type: :f32)

  defp stack_tensors(tensors) do
    # Handle both raw tensors and tensors that need squeezing
    processed =
      Enum.map(tensors, fn t ->
        case Nx.shape(t) do
          {1, _n} -> Nx.squeeze(t, axes: [0])
          _ -> t
        end
      end)

    Nx.stack(processed)
  end

  defp stack_actions(actions_list) when length(actions_list) == 0, do: %{}

  defp stack_actions(actions_list) do
    keys = Map.keys(hd(actions_list))

    Map.new(keys, fn key ->
      values =
        Enum.map(actions_list, fn action ->
          case Map.get(action, key) do
            %Nx.Tensor{} = t -> t
            n when is_number(n) -> Nx.tensor(n)
            other -> Nx.tensor(other)
          end
        end)

      {key, Nx.stack(values)}
    end)
  end

  defp empty_batch do
    %{
      states: Nx.tensor([], type: :f32),
      actions: %{},
      rewards: Nx.tensor([], type: :f32),
      values: Nx.tensor([], type: :f32),
      log_probs: Nx.tensor([], type: :f32),
      dones: Nx.tensor([], type: :f32)
    }
  end
end
