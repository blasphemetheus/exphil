defmodule ExPhil.Training.AsyncCheckpoint do
  @moduledoc """
  Async checkpoint saving to avoid blocking training.

  Saves checkpoints in a background process so training can continue
  while disk I/O happens in parallel. Uses atomic file writes to
  prevent corruption if training is interrupted during save.

  ## Usage

      # Start the async saver (typically in your app supervision tree)
      {:ok, pid} = AsyncCheckpoint.start_link()

      # Save asynchronously (non-blocking)
      :ok = AsyncCheckpoint.save_async(checkpoint, path)

      # Wait for all pending saves to complete (e.g., at end of training)
      :ok = AsyncCheckpoint.await_pending()

  ## Atomic Writes

  To prevent corruption, checkpoints are written atomically:
  1. Write to a temp file (same directory)
  2. Sync to disk
  3. Rename to final path (atomic on POSIX systems)

  This ensures the checkpoint file is either complete or doesn't exist.
  """

  use GenServer
  require Logger

  @type checkpoint :: map()

  # Client API

  @doc """
  Start the async checkpoint saver.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Save a checkpoint asynchronously.

  Returns immediately while the checkpoint is written in the background.
  If a previous save is still in progress, this save will be queued.

  ## Options
    - `:timeout` - Max time to wait if queue is full (default: 5000ms)
  """
  @spec save_async(checkpoint(), Path.t(), keyword()) :: :ok | {:error, :queue_full}
  def save_async(checkpoint, path, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5000)

    # Deep copy tensors to BinaryBackend before sending to avoid EXLA issues
    checkpoint_copy = deep_binary_copy(checkpoint)

    try do
      GenServer.call(__MODULE__, {:save, checkpoint_copy, path}, timeout)
    catch
      :exit, {:timeout, _} ->
        Logger.warning("Async checkpoint queue timeout - save may be delayed")
        {:error, :queue_full}
    end
  end

  @doc """
  Wait for all pending checkpoint saves to complete.

  Call this at the end of training to ensure all checkpoints are written.
  """
  @spec await_pending(timeout()) :: :ok
  def await_pending(timeout \\ 60_000) do
    GenServer.call(__MODULE__, :await_pending, timeout)
  end

  @doc """
  Get the number of pending saves in the queue.
  """
  @spec pending_count() :: non_neg_integer()
  def pending_count do
    GenServer.call(__MODULE__, :pending_count)
  end

  @doc """
  Check if a save is currently in progress.
  """
  @spec saving?() :: boolean()
  def saving? do
    GenServer.call(__MODULE__, :saving?)
  end

  # Server implementation

  @impl true
  def init(_opts) do
    {:ok,
     %{
       queue: :queue.new(),
       saving: false,
       current_task: nil
     }}
  end

  @impl true
  def handle_call({:save, checkpoint, path}, from, state) do
    # Add to queue
    new_queue = :queue.in({checkpoint, path, from}, state.queue)
    new_state = %{state | queue: new_queue}

    # Start saving if not already
    new_state = maybe_start_save(new_state)

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:await_pending, from, state) do
    if :queue.is_empty(state.queue) and not state.saving do
      {:reply, :ok, state}
    else
      # Add a waiter to be notified when queue is empty
      new_queue = :queue.in({:await, from}, state.queue)
      {:noreply, %{state | queue: new_queue}}
    end
  end

  @impl true
  def handle_call(:pending_count, _from, state) do
    count = :queue.len(state.queue) + if(state.saving, do: 1, else: 0)
    {:reply, count, state}
  end

  @impl true
  def handle_call(:saving?, _from, state) do
    {:reply, state.saving, state}
  end

  @impl true
  def handle_info({ref, :ok}, state) when is_reference(ref) do
    # Task completed successfully
    Process.demonitor(ref, [:flush])
    new_state = %{state | saving: false, current_task: nil}
    new_state = maybe_start_save(new_state)
    {:noreply, new_state}
  end

  @impl true
  def handle_info({ref, {:error, reason}}, state) when is_reference(ref) do
    # Task failed
    Process.demonitor(ref, [:flush])
    Logger.error("Async checkpoint save failed: #{inspect(reason)}")
    new_state = %{state | saving: false, current_task: nil}
    new_state = maybe_start_save(new_state)
    {:noreply, new_state}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    if state.current_task && state.current_task.ref == ref do
      Logger.error("Async checkpoint task crashed: #{inspect(reason)}")
      new_state = %{state | saving: false, current_task: nil}
      new_state = maybe_start_save(new_state)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  # Private helpers

  defp maybe_start_save(%{saving: true} = state), do: state

  defp maybe_start_save(state) do
    case :queue.out(state.queue) do
      {:empty, _} ->
        state

      {{:value, {:await, from}}, new_queue} ->
        # This was a waiter - reply and continue
        GenServer.reply(from, :ok)
        maybe_start_save(%{state | queue: new_queue})

      {{:value, {checkpoint, path, from}}, new_queue} ->
        # Start async save task
        task =
          Task.async(fn ->
            result = atomic_write(checkpoint, path)
            # Reply to the original caller
            GenServer.reply(from, :ok)
            result
          end)

        %{state | queue: new_queue, saving: true, current_task: task}
    end
  end

  @doc """
  Write checkpoint atomically using temp file + rename.

  This ensures the checkpoint is either complete or doesn't exist,
  preventing corruption if training is interrupted.
  """
  @spec atomic_write(checkpoint(), Path.t()) :: :ok | {:error, term()}
  def atomic_write(checkpoint, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    # Use a unique temp file in the same directory
    temp_path = "#{path}.#{System.unique_integer([:positive])}.tmp"

    try do
      # Write to temp file
      binary = :erlang.term_to_binary(checkpoint)
      File.write!(temp_path, binary)

      # Sync to disk (ensures data is persisted before rename)
      case File.open(temp_path, [:raw, :binary]) do
        {:ok, fd} ->
          :file.datasync(fd)
          File.close(fd)

        _ ->
          :ok
      end

      # Atomic rename
      File.rename!(temp_path, path)

      Logger.debug("Saved checkpoint to #{path}")
      :ok
    rescue
      e ->
        # Clean up temp file on error
        File.rm(temp_path)
        {:error, e}
    end
  end

  # Deep copy all tensors to BinaryBackend to avoid EXLA issues
  # when accessing tensors from a different process
  defp deep_binary_copy(%Nx.Tensor{} = tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defp deep_binary_copy(%Axon.ModelState{data: data, state: model_state} = ms) do
    %{ms | data: deep_binary_copy(data), state: deep_binary_copy(model_state)}
  end

  defp deep_binary_copy(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_binary_copy(v)} end)
  end

  defp deep_binary_copy(list) when is_list(list) do
    Enum.map(list, &deep_binary_copy/1)
  end

  defp deep_binary_copy(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&deep_binary_copy/1)
    |> List.to_tuple()
  end

  defp deep_binary_copy(other), do: other
end
