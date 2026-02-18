defmodule ExPhil.Training.Imitation.Validation do
  @moduledoc """
  Validation and evaluation functions for imitation learning.

  Provides functions to evaluate model performance on validation datasets
  without computing gradients (inference-only mode).

  ## Key Features

  - **Parallel validation** - Process multiple batches concurrently for better GPU utilization
  - **Progress tracking** - Visual feedback during long validation runs
  - **Cached JIT functions** - Uses pre-compiled eval_loss_fn from trainer for efficiency

  ## See Also

  - `ExPhil.Training.Imitation` - Main imitation learning module
  - `ExPhil.Training.Imitation.Loss` - Loss function builders
  """

  alias ExPhil.Training.Imitation.Loss

  @doc """
  Evaluate on a validation dataset.

  Computes average loss across all batches in the dataset. Supports both
  sequential and parallel evaluation modes.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `dataset` - Enumerable of batch maps with `:states` and `:actions` keys
  - `opts` - Options (see below)

  ## Options

  - `:show_progress` - Show progress bar during evaluation (default: true)
  - `:progress_interval` - Update progress every N batches (default: 10)
  - `:max_concurrency` - Number of concurrent batch evaluations (default: 4)
    - Set to 1 for sequential mode (original behavior)
    - Higher values improve GPU utilization through batched kernel launches

  ## Returns

  Map with:
  - `:loss` - Average loss across all batches
  - `:num_batches` - Number of batches evaluated
  """
  @spec evaluate(struct(), Enumerable.t(), keyword()) :: map()
  def evaluate(trainer, dataset, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, true)
    progress_interval = Keyword.get(opts, :progress_interval, 10)
    max_concurrency = Keyword.get(opts, :max_concurrency, 4)
    debug_jit = System.get_env("EXPHIL_DEBUG_JIT") == "1"

    # Use cached eval_loss_fn if available (JIT-compiled once in new/1)
    # Falls back to building fresh for backwards compatibility
    loss_fn = build_eval_loss_fn_wrapper(trainer)

    # Try to get total count for progress bar (works for lists, not all enumerables)
    total_batches =
      case Enumerable.count(dataset) do
        {:ok, count} -> count
        {:error, _} -> nil
      end

    if show_progress and total_batches && total_batches > 0 do
      IO.write(:stderr, "    Validating: 0/#{total_batches} batches...\e[K")
    end

    # Refresh JIT cache before parallel execution
    # Training steps may have evicted eval_loss_fn from XLA cache
    # Running one batch in main process ensures function is compiled and cached
    dataset_list = if is_list(dataset), do: dataset, else: Enum.to_list(dataset)

    case dataset_list do
      [%{states: states, actions: actions} | _] ->
        # Force computation with backend_transfer to ensure JIT actually runs
        # (Nx tensors are lazy - just calling loss_fn doesn't execute)
        {warmup_us, _} = :timer.tc(fn ->
          _ = loss_fn.(states, actions) |> Nx.backend_transfer()
        end)

        if debug_jit do
          IO.write(:stderr, "\n    [DEBUG] Inline warmup: #{Float.round(warmup_us / 1000, 1)}ms\n")
        end

      _ ->
        :ok
    end

    # Choose between parallel and sequential validation
    {total_loss, count} =
      if max_concurrency > 1 do
        evaluate_parallel(dataset_list, loss_fn, total_batches, show_progress, progress_interval, max_concurrency, debug_jit)
      else
        evaluate_sequential(dataset_list, loss_fn, total_batches, show_progress, progress_interval)
      end

    # Clear progress line
    if show_progress and total_batches && total_batches > 0 do
      IO.write(:stderr, "\r\e[K")
    end

    avg_loss =
      if count > 0 do
        Nx.to_number(total_loss) / count
      else
        0.0
      end

    %{
      loss: avg_loss,
      num_batches: count
    }
  end

  @doc """
  Evaluate on a single batch.

  Returns loss as a tensor (not number) for efficiency.
  Caller should accumulate tensors and convert to number once at epoch end.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `batch` - Map with `:states` and `:actions` keys

  ## Returns

  Map with `:loss` key containing the loss tensor.
  """
  @spec evaluate_batch(struct(), map()) :: %{loss: Nx.Tensor.t()}
  def evaluate_batch(trainer, batch) do
    %{states: states, actions: actions} = batch

    # Use cached eval_loss_fn if available
    loss =
      if trainer.eval_loss_fn do
        trainer.eval_loss_fn.(trainer.policy_params, states, actions)
      else
        # Fallback for backwards compatibility
        label_smoothing = trainer.config[:label_smoothing] || 0.0
        {_predict_fn, loss_fn} = Loss.build_loss_fn(trainer.policy_model, label_smoothing: label_smoothing)
        loss_fn.(trainer.policy_params, states, actions)
      end

    %{loss: loss}
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Build wrapper function that handles cached vs fresh loss function
  defp build_eval_loss_fn_wrapper(trainer) do
    if trainer.eval_loss_fn do
      # Cached JIT function: takes (params, states, actions)
      fn states, actions -> trainer.eval_loss_fn.(trainer.policy_params, states, actions) end
    else
      # Fallback: build fresh (will JIT on first use)
      label_smoothing = trainer.config[:label_smoothing] || 0.0
      {_predict_fn, built_loss_fn} = Loss.build_loss_fn(trainer.policy_model, label_smoothing: label_smoothing)
      fn states, actions -> built_loss_fn.(trainer.policy_params, states, actions) end
    end
  end

  # Parallel validation with Task.async_stream
  defp evaluate_parallel(dataset_list, loss_fn, total_batches, show_progress, progress_interval, max_concurrency, debug_jit) do
    # Counter for progress tracking (use Agent for thread-safe updates)
    {:ok, counter} = Agent.start_link(fn -> 0 end)

    {stream_us, losses} = :timer.tc(fn ->
      dataset_list
      |> Task.async_stream(
        fn batch ->
          %{states: states, actions: actions} = batch
          loss = loss_fn.(states, actions)

          # Update progress counter
          new_count = Agent.get_and_update(counter, fn c -> {c + 1, c + 1} end)

          if show_progress and is_integer(total_batches) and total_batches > 0 and rem(new_count, progress_interval) == 0 do
            pct = round(new_count / total_batches * 100)
            IO.write(:stderr, "\r    Validating: #{new_count}/#{total_batches} batches (#{pct}%)...\e[K")
          end

          # Return scalar loss to avoid GPU memory accumulation
          Nx.to_number(loss)
        end,
        max_concurrency: max_concurrency,
        ordered: false,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, loss} -> loss end)
    end)

    if debug_jit do
      IO.write(:stderr, "    [DEBUG] Task.async_stream total: #{Float.round(stream_us / 1000, 1)}ms\n")
    end

    Agent.stop(counter)

    # Sum losses on CPU (already converted to numbers)
    total = Enum.sum(losses)
    {Nx.tensor(total), length(losses)}
  end

  # Sequential validation (original behavior)
  defp evaluate_sequential(dataset_list, loss_fn, total_batches, show_progress, progress_interval) do
    # Accumulate loss with running sum tensor (avoids Nx.stack overhead)
    Enum.reduce(dataset_list, {Nx.tensor(0.0), 0}, fn batch, {acc_loss, acc_count} ->
      %{states: states, actions: actions} = batch
      loss = loss_fn.(states, actions)
      new_count = acc_count + 1

      # Show progress
      if show_progress and total_batches && total_batches > 0 and rem(new_count, progress_interval) == 0 do
        pct = round(new_count / total_batches * 100)
        IO.write(:stderr, "\r    Validating: #{new_count}/#{total_batches} batches (#{pct}%)...\e[K")
      end

      # Running sum on GPU (no intermediate list allocation)
      {Nx.add(acc_loss, loss), new_count}
    end)
  end
end
