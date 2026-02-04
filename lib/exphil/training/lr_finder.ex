defmodule ExPhil.Training.LRFinder do
  @moduledoc """
  Learning Rate Finder for automatically discovering optimal learning rate ranges.

  Implements the LR range test popularized by Leslie Smith and fastai.
  The algorithm:
  1. Save current model weights
  2. Start with a very small learning rate
  3. Train for N steps, exponentially increasing LR each step
  4. Record loss at each step
  5. Restore original weights
  6. Return loss/LR data for analysis

  ## Usage

      # Find optimal LR range
      {:ok, results} = LRFinder.find(model, dataset, min_lr: 1e-7, max_lr: 1.0)

      # Results contain loss/LR pairs for analysis
      suggested_lr = LRFinder.suggest_lr(results)

  ## Interpreting Results

  The optimal learning rate is typically:
  - Where loss is decreasing fastest (steepest negative slope)
  - About 1/10th of the LR where loss starts exploding
  - In the "valley" of the loss curve

  """

  alias ExPhil.Training.Imitation
  alias ExPhil.Error.DataError

  @default_min_lr 1.0e-7
  @default_max_lr 1.0
  @default_num_steps 100
  @default_smooth_factor 0.05

  @type result :: %{
          lr: float(),
          loss: float(),
          smoothed_loss: float(),
          step: non_neg_integer()
        }

  @type find_result :: %{
          history: [result()],
          suggested_lr: float() | nil,
          min_loss_lr: float(),
          min_loss: float()
        }

  @doc """
  Run the learning rate finder on a dataset.

  ## Options

    * `:min_lr` - Starting learning rate (default: 1e-7)
    * `:max_lr` - Ending learning rate (default: 1.0)
    * `:num_steps` - Number of steps to run (default: 100)
    * `:smooth_factor` - Exponential smoothing factor for loss (default: 0.05)
    * `:stop_div` - Stop if loss diverges by this factor (default: 4.0)

  ## Returns

    `{:ok, results}` with history of LR/loss pairs and suggested LR,
    or `{:error, reason}` if something goes wrong.

  """
  @spec find(map(), Enumerable.t(), keyword()) :: {:ok, find_result()} | {:error, term()}
  def find(model_params, dataset, opts \\ []) do
    min_lr = Keyword.get(opts, :min_lr, @default_min_lr)
    max_lr = Keyword.get(opts, :max_lr, @default_max_lr)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)
    smooth_factor = Keyword.get(opts, :smooth_factor, @default_smooth_factor)
    stop_div = Keyword.get(opts, :stop_div, 4.0)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, [64, 64])
    embed_size = Keyword.get(opts, :embed_size, 1991)

    # Calculate LR multiplier for exponential growth
    # lr(step) = min_lr * (max_lr / min_lr)^(step / num_steps)
    lr_mult = :math.pow(max_lr / min_lr, 1.0 / num_steps)

    # Take enough batches from dataset
    batches = dataset |> Enum.take(num_steps)

    if length(batches) < num_steps do
      {:error, DataError.new(:insufficient_data, context: %{required: num_steps, actual: length(batches)})}
    else
      run_finder(
        model_params,
        batches,
        min_lr,
        lr_mult,
        num_steps,
        smooth_factor,
        stop_div,
        hidden_sizes,
        embed_size
      )
    end
  end

  defp run_finder(
         model_params,
         batches,
         min_lr,
         lr_mult,
         _num_steps,
         smooth_factor,
         stop_div,
         hidden_sizes,
         embed_size
       ) do
    # Create initial optimizer with min_lr
    config = %{
      learning_rate: min_lr,
      lr_schedule: :constant,
      warmup_steps: 0,
      weight_decay: 0.01,
      hidden_sizes: hidden_sizes,
      embed_size: embed_size
    }

    {init_fn, update_fn} = Imitation.create_optimizer(config)
    opt_state = init_fn.(model_params)

    # Initialize tracking
    initial_state = %{
      params: model_params,
      opt_state: opt_state,
      history: [],
      best_loss: :infinity,
      smoothed_loss: nil,
      stopped_early: false
    }

    # Run through batches, increasing LR each step
    final_state =
      batches
      |> Enum.with_index()
      |> Enum.reduce_while(initial_state, fn {{states, actions}, step}, state ->
        if state.stopped_early do
          {:halt, state}
        else
          current_lr = min_lr * :math.pow(lr_mult, step)

          # Compute loss and gradients
          {loss, gradients} =
            compute_loss_and_grads(state.params, states, actions, hidden_sizes, embed_size)

          loss_val = Nx.to_number(loss)

          # Update smoothed loss
          smoothed =
            case state.smoothed_loss do
              nil -> loss_val
              prev -> smooth_factor * loss_val + (1 - smooth_factor) * prev
            end

          # Check for divergence
          should_stop = state.best_loss != :infinity and smoothed > stop_div * state.best_loss

          # Update best loss
          new_best = if loss_val < state.best_loss, do: loss_val, else: state.best_loss

          # Record history
          record = %{
            lr: current_lr,
            loss: loss_val,
            smoothed_loss: smoothed,
            step: step
          }

          # Scale gradients by LR ratio if we're changing LR
          # (This is a simplification - proper implementation would recreate optimizer)
          scaled_gradients = scale_gradients(gradients, current_lr / min_lr)

          # Update parameters
          {new_params, new_opt_state} =
            update_fn.(scaled_gradients, state.opt_state, state.params)

          new_state = %{
            state
            | params: new_params,
              opt_state: new_opt_state,
              history: [record | state.history],
              best_loss: new_best,
              smoothed_loss: smoothed,
              stopped_early: should_stop
          }

          if should_stop do
            {:halt, new_state}
          else
            {:cont, new_state}
          end
        end
      end)

    # Analyze results
    history = Enum.reverse(final_state.history)
    suggested = suggest_lr(history)

    min_loss_record = Enum.min_by(history, & &1.loss)

    {:ok,
     %{
       history: history,
       suggested_lr: suggested,
       min_loss_lr: min_loss_record.lr,
       min_loss: min_loss_record.loss,
       stopped_early: final_state.stopped_early
     }}
  end

  # Compute loss and gradients for a batch
  defp compute_loss_and_grads(params, states, actions, hidden_sizes, embed_size) do
    loss_fn = fn p ->
      # Simple forward pass through policy network
      logits = forward_policy(p, states, hidden_sizes, embed_size)
      compute_cross_entropy_loss(logits, actions)
    end

    Nx.Defn.value_and_grad(loss_fn).(params)
  end

  # Simple policy forward pass for LR finding
  # Uses a basic MLP structure
  defp forward_policy(params, states, hidden_sizes, _embed_size) do
    # Flatten states if needed
    x =
      case Nx.shape(states) do
        {_batch, _} -> states
        {batch, seq, feat} -> Nx.reshape(states, {batch * seq, feat})
        _ -> states
      end

    # Apply MLP layers with ReLU activation
    Enum.reduce(0..(length(hidden_sizes) - 1), x, fn i, acc ->
      w_key = "dense_#{i}_kernel"
      b_key = "dense_#{i}_bias"

      w = params[w_key] || params[String.to_atom(w_key)]
      b = params[b_key] || params[String.to_atom(b_key)]

      if w && b do
        acc
        |> Nx.dot(w)
        |> Nx.add(b)
        # ReLU
        |> Nx.max(0)
      else
        acc
      end
    end)
  end

  # Compute cross-entropy loss
  defp compute_cross_entropy_loss(logits, targets) do
    # Simplified cross-entropy for multi-head output
    # Just compute mean squared error as a proxy for loss direction
    case {Nx.shape(logits), Nx.shape(targets)} do
      {{n, d}, {n, d}} ->
        Nx.mean(Nx.pow(Nx.subtract(logits, targets), 2))

      _ ->
        # Fallback to simple loss
        Nx.mean(Nx.abs(logits))
    end
  end

  # Scale gradients by a factor
  defp scale_gradients(gradients, factor) when is_map(gradients) do
    Map.new(gradients, fn {k, v} ->
      {k, scale_gradients(v, factor)}
    end)
  end

  defp scale_gradients(%Nx.Tensor{} = tensor, factor) do
    Nx.multiply(tensor, factor)
  end

  defp scale_gradients(other, _factor), do: other

  @doc """
  Suggest an optimal learning rate from finder results.

  Uses the steepest descent heuristic: finds the LR where loss
  is decreasing fastest, then returns a slightly lower LR for safety.

  ## Options

    * `:skip_start` - Skip first N% of results (default: 0.1)
    * `:skip_end` - Skip last N% of results (default: 0.2)

  """
  @spec suggest_lr([result()], keyword()) :: float() | nil
  def suggest_lr(history, opts \\ [])

  def suggest_lr([], _opts), do: nil

  def suggest_lr(history, _opts) when length(history) < 5, do: nil

  def suggest_lr(history, opts) do
    skip_start = Keyword.get(opts, :skip_start, 0.1)
    skip_end = Keyword.get(opts, :skip_end, 0.2)

    n = length(history)
    start_idx = round(n * skip_start)
    end_idx = round(n * (1 - skip_end))

    # Get the middle portion of results
    middle = history |> Enum.slice(start_idx, end_idx - start_idx)

    if length(middle) < 3 do
      # Fallback to minimum loss LR / 10
      min_record = Enum.min_by(history, & &1.smoothed_loss)
      min_record.lr / 10
    else
      # Find steepest descent (most negative gradient)
      gradients =
        middle
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] ->
          # Use log scale for LR
          lr_diff = :math.log10(b.lr) - :math.log10(a.lr)
          loss_diff = b.smoothed_loss - a.smoothed_loss
          gradient = loss_diff / max(lr_diff, 1.0e-10)
          {a.lr, gradient}
        end)

      # Find most negative gradient (steepest descent)
      {best_lr, _grad} = Enum.min_by(gradients, fn {_lr, g} -> g end)

      # Return this LR (could also divide by 10 for extra safety)
      best_lr
    end
  end

  @doc """
  Format LR finder results for display.

  Returns a string showing the loss curve that can be printed to console.
  """
  @spec format_results(find_result()) :: String.t()
  def format_results(%{
        history: history,
        suggested_lr: suggested,
        min_loss_lr: min_lr,
        min_loss: min_loss
      }) do
    """
    === Learning Rate Finder Results ===

    Suggested LR: #{format_lr(suggested)}
    Min Loss LR:  #{format_lr(min_lr)} (loss: #{Float.round(min_loss, 4)})

    LR Range Tested: #{format_lr(List.first(history).lr)} -> #{format_lr(List.last(history).lr)}
    Steps: #{length(history)}

    Loss Curve (smoothed):
    #{format_loss_curve(history)}
    """
  end

  defp format_lr(nil), do: "N/A"
  defp format_lr(lr) when lr < 1.0e-4, do: :io_lib.format("~.2e", [lr]) |> to_string()
  defp format_lr(lr), do: Float.round(lr, 6) |> to_string()

  defp format_loss_curve(history) do
    # Sample ~20 points for display
    n = length(history)
    step = max(1, div(n, 20))

    sampled = history |> Enum.take_every(step) |> Enum.take(20)

    max_loss = sampled |> Enum.map(& &1.smoothed_loss) |> Enum.max()
    min_loss = sampled |> Enum.map(& &1.smoothed_loss) |> Enum.min()
    range = max(max_loss - min_loss, 0.001)

    sampled
    |> Enum.map(fn %{lr: lr, smoothed_loss: loss} ->
      bar_len = round((loss - min_loss) / range * 30)
      bar = String.duplicate("█", bar_len) <> String.duplicate("░", 30 - bar_len)
      "  #{format_lr(lr) |> String.pad_leading(10)} │#{bar}│ #{Float.round(loss, 4)}"
    end)
    |> Enum.join("\n")
  end
end
