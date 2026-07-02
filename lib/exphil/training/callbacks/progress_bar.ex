defmodule ExPhil.Training.Callbacks.ProgressBar do
  @moduledoc """
  Live-updating progress bar with colored loss indicator.

  Shows epoch progress, loss, throughput, and ETA. Bar color reflects
  loss trend: green=improving, yellow=flat, red=worsening.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Output, GPUUtils}

  @impl true
  def init(opts) do
    %{
      log_interval: Keyword.get(opts, :log_interval, 10),
      smoothed_loss: nil,
      epoch_start_ms: nil,
      num_batches: 0,
      time_estimate_shown: false,
      first_batch_ms: nil
    }
  end

  @impl true
  def on_epoch_begin(state, cb) do
    gpu_status = GPUUtils.memory_status_string()
    Output.puts("  --- Epoch #{state.epoch}/#{state.epochs} ---")
    Output.puts("  #{gpu_status}")
    Output.puts("  Starting #{state.pipeline.estimated_batches} batches...")

    cb = %{cb |
      epoch_start_ms: System.monotonic_time(:millisecond),
      smoothed_loss: nil,
      num_batches: state.pipeline.estimated_batches
    }
    {:cont, state, cb}
  end

  @impl true
  def on_batch_end(state, cb) do
    loss = state.batch_metrics.loss
    batch_idx = state.batch_idx

    # Update smoothed loss (EMA alpha=0.1)
    smoothed =
      cond do
        not is_number(loss) -> cb.smoothed_loss
        cb.smoothed_loss == nil -> loss
        true -> 0.1 * loss + 0.9 * cb.smoothed_loss
      end

    cb = %{cb | smoothed_loss: smoothed}

    # Show time estimate after a few real batches (skip JIT batch 0)
    cb =
      if batch_idx == 100 and not cb.time_estimate_shown do
        elapsed_ms = System.monotonic_time(:millisecond) - cb.epoch_start_ms
        # Subtract approximate JIT time (batch 0), use remaining for estimate
        batch_ms = elapsed_ms / (batch_idx + 1)
        total_batches = cb.num_batches * state.epochs
        train_ms = total_batches * batch_ms
        # Validation adds ~30% overhead per epoch
        total_est = trunc(train_ms * 1.3 / 1000)
        hours = div(total_est, 3600)
        mins = div(rem(total_est, 3600), 60)

        est_str = if hours > 0, do: "~#{hours}h #{mins}m", else: "~#{mins}m"
        Output.puts("\n  Estimated total training time: #{est_str}")

        %{cb | time_estimate_shown: true, first_batch_ms: batch_ms}
      else
        cb
      end

    # Display progress at interval
    if rem(batch_idx, cb.log_interval) == 0 and is_number(smoothed) do
      elapsed_ms = System.monotonic_time(:millisecond) - cb.epoch_start_ms
      avg_ms = elapsed_ms / max(batch_idx + 1, 1)
      num_batches = max(cb.num_batches, batch_idx + 1)
      pct = min(round((batch_idx + 1) / num_batches * 100), 100)
      remaining = max(num_batches - (batch_idx + 1), 0)
      eta_sec = round(remaining * avg_ms / 1000)
      eta_min = div(eta_sec, 60)
      eta_sec_rem = rem(eta_sec, 60)

      # Colored bar
      bar_width = 20
      filled = min(round(pct / 100 * bar_width), bar_width)
      bar_color =
        cond do
          cb.smoothed_loss == nil -> ""
          loss < smoothed * 0.999 -> "\e[32m"
          loss > smoothed * 1.001 -> "\e[31m"
          true -> "\e[33m"
        end
      bar_reset = if bar_color != "", do: "\e[0m", else: ""
      bar = bar_color <> String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled) <> bar_reset

      time_str = if avg_ms >= 1000, do: "#{Float.round(avg_ms / 1000, 2)}s/it", else: "#{round(avg_ms)}ms/it"
      pct_str = pct |> Integer.to_string() |> String.pad_leading(3)
      loss_str = Float.round(smoothed, 4)

      line = "  Epoch #{state.epoch}: #{bar} #{pct_str}% | #{batch_idx + 1}/#{num_batches} | loss: #{loss_str} | #{time_str} | ETA: #{eta_min}m #{eta_sec_rem}s"

      # Truncate to terminal width
      width = case :io.columns() do
        {:ok, cols} -> cols
        _ -> 120
      end
      line = if String.length(line) > width - 1, do: String.slice(line, 0, width - 4) <> "...", else: line
      IO.write(:stderr, "\r#{line}\e[K")
    end

    {:cont, state, cb}
  end

  @impl true
  def on_epoch_end(state, cb) do
    # Clear progress line, show epoch summary
    IO.write(:stderr, "\n")
    {:cont, state, cb}
  end
end
