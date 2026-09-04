defmodule ExPhil.Training.Callbacks.ProgressBar do
  @moduledoc """
  Live-updating progress bar with colored loss indicator.

  Shows epoch progress, loss, throughput, and ETA. Bar color reflects
  loss trend: green=improving, yellow=flat, red=worsening.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Output, GPUUtils}

  # Rolling-rate window: slide the anchor every N batches so s/it and
  # ETA reflect the last ~N-2N batches, not the epoch-cumulative average
  # (which bakes in chunk parse/embed + JIT before step 1 — observed
  # reporting "1.77s/it" while steady-state steps were 30ms, 09-04).
  @rate_window 100

  @impl true
  def init(opts) do
    %{
      log_interval: Keyword.get(opts, :log_interval, 10),
      smoothed_loss: nil,
      epoch_start_ms: nil,
      num_batches: 0,
      time_estimate_shown: false,
      first_batch_ms: nil,
      # {batch_idx, ms} anchors: rate is measured from prev_anchor (or
      # cur_anchor before the first slide). cur_anchor is set at batch 1
      # so batch 0 (JIT compile + first chunk prep) never pollutes it.
      cur_anchor: nil,
      prev_anchor: nil
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
      num_batches: state.pipeline.estimated_batches,
      cur_anchor: nil,
      prev_anchor: nil
    }
    {:cont, state, cb}
  end

  # Windowed batch rate in ms/batch, nil until 2+ post-JIT batches exist.
  defp windowed_rate(cb, batch_idx, now_ms) do
    case {cb.prev_anchor, cb.cur_anchor} do
      {{a_idx, a_ms}, _} when batch_idx > a_idx -> (now_ms - a_ms) / (batch_idx - a_idx)
      {nil, {a_idx, a_ms}} when batch_idx > a_idx -> (now_ms - a_ms) / (batch_idx - a_idx)
      _ -> nil
    end
  end

  defp update_anchors(cb, batch_idx, now_ms) do
    cond do
      # First anchor at batch 1: batch 0 carries JIT + first-chunk prep.
      cb.cur_anchor == nil and batch_idx >= 1 ->
        %{cb | cur_anchor: {batch_idx, now_ms}}

      match?({idx, _} when batch_idx - idx >= @rate_window, cb.cur_anchor) ->
        %{cb | prev_anchor: cb.cur_anchor, cur_anchor: {batch_idx, now_ms}}

      true ->
        cb
    end
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

    now_ms = System.monotonic_time(:millisecond)
    cb = %{cb | smoothed_loss: smoothed}
    cb = update_anchors(cb, batch_idx, now_ms)

    # Show time estimate after a few real batches (windowed rate — the
    # epoch-cumulative average bakes in chunk prep + JIT before step 1)
    cb =
      if batch_idx == 100 and not cb.time_estimate_shown do
        batch_ms = windowed_rate(cb, batch_idx, now_ms) || (now_ms - cb.epoch_start_ms) / (batch_idx + 1)
        total_batches = cb.num_batches * state.epochs
        train_ms = total_batches * batch_ms
        # Validation adds ~30% overhead per epoch
        total_est = trunc(train_ms * 1.3 / 1000)
        hours = div(total_est, 3600)
        mins = div(rem(total_est, 3600), 60)

        est_str = if hours > 0, do: "~#{hours}h #{mins}m", else: "~#{mins}m"
        Output.puts("\n  Estimated training time (batch-rate only, excludes per-chunk parse/embed): #{est_str}")

        %{cb | time_estimate_shown: true, first_batch_ms: batch_ms}
      else
        cb
      end

    # Display progress at interval
    if rem(batch_idx, cb.log_interval) == 0 and is_number(smoothed) do
      elapsed_ms = now_ms - cb.epoch_start_ms
      avg_ms = windowed_rate(cb, batch_idx, now_ms) || elapsed_ms / max(batch_idx + 1, 1)
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
