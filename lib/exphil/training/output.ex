defmodule ExPhil.Training.Output do
  @moduledoc """
  Training output utilities with colors, progress bars, and formatting.

  All output goes to stderr for immediate display (no buffering issues).
  """

  # ANSI color codes
  @reset "\e[0m"
  @bold "\e[1m"
  @dim "\e[2m"
  @red "\e[31m"
  @green "\e[32m"
  @yellow "\e[33m"
  @blue "\e[34m"
  @cyan "\e[36m"

  @doc """
  Print a line with timestamp.
  """
  def puts(line) do
    timestamp = DateTime.utc_now() |> Calendar.strftime("%H:%M:%S")
    IO.puts(:stderr, "[#{timestamp}] #{line}")
  end

  @doc """
  Print a line without timestamp.
  """
  def puts_raw(line), do: IO.puts(:stderr, line)

  @doc """
  Print with color. Colors: :red, :green, :yellow, :blue, :cyan, :bold, :dim
  """
  def puts(line, color) when is_atom(color) do
    timestamp = DateTime.utc_now() |> Calendar.strftime("%H:%M:%S")
    IO.puts(:stderr, "[#{timestamp}] #{colorize(line, color)}")
  end

  @doc """
  Print colored text without timestamp.
  """
  def puts_raw(line, color) when is_atom(color) do
    IO.puts(:stderr, colorize(line, color))
  end

  @doc """
  Print a success message (green).
  """
  def success(line), do: puts(line, :green)

  @doc """
  Print a warning message (yellow).
  """
  def warning(line), do: puts("⚠️  #{line}", :yellow)

  @doc """
  Print an error message (red).
  """
  def error(line), do: puts("❌ #{line}", :red)

  @doc """
  Print an info message (cyan).
  """
  def info(line), do: puts(line, :cyan)

  @doc """
  Apply color to text.
  """
  def colorize(text, color) do
    color_code = case color do
      :red -> @red
      :green -> @green
      :yellow -> @yellow
      :blue -> @blue
      :cyan -> @cyan
      :bold -> @bold
      :dim -> @dim
      _ -> ""
    end

    "#{color_code}#{text}#{@reset}"
  end

  @doc """
  Print a progress bar.

  ## Options
  - `:width` - Total width of the bar (default: 30)
  - `:label` - Label to show before the bar
  - `:show_percent` - Show percentage (default: true)
  - `:show_count` - Show current/total (default: true)
  - `:color` - Color of the filled portion (default: :green)

  ## Examples

      iex> Output.progress_bar(3, 10, label: "Epoch")
      # prints: Epoch [████████░░░░░░░░░░░░░░░░░░░░░░] 30% (3/10)

  """
  def progress_bar(current, total, opts \\ []) do
    width = Keyword.get(opts, :width, 30)
    label = Keyword.get(opts, :label, "")
    show_percent = Keyword.get(opts, :show_percent, true)
    show_count = Keyword.get(opts, :show_count, true)
    color = Keyword.get(opts, :color, :green)

    percent = if total > 0, do: current / total, else: 0
    filled = round(percent * width)
    empty = width - filled

    bar_filled = String.duplicate("█", filled)
    bar_empty = String.duplicate("░", empty)

    bar = colorize(bar_filled, color) <> @dim <> bar_empty <> @reset

    parts = [
      if(label != "", do: "#{label} ", else: ""),
      "[#{bar}]",
      if(show_percent, do: " #{round(percent * 100)}%", else: ""),
      if(show_count, do: " (#{current}/#{total})", else: "")
    ]

    line = Enum.join(parts)

    # Use carriage return to overwrite the line
    IO.write(:stderr, "\r#{line}")
  end

  @doc """
  Finish a progress bar by printing a newline.
  """
  def progress_done do
    IO.puts(:stderr, "")
  end

  @doc """
  Print a section header.
  """
  def section(title) do
    puts_raw("")
    puts_raw(colorize("═══ #{title} ═══", :bold))
  end

  @doc """
  Print a divider line.
  """
  def divider do
    puts_raw(@dim <> String.duplicate("─", 50) <> @reset)
  end

  @doc """
  Print a key-value pair.
  """
  def kv(key, value) do
    puts_raw("  #{colorize(key <> ":", :dim)} #{value}")
  end

  @doc """
  Print training summary statistics.
  """
  def training_summary(stats) do
    section("Training Complete")

    kv("Total time", format_duration(stats[:total_time_ms] || 0))
    kv("Epochs", "#{stats[:epochs_completed] || 0}/#{stats[:epochs_total] || 0}")
    kv("Steps", "#{stats[:total_steps] || 0}")

    if stats[:final_loss] do
      kv("Final loss", Float.round(stats[:final_loss], 4))
    end

    if stats[:best_loss] do
      kv("Best loss", "#{Float.round(stats[:best_loss], 4)} (epoch #{stats[:best_epoch] || "?"})")
    end

    if stats[:checkpoint_path] do
      kv("Checkpoint", stats[:checkpoint_path])
    end

    divider()
  end

  @doc """
  Format a duration in milliseconds to human readable string.
  """
  def format_duration(ms) when is_number(ms) do
    seconds = div(trunc(ms), 1000)
    minutes = div(seconds, 60)
    hours = div(minutes, 60)

    cond do
      hours > 0 -> "#{hours}h #{rem(minutes, 60)}m #{rem(seconds, 60)}s"
      minutes > 0 -> "#{minutes}m #{rem(seconds, 60)}s"
      true -> "#{seconds}s"
    end
  end

  @doc """
  Format bytes to human readable string.
  """
  def format_bytes(bytes) when is_number(bytes) do
    cond do
      bytes >= 1_073_741_824 -> "#{Float.round(bytes / 1_073_741_824, 1)} GB"
      bytes >= 1_048_576 -> "#{Float.round(bytes / 1_048_576, 1)} MB"
      bytes >= 1024 -> "#{Float.round(bytes / 1024, 1)} KB"
      true -> "#{bytes} B"
    end
  end

  @doc """
  Print a spinner for indeterminate progress.
  Call repeatedly with increasing frame number.
  """
  def spinner(frame, label \\ "Loading") do
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    char = Enum.at(frames, rem(frame, length(frames)))
    IO.write(:stderr, "\r#{colorize(char, :cyan)} #{label}...")
  end

  @doc """
  Clear the current line.
  """
  def clear_line do
    IO.write(:stderr, "\r\e[K")
  end

  @doc """
  Render a terminal-based loss graph.

  Draws an ASCII chart showing loss over epochs, suitable for in-terminal
  visualization during training.

  ## Options
  - `:width` - Width in characters (default: 60)
  - `:height` - Height in lines (default: 12)
  - `:title` - Chart title (default: "Loss")
  - `:show_val` - Show validation loss if present (default: true)

  ## Examples

      iex> history = [
      ...>   %{epoch: 1, train_loss: 0.8, val_loss: 0.9},
      ...>   %{epoch: 2, train_loss: 0.5, val_loss: 0.6},
      ...>   %{epoch: 3, train_loss: 0.3, val_loss: 0.35}
      ...> ]
      iex> Output.terminal_loss_graph(history)
      # prints:
      # ┌─────────────────────── Loss ───────────────────────┐
      # │ 0.90 ┤●                                            │
      # │      │ ○                                           │
      # │ 0.70 ┤  ●                                          │
      # │      │   ○                                         │
      # │ 0.50 ┤    ●                                        │
      # │      │                                             │
      # │ 0.30 ┤     ○ ●                                     │
      # └──────┴─────────────────────────────────────────────┘
      #         1  2  3
      # ● train  ○ val
  """
  def terminal_loss_graph(history, opts \\ [])

  def terminal_loss_graph([], _opts) do
    puts_raw("  No training data to plot")
  end

  def terminal_loss_graph(history, opts) when is_list(history) do
    width = Keyword.get(opts, :width, 60)
    height = Keyword.get(opts, :height, 12)
    title = Keyword.get(opts, :title, "Loss")
    show_val = Keyword.get(opts, :show_val, true)

    # Extract loss values
    train_losses = Enum.map(history, & &1.train_loss)
    val_losses = if show_val do
      history
      |> Enum.map(& &1[:val_loss])
      |> Enum.reject(&is_nil/1)
    else
      []
    end

    all_losses = train_losses ++ val_losses
    min_loss = Enum.min(all_losses)
    max_loss = Enum.max(all_losses)

    # Add padding to range
    range = max(max_loss - min_loss, 0.001)
    min_y = max(0, min_loss - range * 0.1)
    max_y = max_loss + range * 0.1

    epochs = length(history)

    # Calculate dimensions
    y_axis_width = 6  # Space for Y axis labels
    plot_width = width - y_axis_width - 2  # -2 for borders
    plot_height = height - 2  # -2 for borders

    # Scale function
    scale_y = fn loss ->
      plot_height - 1 - round((loss - min_y) / (max_y - min_y) * (plot_height - 1))
    end

    scale_x = fn epoch ->
      round((epoch - 1) / max(epochs - 1, 1) * (plot_width - 1))
    end

    # Initialize plot grid
    grid = for _ <- 0..(plot_height - 1), do: List.duplicate(" ", plot_width)
    grid = :array.from_list(Enum.map(grid, &:array.from_list/1))

    # Plot train losses
    grid = Enum.reduce(Enum.with_index(train_losses, 1), grid, fn {loss, epoch}, g ->
      x = scale_x.(epoch)
      y = scale_y.(loss)
      y = min(max(y, 0), plot_height - 1)
      row = :array.get(y, g)
      row = :array.set(x, "●", row)
      :array.set(y, row, g)
    end)

    # Plot validation losses
    grid = if val_losses != [] do
      val_with_idx = Enum.zip(val_losses, Enum.take(1..epochs, length(val_losses)))
      Enum.reduce(val_with_idx, grid, fn {loss, epoch}, g ->
        x = scale_x.(epoch)
        y = scale_y.(loss)
        y = min(max(y, 0), plot_height - 1)
        row = :array.get(y, g)
        # Don't overwrite train point
        current = :array.get(x, row)
        if current == " " do
          row = :array.set(x, "○", row)
          :array.set(y, row, g)
        else
          g
        end
      end)
    else
      grid
    end

    # Build output
    # Title bar
    title_padding = div(width - String.length(title) - 4, 2)
    title_bar = "┌" <> String.duplicate("─", title_padding) <> " #{title} " <>
                String.duplicate("─", width - title_padding - String.length(title) - 4) <> "┐"
    puts_raw(title_bar)

    # Plot rows with Y axis
    for row_idx <- 0..(plot_height - 1) do
      row = :array.get(row_idx, grid)
      row_str = Enum.join(:array.to_list(row))

      # Y axis label (show a few key values)
      y_val = max_y - (row_idx / (plot_height - 1)) * (max_y - min_y)
      y_label = if rem(row_idx, max(div(plot_height, 4), 1)) == 0 do
        :io_lib.format("~5.2f", [y_val]) |> IO.iodata_to_binary()
      else
        "     "
      end

      separator = if rem(row_idx, max(div(plot_height, 4), 1)) == 0, do: "┤", else: "│"
      puts_raw("│#{y_label}#{separator}#{row_str}│")
    end

    # Bottom border with X axis
    puts_raw("└" <> String.duplicate("─", y_axis_width) <> "┴" <>
             String.duplicate("─", plot_width) <> "┘")

    # X axis epoch labels
    mid_epoch = div(epochs, 2)
    spacing = max(div(plot_width, 3) - 2, 1)
    epoch_info = "  Epochs: 1" <> String.duplicate(" ", spacing) <>
                 "#{mid_epoch}" <> String.duplicate(" ", spacing) <>
                 "#{epochs}"
    puts_raw(epoch_info)

    # Legend
    legend = if val_losses != [] do
      "  #{colorize("●", :green)} train  #{colorize("○", :cyan)} val"
    else
      "  #{colorize("●", :green)} train"
    end
    puts_raw(legend)
  end

  @doc """
  Print a compact inline loss sparkline.

  Useful for showing loss history in a single line during training.

  ## Examples

      iex> losses = [0.8, 0.6, 0.4, 0.3, 0.25]
      iex> Output.loss_sparkline(losses)
      # prints: Loss: ▇▅▃▂▁ (0.25)
  """
  def loss_sparkline(losses, opts \\ [])

  def loss_sparkline([], opts) do
    label = Keyword.get(opts, :label, "Loss")
    puts_raw("  #{label}: (no data)")
  end

  def loss_sparkline(losses, opts) do
    label = Keyword.get(opts, :label, "Loss")

    # Sparkline characters (8 levels)
    chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    min_val = Enum.min(losses)
    max_val = Enum.max(losses)
    range = max(max_val - min_val, 0.001)

    sparkline = losses
    |> Enum.map(fn val ->
      idx = round((val - min_val) / range * 7)
      idx = min(max(idx, 0), 7)
      Enum.at(chars, idx)
    end)
    |> Enum.join()

    current = List.last(losses)
    puts_raw("  #{label}: #{sparkline} (#{Float.round(current, 4)})")
  end

  @doc """
  Print replay statistics (character and stage distribution).
  """
  def replay_stats(stats) do
    section("Replay Statistics")

    if stats[:total] do
      kv("Total replays", "#{stats[:total]}")
    end

    if stats[:characters] && map_size(stats[:characters]) > 0 do
      puts_raw("")
      puts_raw("  " <> colorize("Characters:", :bold))
      stats[:characters]
      |> Enum.sort_by(fn {_name, count} -> -count end)
      |> Enum.take(10)
      |> Enum.each(fn {name, count} ->
        pct = if stats[:total] > 0, do: Float.round(count / stats[:total] * 100, 1), else: 0
        bar_width = min(round(pct / 2), 25)
        bar = String.duplicate("█", bar_width)
        puts_raw("    #{String.pad_trailing(name, 18)} #{colorize(bar, :green)} #{count} (#{pct}%)")
      end)
    end

    if stats[:stages] && map_size(stats[:stages]) > 0 do
      puts_raw("")
      puts_raw("  " <> colorize("Stages:", :bold))
      stats[:stages]
      |> Enum.sort_by(fn {_name, count} -> -count end)
      |> Enum.each(fn {name, count} ->
        pct = if stats[:total] > 0, do: Float.round(count / stats[:total] * 100, 1), else: 0
        bar_width = min(round(pct / 2), 25)
        bar = String.duplicate("█", bar_width)
        puts_raw("    #{String.pad_trailing(name, 22)} #{colorize(bar, :cyan)} #{count} (#{pct}%)")
      end)
    end

    divider()
  end
end
