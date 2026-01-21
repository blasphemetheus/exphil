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
end
