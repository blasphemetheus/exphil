defmodule ExPhil.Training.Output do
  @moduledoc """
  Training output utilities with colors, progress bars, and formatting.

  All output goes to stderr for immediate display (no buffering issues).

  ## Verbosity Levels

  - `0` (quiet): Errors only, no progress bars
  - `1` (normal): Standard output with progress bars
  - `2` (verbose): Debug info including timing, memory, gradients

  Set verbosity with `set_verbosity/1` or via `--quiet`/`--verbose` CLI flags.
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

  # Verbosity levels
  @quiet 0
  @normal 1
  @verbose 2

  # Default timezone for timestamps (Central Time)
  @default_timezone "America/Chicago"

  @doc """
  Set the output verbosity level.

  - `0` or `:quiet` - Errors only
  - `1` or `:normal` - Standard output (default)
  - `2` or `:verbose` - Debug output

  Stored in process dictionary, inherited by spawned processes.
  """
  @spec set_verbosity(integer() | atom()) :: :ok
  def set_verbosity(level) when is_integer(level) do
    Process.put(:exphil_verbosity, level)
    :ok
  end

  def set_verbosity(:quiet), do: set_verbosity(@quiet)
  def set_verbosity(:normal), do: set_verbosity(@normal)
  def set_verbosity(:verbose), do: set_verbosity(@verbose)

  @doc """
  Get the current verbosity level.
  """
  @spec get_verbosity() :: integer()
  def get_verbosity do
    Process.get(:exphil_verbosity, @normal)
  end

  @doc """
  Check if output at the given level should be shown.
  """
  @spec should_output?(integer()) :: boolean()
  def should_output?(level), do: get_verbosity() >= level

  @doc """
  Check if we're in quiet mode.
  """
  @spec quiet?() :: boolean()
  def quiet?, do: get_verbosity() == @quiet

  @doc """
  Check if we're in verbose mode.
  """
  @spec verbose?() :: boolean()
  def verbose?, do: get_verbosity() >= @verbose

  @doc """
  Set the timezone for timestamps.

  Defaults to "America/Chicago" (Central Time).
  """
  @spec set_timezone(String.t()) :: :ok
  def set_timezone(tz) do
    Process.put(:exphil_timezone, tz)
    :ok
  end

  @doc """
  Get the current timezone for timestamps.
  """
  @spec get_timezone() :: String.t()
  def get_timezone do
    Process.get(:exphil_timezone, @default_timezone)
  end

  @doc """
  Set a log file path. Output will be written to both stderr and the file.
  ANSI codes are stripped when writing to the file.
  """
  def set_log_file(path) do
    File.mkdir_p!(Path.dirname(path))
    Process.put(:exphil_log_file, path)
    :ok
  end

  @doc "Get current log file path, if set."
  def get_log_file, do: Process.get(:exphil_log_file)

  defp maybe_log_to_file(line) do
    if path = get_log_file() do
      # Strip ANSI escape codes for clean log files
      clean = Regex.replace(~r/\e\[[0-9;]*[a-zA-Z]/, line, "")
      File.write!(path, clean <> "\n", [:append])
    end
  end

  @doc """
  Get current timestamp string in the configured timezone.
  """
  @spec local_timestamp() :: String.t()
  def local_timestamp do
    case DateTime.shift_zone(DateTime.utc_now(), get_timezone()) do
      {:ok, local} -> Calendar.strftime(local, "%H:%M:%S")
      {:error, _} -> Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")
    end
  end

  @doc """
  Print a line with timestamp (respects verbosity).
  """
  def puts(line) do
    if should_output?(@normal) do
      timestamp = local_timestamp()
      full = "[#{timestamp}] #{line}"
      IO.puts(:stderr, full)
      maybe_log_to_file(full)
    end
  end

  @doc """
  Print a line without timestamp (respects verbosity).
  """
  def puts_raw(line) do
    if should_output?(@normal) do
      IO.puts(:stderr, line)
      maybe_log_to_file(line)
    end
  end

  @doc """
  Print with color (respects verbosity). Colors: :red, :green, :yellow, :blue, :cyan, :bold, :dim
  """
  def puts(line, color) when is_atom(color) do
    if should_output?(@normal) do
      timestamp = local_timestamp()
      IO.puts(:stderr, "[#{timestamp}] #{colorize(line, color)}")
    end
  end

  @doc """
  Print colored text without timestamp (respects verbosity).
  """
  def puts_raw(line, color) when is_atom(color) do
    if should_output?(@normal) do
      IO.puts(:stderr, colorize(line, color))
    end
  end

  @doc """
  Print a debug message (verbose mode only).
  """
  def debug(line) do
    if verbose?() do
      timestamp = local_timestamp()
      IO.puts(:stderr, "[#{timestamp}] #{colorize("[DEBUG] #{line}", :dim)}")
    end
  end

  @doc """
  Print a success message (green).
  """
  def success(line), do: puts("✓ #{line}", :green)

  @doc """
  Print a warning message (yellow). Suppressed in quiet mode (--quiet).
  """
  def warning(line) do
    # Warnings respect quiet mode (verbosity 0)
    if should_output?(@normal) do
      timestamp = local_timestamp()
      IO.puts(:stderr, "[#{timestamp}] #{colorize("⚠️  #{line}", :yellow)}")
    end
  end

  @doc """
  Print an error message (red). Always shown (even in quiet mode).
  """
  def error(line) do
    # Errors always show
    timestamp = local_timestamp()
    IO.puts(:stderr, "[#{timestamp}] #{colorize("❌ #{line}", :red)}")
  end

  @doc """
  Print an info message (cyan).
  """
  def info(line), do: puts(line, :cyan)

  @doc """
  Apply color to text.
  """
  def colorize(text, color) do
    color_code =
      case color do
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

    percent = if total > 0, do: min(current / total, 1.0), else: 0
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
    # \e[K clears from cursor to end of line (prevents artifacts in narrow terminals)
    IO.write(:stderr, "\r#{line}\e[K")
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
    IO.write(:stderr, "\r#{colorize(char, :cyan)} #{label}...\e[K")
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

    val_losses =
      if show_val do
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
    # Space for Y axis labels
    y_axis_width = 6
    # -2 for borders
    plot_width = width - y_axis_width - 2
    # -2 for borders
    plot_height = height - 2

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
    grid =
      Enum.reduce(Enum.with_index(train_losses, 1), grid, fn {loss, epoch}, g ->
        x = scale_x.(epoch)
        y = scale_y.(loss)
        y = min(max(y, 0), plot_height - 1)
        row = :array.get(y, g)
        row = :array.set(x, "●", row)
        :array.set(y, row, g)
      end)

    # Plot validation losses
    grid =
      if val_losses != [] do
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

    title_bar =
      "┌" <>
        String.duplicate("─", title_padding) <>
        " #{title} " <>
        String.duplicate("─", width - title_padding - String.length(title) - 4) <> "┐"

    puts_raw(title_bar)

    # Plot rows with Y axis
    for row_idx <- 0..(plot_height - 1) do
      row = :array.get(row_idx, grid)
      row_str = Enum.join(:array.to_list(row))

      # Y axis label (show a few key values)
      y_val = max_y - row_idx / (plot_height - 1) * (max_y - min_y)

      y_label =
        if rem(row_idx, max(div(plot_height, 4), 1)) == 0 do
          :io_lib.format("~5.2f", [y_val]) |> IO.iodata_to_binary()
        else
          "     "
        end

      separator = if rem(row_idx, max(div(plot_height, 4), 1)) == 0, do: "┤", else: "│"
      puts_raw("│#{y_label}#{separator}#{row_str}│")
    end

    # Bottom border with X axis
    puts_raw(
      "└" <>
        String.duplicate("─", y_axis_width) <>
        "┴" <>
        String.duplicate("─", plot_width) <> "┘"
    )

    # X axis epoch labels
    mid_epoch = div(epochs, 2)
    spacing = max(div(plot_width, 3) - 2, 1)

    epoch_info =
      "  Epochs: 1" <>
        String.duplicate(" ", spacing) <>
        "#{mid_epoch}" <>
        String.duplicate(" ", spacing) <>
        "#{epochs}"

    puts_raw(epoch_info)

    # Legend
    legend =
      if val_losses != [] do
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

    sparkline =
      losses
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
  Time a block of code and print status.

  Shows "label..." while running, then "label... done! (Xs)" when complete.

  ## Examples

      iex> Output.timed("Loading data") do
      ...>   # slow operation
      ...> end
      # prints: [12:34:56] Loading data... done! (2.3s)

  """
  defmacro timed(label, do: block) do
    quote do
      IO.write(
        :stderr,
        "[#{ExPhil.Training.Output.local_timestamp()}] #{unquote(label)}..."
      )

      start = System.monotonic_time(:millisecond)
      result = unquote(block)
      elapsed = System.monotonic_time(:millisecond) - start
      IO.puts(:stderr, " done! (#{ExPhil.Training.Output.format_duration(elapsed)})")
      result
    end
  end

  @doc """
  Time a block with a spinner for long operations.

  Shows an animated spinner while running.
  """
  defmacro timed_spinner(label, do: block) do
    quote do
      task = Task.async(fn -> unquote(block) end)

      spinner_task =
        Task.async(fn ->
          Stream.iterate(0, &(&1 + 1))
          |> Enum.reduce_while(nil, fn frame, _ ->
            ExPhil.Training.Output.spinner(frame, unquote(label))
            Process.sleep(100)
            if Task.yield(task, 0), do: {:halt, nil}, else: {:cont, nil}
          end)
        end)

      result = Task.await(task, :infinity)
      Task.shutdown(spinner_task, :brutal_kill)
      ExPhil.Training.Output.clear_line()

      IO.puts(
        :stderr,
        "[#{ExPhil.Training.Output.local_timestamp()}] #{unquote(label)}... done!"
      )

      result
    end
  end

  @doc """
  Print a step indicator for multi-step processes.

  ## Examples

      iex> Output.step(1, 5, "Loading replays")
      # prints: [12:34:56] Step 1/5: Loading replays

  """
  def step(current, total, description) do
    puts("Step #{current}/#{total}: #{description}", :cyan)
  end

  @doc """
  Print a banner for script startup.
  """
  def banner(title, subtitle \\ nil) do
    width = 60
    puts_raw("")
    puts_raw("╔" <> String.duplicate("═", width) <> "╗")
    title_padding = div(width - String.length(title), 2)

    puts_raw(
      "║" <>
        String.duplicate(" ", title_padding) <>
        title <>
        String.duplicate(" ", width - title_padding - String.length(title)) <> "║"
    )

    if subtitle do
      sub_padding = div(width - String.length(subtitle), 2)

      puts_raw(
        "║" <>
          String.duplicate(" ", sub_padding) <>
          colorize(subtitle, :dim) <>
          String.duplicate(" ", width - sub_padding - String.length(subtitle)) <> "║"
      )
    end

    puts_raw("╚" <> String.duplicate("═", width) <> "╝")
    puts_raw("")
  end

  @doc """
  Print configuration as key-value pairs.
  """
  def config(pairs) when is_list(pairs) do
    puts_raw(colorize("Configuration:", :bold))
    Enum.each(pairs, fn {k, v} -> kv("  #{k}", inspect(v)) end)
    puts_raw("")
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

        puts_raw(
          "    #{String.pad_trailing(name, 18)} #{colorize(bar, :green)} #{count} (#{pct}%)"
        )
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

        puts_raw(
          "    #{String.pad_trailing(name, 22)} #{colorize(bar, :cyan)} #{count} (#{pct}%)"
        )
      end)
    end

    divider()
  end

  # ============================================================
  # JIT/Warmup Indicators
  # ============================================================

  @doc """
  Print JIT warmup start indicator.

  Call this before the first inference/training batch to inform users
  about the expected JIT compilation time.

  ## Options

    * `:expected_time` - Expected time string (default: "2-5 minutes")
    * `:operation` - What's being compiled (default: "model")

  ## Examples

      iex> Output.warmup_start()
      # prints: [12:34:56] ⏳ JIT compiling model (first batch)... this may take 2-5 minutes
      #                   (subsequent batches will be fast)

      iex> Output.warmup_start(expected_time: "30-60 seconds", operation: "policy network")
      # prints: [12:34:56] ⏳ JIT compiling policy network (first batch)... this may take 30-60 seconds
      #                   (subsequent batches will be fast)

  """
  def warmup_start(opts \\ []) do
    expected = Keyword.get(opts, :expected_time, "2-5 minutes")
    operation = Keyword.get(opts, :operation, "model")

    puts("⏳ JIT compiling #{operation} (first batch)... this may take #{expected}")
    puts_raw("     (subsequent batches will be fast)")
  end

  @doc """
  Print JIT warmup complete indicator.

  Call this after the first inference/training batch completes to show
  the actual compilation time.

  ## Arguments

    * `elapsed_ms` - Time in milliseconds that JIT compilation took

  ## Examples

      iex> Output.warmup_done(45_000)
      # prints: [12:34:56] ✓ JIT compilation complete (45.0s)

  """
  def warmup_done(elapsed_ms) when is_number(elapsed_ms) do
    seconds = Float.round(elapsed_ms / 1000, 1)
    success("JIT compilation complete (#{seconds}s)")
  end

  @doc """
  Run a function with warmup indicators.

  Automatically shows warmup start message, runs the function,
  and shows completion time.

  ## Options

    * `:expected_time` - Expected time string (default: "2-5 minutes")
    * `:operation` - What's being compiled (default: "model")

  ## Examples

      iex> result = Output.with_warmup(fn ->
      ...>   Imitation.train_step(trainer, batch, nil)
      ...> end)
      # prints warmup_start, runs function, prints warmup_done

  """
  def with_warmup(opts \\ [], fun) when is_function(fun, 0) do
    warmup_start(opts)
    start = System.monotonic_time(:millisecond)
    result = fun.()
    elapsed = System.monotonic_time(:millisecond) - start
    warmup_done(elapsed)
    result
  end

  @doc """
  Run a function with warmup indicator (macro version for blocks).

  ## Examples

      require Output
      result = Output.warmup "policy network" do
        Imitation.train_step(trainer, batch, nil)
      end

  """
  defmacro warmup(operation \\ "model", do: block) do
    quote do
      ExPhil.Training.Output.warmup_start(operation: unquote(operation))
      start = System.monotonic_time(:millisecond)
      result = unquote(block)
      elapsed = System.monotonic_time(:millisecond) - start
      ExPhil.Training.Output.warmup_done(elapsed)
      result
    end
  end

  # ============================================================================
  # Visualization Helpers
  # ============================================================================

  @doc """
  Color-coded comparison bar for predicted vs actual rates.

  Shows side-by-side bars with color indicating how close the prediction is.
  Green = close (within threshold), yellow = off, red = collapse.

  ## Options
    - `:bar_width` - Width of each bar in chars (default: 20)
    - `:threshold` - Delta % for green vs yellow (default: 3.0)
    - `:collapse_threshold` - Actual % below which 0 pred = collapse (default: 1.0)

  ## Examples

      comparison_bar("A", 5.9, 6.3)
      # "    A     ████░░░░░░░░░░░░░░░░  5.9%  vs  ██████░░░░░░░░░░░░░░  6.3%  ✓"
  """
  def comparison_bar(label, pred_pct, actual_pct, opts \\ []) do
    bar_width = Keyword.get(opts, :bar_width, 20)
    threshold = Keyword.get(opts, :threshold, 3.0)
    collapse_threshold = Keyword.get(opts, :collapse_threshold, 1.0)

    delta = abs(pred_pct - actual_pct)

    {color, indicator} =
      cond do
        actual_pct > collapse_threshold and pred_pct < 0.1 -> {@red, " COLLAPSE"}
        delta <= threshold -> {@green, ""}
        delta <= threshold * 2 -> {@yellow, ""}
        pred_pct > actual_pct -> {@yellow, " ↑"}
        true -> {@yellow, " ↓"}
      end

    pred_bar = make_bar(pred_pct, bar_width)
    actual_bar = make_bar(actual_pct, bar_width)
    pred_s = pred_pct |> Float.round(1) |> to_string() |> String.pad_leading(5)
    actual_s = actual_pct |> Float.round(1) |> to_string() |> String.pad_leading(5)
    label_s = String.pad_trailing(to_string(label), 5)

    "    #{color}#{label_s} #{pred_bar} #{pred_s}%  vs  #{actual_bar} #{actual_s}%#{indicator}#{@reset}"
  end

  defp make_bar(pct, width) do
    # Scale: 0-100% maps to 0-width filled chars
    filled = round(pct / 100 * width) |> min(width) |> max(0)
    empty = width - filled
    String.duplicate("█", filled) <> String.duplicate("░", empty)
  end

  @doc """
  Render a summary box with box-drawing characters.

  ## Examples

      summary_box("Epoch 3/30", [
        {"train_loss", "2.545"},
        {"val_loss", "2.581"},
        {"time", "67s"},
        {"GPU", "29.75/31.84 GB (93%)"}
      ])
  """
  def summary_box(title, entries, opts \\ []) do
    highlight = Keyword.get(opts, :highlight, nil)
    min_width = Keyword.get(opts, :min_width, 50)

    lines = Enum.map(entries, fn {k, v} -> "  #{k}  #{v}" end)
    content_width = Enum.map(lines, &String.length/1) |> Enum.max(fn -> 0 end)
    title_width = String.length(title) + 4
    width = max(max(content_width + 2, title_width + 2), min_width)

    top = "╭─── #{title} " <> String.duplicate("─", max(width - String.length(title) - 6, 0)) <> "╮"
    bottom = "╰" <> String.duplicate("─", width) <> "╯"

    body = Enum.map(lines, fn line ->
      padding = width - String.length(line) - 1
      "│#{line}#{String.duplicate(" ", max(padding, 0))}│"
    end)

    output = [top | body] ++ [bottom]

    highlight_line = if highlight, do: "│  #{@green}#{highlight}#{@reset}#{String.duplicate(" ", max(width - String.length(highlight) - 3, 0))}│", else: nil

    if highlight_line do
      [top | body] ++ [highlight_line, bottom]
    else
      output
    end
    |> Enum.join("\n")
  end

  @doc """
  Render a sparkline from a list of values using Unicode block chars.

  ## Examples

      sparkline([3.3, 2.6, 2.5, 2.4, 2.3, 2.2])
      # "▇▃▂▂▁▁"

      sparkline_with_label("Loss", [3.3, 2.6, 2.5])
      # "  Loss: ▇▃▂ (3.3 → 2.5)"
  """
  def sparkline(values) when length(values) < 2, do: ""

  def sparkline(values) do
    min_val = Enum.min(values)
    max_val = Enum.max(values)
    range = max_val - min_val

    blocks = ~w(▁ ▂ ▃ ▄ ▅ ▆ ▇ █)

    if range == 0 do
      String.duplicate("▄", length(values))
    else
      values
      |> Enum.map(fn v ->
        idx = round((v - min_val) / range * 7) |> min(7) |> max(0)
        Enum.at(blocks, idx)
      end)
      |> Enum.join()
    end
  end

  def sparkline_with_label(_label, values) when length(values) < 2, do: ""

  def sparkline_with_label(label, values) do
    spark = sparkline(values)
    first = Float.round(hd(values) * 1.0, 4)
    last = Float.round(List.last(values) * 1.0, 4)
    "  #{label}: #{spark} (#{first} → #{last})"
  end

  @doc """
  Render a formatted table with aligned columns.

  ## Examples

      table(
        ["Head", "Loss", "Delta"],
        [
          ["buttons", "0.285", "-0.02"],
          ["main_x", "1.582", "-0.15"],
        ]
      )
  """
  def table(headers, rows, opts \\ []) do
    indent = Keyword.get(opts, :indent, "    ")
    all_rows = [headers | rows]

    # Calculate column widths
    widths = Enum.reduce(all_rows, List.duplicate(0, length(headers)), fn row, acc ->
      Enum.zip(row, acc)
      |> Enum.map(fn {cell, w} -> max(String.length(to_string(cell)), w) end)
    end)

    # Format header
    header_line = Enum.zip(headers, widths)
      |> Enum.map(fn {h, w} -> String.pad_trailing(to_string(h), w) end)
      |> Enum.join("  ")

    separator = Enum.map(widths, fn w -> String.duplicate("─", w) end) |> Enum.join("──")

    # Format body rows
    body_lines = Enum.map(rows, fn row ->
      Enum.zip(row, widths)
      |> Enum.map(fn {cell, w} -> String.pad_trailing(to_string(cell), w) end)
      |> Enum.join("  ")
    end)

    ([indent <> header_line, indent <> separator] ++ Enum.map(body_lines, &(indent <> &1)))
    |> Enum.join("\n")
  end
end
