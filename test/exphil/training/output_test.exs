defmodule ExPhil.Training.OutputTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Output

  describe "colorize/2" do
    test "applies red color" do
      result = Output.colorize("error", :red)
      assert result =~ "error"
      # Red ANSI code
      assert result =~ "\e[31m"
      # Reset code
      assert result =~ "\e[0m"
    end

    test "applies green color" do
      result = Output.colorize("success", :green)
      assert result =~ "\e[32m"
    end

    test "applies bold" do
      result = Output.colorize("title", :bold)
      assert result =~ "\e[1m"
    end

    test "returns plain text for unknown color" do
      result = Output.colorize("text", :unknown)
      assert result == "text\e[0m"
    end
  end

  describe "format_duration/1" do
    test "formats seconds only" do
      assert Output.format_duration(5_000) == "5s"
      assert Output.format_duration(45_000) == "45s"
    end

    test "formats minutes and seconds" do
      assert Output.format_duration(65_000) == "1m 5s"
      assert Output.format_duration(3_600_000 - 1000) == "59m 59s"
    end

    test "formats hours, minutes and seconds" do
      assert Output.format_duration(3_600_000) == "1h 0m 0s"
      assert Output.format_duration(3_661_000) == "1h 1m 1s"
      assert Output.format_duration(7_261_000) == "2h 1m 1s"
    end
  end

  describe "format_bytes/1" do
    test "formats bytes" do
      assert Output.format_bytes(500) == "500 B"
    end

    test "formats kilobytes" do
      assert Output.format_bytes(1024) == "1.0 KB"
      assert Output.format_bytes(2048) == "2.0 KB"
    end

    test "formats megabytes" do
      assert Output.format_bytes(1_048_576) == "1.0 MB"
      assert Output.format_bytes(10_485_760) == "10.0 MB"
    end

    test "formats gigabytes" do
      assert Output.format_bytes(1_073_741_824) == "1.0 GB"
    end
  end

  describe "terminal_loss_graph/2" do
    import ExUnit.CaptureIO

    test "handles empty history" do
      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph([])
        end)

      assert output =~ "No training data"
    end

    test "renders graph for single epoch" do
      history = [%{epoch: 1, train_loss: 0.5}]

      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph(history, width: 40, height: 8)
        end)

      # Should contain border elements
      assert output =~ "┌"
      assert output =~ "┘"
      # Train point
      assert output =~ "●"
    end

    test "renders graph with multiple epochs" do
      history = [
        %{epoch: 1, train_loss: 0.8},
        %{epoch: 2, train_loss: 0.5},
        %{epoch: 3, train_loss: 0.3}
      ]

      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph(history, width: 50, height: 10)
        end)

      # Should show title and legend
      assert output =~ "Loss"
      assert output =~ "train"
      assert output =~ "●"
    end

    test "renders validation loss when present" do
      history = [
        %{epoch: 1, train_loss: 0.8, val_loss: 0.9},
        %{epoch: 2, train_loss: 0.5, val_loss: 0.6}
      ]

      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph(history)
        end)

      assert output =~ "val"
      # Val point
      assert output =~ "○"
    end

    test "respects show_val option" do
      history = [
        %{epoch: 1, train_loss: 0.8, val_loss: 0.9}
      ]

      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph(history, show_val: false)
        end)

      refute output =~ "val"
    end

    test "uses custom title" do
      history = [%{epoch: 1, train_loss: 0.5}]

      output =
        capture_io(:stderr, fn ->
          Output.terminal_loss_graph(history, title: "Custom Title")
        end)

      assert output =~ "Custom Title"
    end
  end

  describe "loss_sparkline/2" do
    import ExUnit.CaptureIO

    test "handles empty losses" do
      output =
        capture_io(:stderr, fn ->
          Output.loss_sparkline([])
        end)

      assert output =~ "no data"
    end

    test "renders sparkline for loss values" do
      losses = [0.8, 0.6, 0.4, 0.3, 0.2]

      output =
        capture_io(:stderr, fn ->
          Output.loss_sparkline(losses)
        end)

      # Should contain sparkline chars and current value
      assert output =~ "Loss:"
      assert output =~ "0.2"
      # Should contain some sparkline characters
      assert output =~ "▁" or output =~ "▂" or output =~ "▃"
    end

    test "uses custom label" do
      losses = [0.5, 0.3]

      output =
        capture_io(:stderr, fn ->
          Output.loss_sparkline(losses, label: "Val Loss")
        end)

      assert output =~ "Val Loss:"
    end

    test "handles constant values" do
      losses = [0.5, 0.5, 0.5]

      output =
        capture_io(:stderr, fn ->
          Output.loss_sparkline(losses)
        end)

      # Should still render something
      assert output =~ "Loss:"
      assert output =~ "0.5"
    end
  end

  describe "verbosity" do
    test "set_verbosity/1 sets verbosity level" do
      Output.set_verbosity(0)
      assert Output.get_verbosity() == 0

      Output.set_verbosity(2)
      assert Output.get_verbosity() == 2

      # Reset to normal for other tests
      Output.set_verbosity(1)
    end

    test "set_verbosity/1 accepts atoms" do
      Output.set_verbosity(:quiet)
      assert Output.get_verbosity() == 0

      Output.set_verbosity(:normal)
      assert Output.get_verbosity() == 1

      Output.set_verbosity(:verbose)
      assert Output.get_verbosity() == 2

      # Reset
      Output.set_verbosity(1)
    end

    test "quiet?/0 returns true when verbosity is 0" do
      Output.set_verbosity(0)
      assert Output.quiet?() == true

      Output.set_verbosity(1)
      assert Output.quiet?() == false

      # Reset
      Output.set_verbosity(1)
    end

    test "verbose?/0 returns true when verbosity is 2" do
      Output.set_verbosity(2)
      assert Output.verbose?() == true

      Output.set_verbosity(1)
      assert Output.verbose?() == false

      # Reset
      Output.set_verbosity(1)
    end

    test "should_output?/1 respects verbosity level" do
      Output.set_verbosity(0)
      assert Output.should_output?(0) == true
      assert Output.should_output?(1) == false
      assert Output.should_output?(2) == false

      Output.set_verbosity(1)
      assert Output.should_output?(0) == true
      assert Output.should_output?(1) == true
      assert Output.should_output?(2) == false

      Output.set_verbosity(2)
      assert Output.should_output?(0) == true
      assert Output.should_output?(1) == true
      assert Output.should_output?(2) == true

      # Reset
      Output.set_verbosity(1)
    end
  end

  describe "verbosity-aware output" do
    import ExUnit.CaptureIO

    test "puts/1 suppressed in quiet mode" do
      Output.set_verbosity(0)

      output =
        capture_io(:stderr, fn ->
          Output.puts("normal message")
        end)

      assert output == ""

      # Reset
      Output.set_verbosity(1)
    end

    test "puts/1 shown in normal mode" do
      Output.set_verbosity(1)

      output =
        capture_io(:stderr, fn ->
          Output.puts("normal message")
        end)

      assert output =~ "normal message"

      # Reset
      Output.set_verbosity(1)
    end

    test "debug/1 only shown in verbose mode" do
      Output.set_verbosity(1)

      output =
        capture_io(:stderr, fn ->
          Output.debug("debug message")
        end)

      assert output == ""

      Output.set_verbosity(2)

      output =
        capture_io(:stderr, fn ->
          Output.debug("debug message")
        end)

      assert output =~ "debug message"
      assert output =~ "[DEBUG]"

      # Reset
      Output.set_verbosity(1)
    end

    test "error/1 always shown even in quiet mode" do
      Output.set_verbosity(0)

      output =
        capture_io(:stderr, fn ->
          Output.error("error message")
        end)

      assert output =~ "error message"

      # Reset
      Output.set_verbosity(1)
    end

    test "warning/1 suppressed in quiet mode (verbosity 0)" do
      Output.set_verbosity(0)

      output =
        capture_io(:stderr, fn ->
          Output.warning("warning message")
        end)

      # Warnings ARE suppressed in quiet mode for cleaner log output
      refute output =~ "warning message"

      # But shown in normal mode
      Output.set_verbosity(1)

      output =
        capture_io(:stderr, fn ->
          Output.warning("warning message")
        end)

      assert output =~ "warning message"

      # Reset
      Output.set_verbosity(1)
    end
  end

  describe "warmup_start/1" do
    import ExUnit.CaptureIO

    test "shows default JIT message" do
      output =
        capture_io(:stderr, fn ->
          Output.warmup_start()
        end)

      assert output =~ "JIT compiling model"
      assert output =~ "first batch"
      assert output =~ "2-5 minutes"
      assert output =~ "subsequent batches will be fast"
    end

    test "accepts custom expected time" do
      output =
        capture_io(:stderr, fn ->
          Output.warmup_start(expected_time: "30-60 seconds")
        end)

      assert output =~ "30-60 seconds"
    end

    test "accepts custom operation name" do
      output =
        capture_io(:stderr, fn ->
          Output.warmup_start(operation: "policy network")
        end)

      assert output =~ "JIT compiling policy network"
    end
  end

  describe "warmup_done/1" do
    import ExUnit.CaptureIO

    test "shows completion with time in seconds" do
      output =
        capture_io(:stderr, fn ->
          Output.warmup_done(45_000)
        end)

      assert output =~ "JIT compilation complete"
      assert output =~ "45.0s"
    end

    test "rounds to one decimal place" do
      output =
        capture_io(:stderr, fn ->
          Output.warmup_done(123_456)
        end)

      assert output =~ "123.5s"
    end
  end

  describe "with_warmup/2" do
    import ExUnit.CaptureIO

    test "wraps function with warmup indicators" do
      output =
        capture_io(:stderr, fn ->
          result =
            Output.with_warmup(fn ->
              Process.sleep(10)
              :done
            end)

          assert result == :done
        end)

      assert output =~ "JIT compiling"
      assert output =~ "JIT compilation complete"
    end

    test "accepts options" do
      output =
        capture_io(:stderr, fn ->
          Output.with_warmup([operation: "custom"], fn -> :ok end)
        end)

      assert output =~ "JIT compiling custom"
    end
  end
end
