defmodule ExPhil.Training.OutputTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Output

  describe "colorize/2" do
    test "applies red color" do
      result = Output.colorize("error", :red)
      assert result =~ "error"
      assert result =~ "\e[31m"  # Red ANSI code
      assert result =~ "\e[0m"   # Reset code
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
      output = capture_io(:stderr, fn ->
        Output.terminal_loss_graph([])
      end)
      assert output =~ "No training data"
    end

    test "renders graph for single epoch" do
      history = [%{epoch: 1, train_loss: 0.5}]
      output = capture_io(:stderr, fn ->
        Output.terminal_loss_graph(history, width: 40, height: 8)
      end)

      # Should contain border elements
      assert output =~ "┌"
      assert output =~ "┘"
      assert output =~ "●"  # Train point
    end

    test "renders graph with multiple epochs" do
      history = [
        %{epoch: 1, train_loss: 0.8},
        %{epoch: 2, train_loss: 0.5},
        %{epoch: 3, train_loss: 0.3}
      ]

      output = capture_io(:stderr, fn ->
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

      output = capture_io(:stderr, fn ->
        Output.terminal_loss_graph(history)
      end)

      assert output =~ "val"
      assert output =~ "○"  # Val point
    end

    test "respects show_val option" do
      history = [
        %{epoch: 1, train_loss: 0.8, val_loss: 0.9}
      ]

      output = capture_io(:stderr, fn ->
        Output.terminal_loss_graph(history, show_val: false)
      end)

      refute output =~ "val"
    end

    test "uses custom title" do
      history = [%{epoch: 1, train_loss: 0.5}]
      output = capture_io(:stderr, fn ->
        Output.terminal_loss_graph(history, title: "Custom Title")
      end)

      assert output =~ "Custom Title"
    end
  end

  describe "loss_sparkline/2" do
    import ExUnit.CaptureIO

    test "handles empty losses" do
      output = capture_io(:stderr, fn ->
        Output.loss_sparkline([])
      end)
      assert output =~ "no data"
    end

    test "renders sparkline for loss values" do
      losses = [0.8, 0.6, 0.4, 0.3, 0.2]
      output = capture_io(:stderr, fn ->
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
      output = capture_io(:stderr, fn ->
        Output.loss_sparkline(losses, label: "Val Loss")
      end)

      assert output =~ "Val Loss:"
    end

    test "handles constant values" do
      losses = [0.5, 0.5, 0.5]
      output = capture_io(:stderr, fn ->
        Output.loss_sparkline(losses)
      end)

      # Should still render something
      assert output =~ "Loss:"
      assert output =~ "0.5"
    end
  end
end
