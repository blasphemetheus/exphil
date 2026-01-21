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
end
