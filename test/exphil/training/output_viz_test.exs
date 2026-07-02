defmodule ExPhil.Training.OutputVizTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Output

  describe "comparison_bar/4" do
    test "generates bar with correct label and percentages" do
      bar = Output.comparison_bar("A", 5.0, 6.0)
      assert bar =~ "A"
      assert bar =~ "5.0%"
      assert bar =~ "6.0%"
      assert bar =~ "vs"
    end

    test "shows COLLAPSE when pred is 0 and actual > threshold" do
      bar = Output.comparison_bar("R", 0.0, 5.0)
      assert bar =~ "COLLAPSE"
    end

    test "no COLLAPSE when actual is below threshold" do
      bar = Output.comparison_bar("R", 0.0, 0.5)
      refute bar =~ "COLLAPSE"
    end

    test "green color when pred close to actual" do
      bar = Output.comparison_bar("A", 5.0, 5.5)
      # Green ANSI = \e[32m
      assert bar =~ "\e[32m"
    end

    test "yellow color when pred far from actual" do
      bar = Output.comparison_bar("X", 15.0, 3.0)
      # Yellow ANSI = \e[33m
      assert bar =~ "\e[33m"
    end

    test "custom bar width" do
      bar = Output.comparison_bar("A", 50.0, 50.0, bar_width: 10)
      # 50% of 10 = 5 filled blocks
      assert bar =~ "█████"
    end

    test "handles 0% and 100%" do
      bar0 = Output.comparison_bar("X", 0.0, 0.0)
      assert bar0 =~ "0.0%"

      bar100 = Output.comparison_bar("X", 100.0, 100.0)
      assert bar100 =~ "100.0%"
    end
  end

  describe "sparkline/1" do
    test "returns empty for fewer than 2 values" do
      assert Output.sparkline([1.0]) == ""
      assert Output.sparkline([]) == ""
    end

    test "produces block characters for descending values" do
      spark = Output.sparkline([10.0, 7.0, 4.0, 1.0])
      assert String.length(spark) == 4
      # First should be highest block, last should be lowest
      assert String.at(spark, 0) == "█"
      assert String.at(spark, 3) == "▁"
    end

    test "produces uniform blocks for flat values" do
      spark = Output.sparkline([5.0, 5.0, 5.0])
      # All same value → all same block
      assert String.at(spark, 0) == String.at(spark, 1)
      assert String.at(spark, 1) == String.at(spark, 2)
    end

    test "handles two values" do
      spark = Output.sparkline([10.0, 1.0])
      assert String.length(spark) == 2
    end
  end

  describe "sparkline_with_label/2" do
    test "returns empty for fewer than 2 values" do
      assert Output.sparkline_with_label("Loss", [1.0]) == ""
    end

    test "includes label and range" do
      result = Output.sparkline_with_label("Loss", [3.0, 2.0, 1.0])
      assert result =~ "Loss:"
      assert result =~ "3.0"
      assert result =~ "1.0"
      assert result =~ "→"
    end
  end

  describe "summary_box/3" do
    test "renders box with title and entries" do
      box = Output.summary_box("Epoch 1/10", [
        {"train_loss", "2.5"},
        {"val_loss", "2.3"}
      ])

      assert box =~ "╭"
      assert box =~ "╰"
      assert box =~ "Epoch 1/10"
      assert box =~ "train_loss"
      assert box =~ "2.5"
    end

    test "renders highlight line" do
      box = Output.summary_box("Test", [{"k", "v"}], highlight: "Best model!")
      assert box =~ "Best model!"
    end

    test "no highlight line when nil" do
      box = Output.summary_box("Test", [{"k", "v"}], highlight: nil)
      refute box =~ "\e[32m"
    end
  end

  describe "table/3" do
    test "renders aligned table" do
      result = Output.table(["Name", "Value"], [
        ["alpha", "100"],
        ["beta", "2"]
      ])

      assert result =~ "Name"
      assert result =~ "Value"
      assert result =~ "alpha"
      assert result =~ "100"
      assert result =~ "─"
    end

    test "handles varying column widths" do
      result = Output.table(["A", "Long Header"], [
        ["x", "short"],
        ["longer_value", "y"]
      ])

      # Column widths should accommodate longest entry
      assert result =~ "longer_value"
      assert result =~ "Long Header"
    end

    test "respects indent option" do
      result = Output.table(["H"], [["v"]], indent: ">>")
      assert result =~ ">>"
    end

    test "handles empty rows" do
      result = Output.table(["H1", "H2"], [])
      assert result =~ "H1"
      assert result =~ "─"
    end
  end
end
