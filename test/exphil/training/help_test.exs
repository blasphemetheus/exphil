defmodule ExPhil.Training.HelpTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Help

  describe "link/1" do
    test "returns link for known topic" do
      assert Help.link(:temporal) == "See docs/TRAINING.md#temporal-training"
      assert Help.link(:backbone) == "See docs/TRAINING.md#temporal-options"
      assert Help.link(:mamba) == "See docs/TRAINING.md#mamba-specific-options"
    end

    test "returns link without anchor when not specified" do
      assert Help.link(:architecture) == "See docs/ARCHITECTURE.md"
      assert Help.link(:gotchas) == "See docs/GOTCHAS.md"
    end

    test "returns nil for unknown topic" do
      assert Help.link(:unknown_topic) == nil
      assert Help.link(:nonexistent) == nil
    end
  end

  describe "link!/1" do
    test "returns link for known topic" do
      assert Help.link!(:temporal) == "See docs/TRAINING.md#temporal-training"
    end

    test "returns empty string for unknown topic" do
      assert Help.link!(:unknown) == ""
    end
  end

  describe "with_link/2" do
    test "appends help link to message" do
      result = Help.with_link("Invalid backbone", :backbone)
      assert result == "Invalid backbone (See docs/TRAINING.md#temporal-options)"
    end

    test "returns message unchanged for unknown topic" do
      result = Help.with_link("Some error", :unknown)
      assert result == "Some error"
    end
  end

  describe "warning_with_help/2" do
    test "appends help link on new line" do
      result = Help.warning_with_help("Large window size", :window_size)
      assert result == "Large window size\n       See docs/TRAINING.md#temporal-options"
    end

    test "returns message unchanged for unknown topic" do
      result = Help.warning_with_help("Some warning", :unknown)
      assert result == "Some warning"
    end
  end

  describe "suggest_topic/1" do
    test "suggests temporal for temporal-related messages" do
      assert Help.suggest_topic("temporal training requires...") == :temporal
    end

    test "suggests backbone for backbone messages" do
      assert Help.suggest_topic("Invalid backbone: gru") == :backbone
    end

    test "suggests batch_size for batch-related messages" do
      assert Help.suggest_topic("batch size too large") == :batch_size
    end

    test "suggests learning_rate for lr messages" do
      assert Help.suggest_topic("learning rate should be...") == :learning_rate
      assert Help.suggest_topic("lr is too high") == :learning_rate
    end

    test "suggests checkpoint for checkpoint messages" do
      assert Help.suggest_topic("checkpoint not found") == :checkpoint
    end

    test "suggests wandb for wandb messages" do
      assert Help.suggest_topic("enable --wandb for tracking") == :wandb
      assert Help.suggest_topic("W&B logging disabled") == :wandb
    end

    test "suggests frame_delay for online/delay messages" do
      assert Help.suggest_topic("frame delay is too high") == :frame_delay
      assert Help.suggest_topic("online play requires delay") == :frame_delay
    end

    test "returns nil for unrecognized messages" do
      assert Help.suggest_topic("completely unrelated message") == nil
    end
  end

  describe "auto_help/1" do
    test "adds help link based on keyword detection" do
      result = Help.auto_help("temporal training with window_size < 30")
      assert result =~ "temporal training with window_size < 30"
      assert result =~ "See docs/TRAINING.md"
    end

    test "returns message unchanged when no topic detected" do
      result = Help.auto_help("everything is fine")
      assert result == "everything is fine"
    end
  end

  describe "topics/0" do
    test "returns list of available topics" do
      topics = Help.topics()
      assert is_list(topics)
      assert :temporal in topics
      assert :backbone in topics
      assert :batch_size in topics
    end
  end
end
