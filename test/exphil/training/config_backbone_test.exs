defmodule ExPhil.Training.ConfigBackboneTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Config

  describe "backbone_defaults/1" do
    test "mamba returns temporal + f32 + cosine_restarts" do
      defaults = Config.backbone_defaults(:mamba)
      assert defaults[:temporal] == true
      assert defaults[:precision] == :f32
      assert defaults[:lr_schedule] == :cosine_restarts
      assert defaults[:state_size] == 16
      assert defaults[:expand_factor] == 2
    end

    test "mlp returns non-temporal" do
      defaults = Config.backbone_defaults(:mlp)
      assert defaults[:temporal] == false
      assert defaults[:dropout] == 0.1
    end

    test "lstm returns f32 precision" do
      defaults = Config.backbone_defaults(:lstm)
      assert defaults[:temporal] == true
      assert defaults[:precision] == :f32
    end

    test "griffin returns temporal + f32" do
      defaults = Config.backbone_defaults(:griffin)
      assert defaults[:temporal] == true
      assert defaults[:precision] == :f32
    end

    test "attention returns chunked_attention true" do
      defaults = Config.backbone_defaults(:attention)
      assert defaults[:chunked_attention] == true
      assert defaults[:num_heads] == 4
    end

    test "jamba has low learning rate" do
      defaults = Config.backbone_defaults(:jamba)
      assert defaults[:learning_rate] == 5.0e-6
      assert defaults[:max_grad_norm] == 0.25
    end

    test "unknown backbone returns empty list" do
      assert Config.backbone_defaults(:nonexistent) == []
    end
  end

  describe "parse_args with backbone defaults" do
    test "backbone defaults are auto-applied" do
      opts = Config.parse_args(["--backbone", "mamba", "--replays", "./replays"])
      assert opts[:temporal] == true
      assert opts[:precision] == :f32
    end

    test "explicit CLI args override backbone defaults" do
      opts = Config.parse_args(["--backbone", "mamba", "--precision", "f32", "--replays", "./replays"])
      assert opts[:precision] == :f32
    end

    test "explicit temporal false overrides backbone" do
      # MLP with explicit non-temporal flag
      opts = Config.parse_args(["--backbone", "mlp", "--replays", "./replays"])
      assert opts[:temporal] == false
    end
  end

  describe "defaults/0" do
    test "stick_edge_weight defaults to 2.0" do
      defaults = Config.defaults()
      assert defaults[:stick_edge_weight] == 2.0
    end

    test "lazy_sequences defaults to true" do
      defaults = Config.defaults()
      assert defaults[:lazy_sequences] == true
    end

    test "stride defaults to 5" do
      defaults = Config.defaults()
      assert defaults[:stride] == 5
    end

    test "dropout defaults to 0.0" do
      defaults = Config.defaults()
      assert defaults[:dropout] == 0.0
    end
  end
end
