defmodule ExPhil.PropertyTests.ConfigPropertiesTest do
  @moduledoc """
  Property-based tests for training configuration.

  These tests verify that config parsing is deterministic and that
  all presets produce valid configurations.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExPhil.Training.Config

  @moduletag :property

  describe "parsing determinism" do
    property "parse_args produces same result for same input" do
      check all(
              epochs <- StreamData.integer(1..100),
              batch_size <- StreamData.member_of([32, 64, 128, 256, 512]),
              learning_rate <- StreamData.float(min: 1.0e-5, max: 1.0e-2),
              max_runs: 50
            ) do
        args = [
          "--epochs",
          "#{epochs}",
          "--batch-size",
          "#{batch_size}",
          "--learning-rate",
          "#{learning_rate}"
        ]

        # parse_args returns keyword list directly
        result1 = Config.parse_args(args)
        result2 = Config.parse_args(args)

        assert result1[:epochs] == result2[:epochs]
        assert result1[:batch_size] == result2[:batch_size]
        assert result1[:learning_rate] == result2[:learning_rate]
      end
    end

    property "parse_args order independence for independent flags" do
      check all(
              epochs <- StreamData.integer(1..100),
              batch_size <- StreamData.member_of([32, 64, 128, 256]),
              max_runs: 30
            ) do
        args1 = ["--epochs", "#{epochs}", "--batch-size", "#{batch_size}"]
        args2 = ["--batch-size", "#{batch_size}", "--epochs", "#{epochs}"]

        result1 = Config.parse_args(args1)
        result2 = Config.parse_args(args2)

        assert result1[:epochs] == result2[:epochs]
        assert result1[:batch_size] == result2[:batch_size]
      end
    end
  end

  describe "preset validity" do
    # Must match actual preset names in Config.Presets
    @presets [:quick, :standard, :production, :full, :mewtwo, :ganondorf, :link, :gameandwatch, :zelda]

    property "all presets produce valid configurations" do
      check all(
              preset <- StreamData.member_of(@presets),
              max_runs: length(@presets)
            ) do
        # Get preset config
        preset_opts = Config.preset(preset)
        assert is_list(preset_opts)

        # Merge with defaults
        merged = Keyword.merge(Config.defaults(), preset_opts)

        # Should have required fields
        assert Keyword.has_key?(merged, :epochs)
        assert Keyword.has_key?(merged, :batch_size)
        assert Keyword.has_key?(merged, :learning_rate)

        # Values should be valid
        assert merged[:epochs] > 0
        assert merged[:batch_size] > 0
        assert merged[:learning_rate] > 0
      end
    end

    property "preset validation passes for all presets" do
      check all(
              preset <- StreamData.member_of(@presets),
              max_runs: length(@presets)
            ) do
        # Parse with preset
        args = ["--preset", "#{preset}"]
        opts = Config.parse_args(args)

        # Validation should pass (may have warnings but no errors)
        result = Config.validate(opts)
        assert match?({:ok, _}, result) or match?({:ok, _, _}, result)
      end
    end
  end

  describe "numeric bounds" do
    property "epochs must be positive when provided" do
      check all(
              epochs <- StreamData.integer(1..100),
              max_runs: 20
            ) do
        args = ["--epochs", "#{epochs}"]
        opts = Config.parse_args(args)

        assert opts[:epochs] == epochs
        assert opts[:epochs] > 0
      end
    end

    property "batch_size must be positive when provided" do
      check all(
              batch_size <- StreamData.member_of([16, 32, 64, 128, 256, 512]),
              max_runs: 20
            ) do
        args = ["--batch-size", "#{batch_size}"]
        opts = Config.parse_args(args)

        assert opts[:batch_size] == batch_size
        assert opts[:batch_size] > 0
      end
    end

    property "learning_rate must be positive when provided" do
      check all(
              lr <- StreamData.float(min: 1.0e-6, max: 1.0e-2),
              max_runs: 20
            ) do
        args = ["--learning-rate", "#{lr}"]
        opts = Config.parse_args(args)

        assert_in_delta opts[:learning_rate], lr, 1.0e-10
        assert opts[:learning_rate] > 0
      end
    end
  end

  describe "flag combinations" do
    property "temporal flag enables window_size" do
      check all(
              window_size <- StreamData.integer(10..120),
              max_runs: 20
            ) do
        args = ["--temporal", "--window-size", "#{window_size}"]
        opts = Config.parse_args(args)

        assert opts[:temporal] == true
        assert opts[:window_size] == window_size
      end
    end

    property "backbone selection is preserved" do
      backbones = [:mlp, :lstm, :gru, :mamba, :attention]

      check all(
              backbone <- StreamData.member_of(backbones),
              max_runs: length(backbones)
            ) do
        args = ["--backbone", "#{backbone}"]
        opts = Config.parse_args(args)

        assert opts[:backbone] == backbone
      end
    end
  end
end
