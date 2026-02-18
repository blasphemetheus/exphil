defmodule ExPhil.Training.ConfigTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config
  alias ExPhil.Error.YamlError

  # Run doctests for this module
  doctest ExPhil.Training.Config

  # ============================================================================
  # Argument Parsing Tests
  # ============================================================================

  describe "parse_args/1" do
    test "returns defaults for empty args" do
      opts = Config.parse_args([])

      assert opts[:epochs] == 10
      assert opts[:batch_size] == 64
      assert opts[:hidden_sizes] == [512, 512, 256]
      assert opts[:temporal] == false
      # FP32 is default - BF16 is 2x slower on RTX 4090 due to XLA issues
      assert opts[:precision] == :f32
    end

    test "parses --epochs" do
      opts = Config.parse_args(["--epochs", "5"])
      assert opts[:epochs] == 5
    end

    test "parses --batch-size" do
      opts = Config.parse_args(["--batch-size", "128"])
      assert opts[:batch_size] == 128
    end

    test "parses --hidden-sizes" do
      opts = Config.parse_args(["--hidden-sizes", "32,32"])
      assert opts[:hidden_sizes] == [32, 32]
    end

    test "parses --hidden-sizes with spaces" do
      opts = Config.parse_args(["--hidden-sizes", "128, 64, 32"])
      assert opts[:hidden_sizes] == [128, 64, 32]
    end

    test "parses --temporal flag" do
      opts = Config.parse_args(["--temporal"])
      assert opts[:temporal] == true
    end

    test "parses --backbone" do
      opts = Config.parse_args(["--backbone", "mamba"])
      assert opts[:backbone] == :mamba
    end

    test "parses --precision f32" do
      opts = Config.parse_args(["--precision", "f32"])
      assert opts[:precision] == :f32
    end

    test "parses --precision bf16" do
      opts = Config.parse_args(["--precision", "bf16"])
      assert opts[:precision] == :bf16
    end

    test "raises on unknown precision" do
      assert_raise RuntimeError, ~r/Unknown precision/, fn ->
        Config.parse_args(["--precision", "fp16"])
      end
    end

    test "parses --max-files" do
      opts = Config.parse_args(["--max-files", "10"])
      assert opts[:max_files] == 10
    end

    test "parses --frame-delay" do
      opts = Config.parse_args(["--frame-delay", "18"])
      assert opts[:frame_delay] == 18
    end

    test "parses --wandb flag" do
      opts = Config.parse_args(["--wandb"])
      assert opts[:wandb] == true
    end

    test "parses multiple args together" do
      opts =
        Config.parse_args([
          "--epochs",
          "3",
          "--batch-size",
          "32",
          "--temporal",
          "--backbone",
          "lstm",
          "--window-size",
          "30"
        ])

      assert opts[:epochs] == 3
      assert opts[:batch_size] == 32
      assert opts[:temporal] == true
      assert opts[:backbone] == :lstm
      assert opts[:window_size] == 30
    end

    test "parses Mamba-specific options" do
      opts =
        Config.parse_args([
          "--backbone",
          "mamba",
          "--state-size",
          "32",
          "--expand-factor",
          "4",
          "--conv-size",
          "8"
        ])

      assert opts[:backbone] == :mamba
      assert opts[:state_size] == 32
      assert opts[:expand_factor] == 4
      assert opts[:conv_size] == 8
    end

    test "parses --truncate-bptt" do
      opts = Config.parse_args(["--truncate-bptt", "20"])
      assert opts[:truncate_bptt] == 20
    end

    test "leaves truncate_bptt nil if not specified" do
      opts = Config.parse_args([])
      assert opts[:truncate_bptt] == nil
    end

    test "parses --layer-norm flag" do
      opts = Config.parse_args(["--layer-norm"])
      assert opts[:layer_norm] == true
    end

    test "parses --no-layer-norm flag" do
      opts = Config.parse_args(["--layer-norm", "--no-layer-norm"])
      assert opts[:layer_norm] == false
    end

    test "layer_norm defaults to false" do
      opts = Config.parse_args([])
      assert opts[:layer_norm] == false
    end

    test "parses --residual flag" do
      opts = Config.parse_args(["--residual"])
      assert opts[:residual] == true
    end

    test "parses --no-residual flag" do
      opts = Config.parse_args(["--residual", "--no-residual"])
      assert opts[:residual] == false
    end

    test "residual defaults to false" do
      opts = Config.parse_args([])
      assert opts[:residual] == false
    end

    test "parses --residual with --layer-norm together" do
      opts = Config.parse_args(["--residual", "--layer-norm"])
      assert opts[:residual] == true
      assert opts[:layer_norm] == true
    end

    test "parses --kmeans-centers" do
      opts = Config.parse_args(["--kmeans-centers", "priv/kmeans_centers.nx"])
      assert opts[:kmeans_centers] == "priv/kmeans_centers.nx"
    end

    test "kmeans_centers defaults to nil" do
      opts = Config.parse_args([])
      assert opts[:kmeans_centers] == nil
    end
  end

  # ============================================================================
  # Checkpoint Naming Tests
  # ============================================================================

  describe "ensure_checkpoint_name/1" do
    test "generates timestamped name for MLP when checkpoint is nil" do
      opts = [checkpoint: nil, temporal: false, backbone: :sliding_window]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mlp_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mlp_[a-z0-9_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end

    test "generates timestamped name with backbone for temporal" do
      opts = [checkpoint: nil, temporal: true, backbone: :mamba]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mamba_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mamba_[a-z0-9_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end

    test "generates timestamped name with lstm backbone" do
      opts = [checkpoint: nil, temporal: true, backbone: :lstm]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/lstm_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/lstm_[a-z0-9_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end

    test "preserves explicit checkpoint name" do
      opts = [checkpoint: "my_custom_model.axon", temporal: false]
      result = Config.ensure_checkpoint_name(opts)

      assert result[:checkpoint] == "my_custom_model.axon"
    end

    test "preserves explicit checkpoint path" do
      opts = [checkpoint: "models/v2/policy.axon", temporal: true, backbone: :mamba]
      result = Config.ensure_checkpoint_name(opts)

      assert result[:checkpoint] == "models/v2/policy.axon"
    end

    test "user-provided --name does not get backbone prefix" do
      # When user provides --name mlp_mewtwo_prod, it should NOT become mlp_mlp_mewtwo_prod
      opts = [checkpoint: nil, temporal: false, backbone: :mlp, name: "mlp_mewtwo_prod"]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/name_timestamp.axon (no backbone prefix)
      assert result[:checkpoint] =~ ~r/checkpoints\/mlp_mewtwo_prod_\d{8}_\d{6}\.axon/
      assert result[:name] == "mlp_mewtwo_prod"
    end

    test "user-provided --name with character does not get backbone prefix" do
      opts = [checkpoint: nil, temporal: false, backbone: :mlp, name: "my_model", character: :mewtwo]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/character_name_timestamp.axon (no backbone prefix)
      assert result[:checkpoint] =~ ~r/checkpoints\/mewtwo_my_model_\d{8}_\d{6}\.axon/
      assert result[:name] == "my_model"
    end
  end

  describe "generate_timestamp/1" do
    test "formats DateTime correctly" do
      dt = ~U[2026-01-19 12:34:56Z]
      result = Config.generate_timestamp(dt)

      assert result == "20260119_123456"
    end

    test "pads single-digit values" do
      dt = ~U[2026-03-05 01:02:03Z]
      result = Config.generate_timestamp(dt)

      assert result == "20260305_010203"
    end
  end

  # ============================================================================
  # Path Derivation Tests
  # ============================================================================

  describe "derive_policy_path/1" do
    test "replaces .axon with _policy.bin" do
      result = Config.derive_policy_path("checkpoints/mlp_20260119.axon")
      assert result == "checkpoints/mlp_20260119_policy.bin"
    end

    test "handles nested paths" do
      result = Config.derive_policy_path("models/v2/mamba_123.axon")
      assert result == "models/v2/mamba_123_policy.bin"
    end

    test "returns nil for nil input" do
      assert Config.derive_policy_path(nil) == nil
    end
  end

  describe "derive_config_path/1" do
    test "replaces .axon with _config.json" do
      result = Config.derive_config_path("checkpoints/mlp_20260119.axon")
      assert result == "checkpoints/mlp_20260119_config.json"
    end

    test "returns nil for nil input" do
      assert Config.derive_config_path(nil) == nil
    end
  end

  # ============================================================================
  # Hidden Sizes Parsing Tests
  # ============================================================================

  describe "parse_hidden_sizes/1" do
    test "parses comma-separated integers" do
      assert Config.parse_hidden_sizes("64,64") == [64, 64]
    end

    test "parses with spaces" do
      assert Config.parse_hidden_sizes("128, 64, 32") == [128, 64, 32]
    end

    test "parses single value" do
      assert Config.parse_hidden_sizes("256") == [256]
    end

    test "parses many values" do
      assert Config.parse_hidden_sizes("512,256,128,64,32") == [512, 256, 128, 64, 32]
    end
  end

  # ============================================================================
  # Config JSON Building Tests
  # ============================================================================

  describe "build_config_json/2" do
    test "builds config with all required fields" do
      opts = [
        replays: "/path/to/replays",
        max_files: 10,
        player_port: 1,
        temporal: false,
        backbone: :sliding_window,
        hidden_sizes: [64, 64],
        window_size: 60,
        stride: 1,
        num_layers: 2,
        truncate_bptt: nil,
        state_size: 16,
        expand_factor: 2,
        conv_size: 4,
        epochs: 5,
        batch_size: 32,
        precision: :bf16,
        frame_delay: 0,
        checkpoint: "checkpoints/test.axon"
      ]

      results = %{
        embed_size: 1991,
        training_frames: 10000,
        validation_frames: 1000,
        total_time_seconds: 300,
        final_training_loss: 5.5
      }

      config = Config.build_config_json(opts, results)

      # Check input parameters
      assert config[:replays_dir] == "/path/to/replays"
      assert config[:max_files] == 10
      assert config[:player_port] == 1

      # Check architecture
      assert config[:temporal] == false
      assert config[:backbone] == "mlp"
      assert config[:hidden_sizes] == [64, 64]
      assert config[:embed_size] == 1991

      # Check training params
      assert config[:epochs] == 5
      assert config[:batch_size] == 32
      assert config[:precision] == "bf16"

      # Check results
      assert config[:training_frames] == 10000
      assert config[:final_training_loss] == 5.5

      # Check derived paths
      assert config[:checkpoint_path] == "checkpoints/test.axon"
      assert config[:policy_path] == "checkpoints/test_policy.bin"

      # Check timestamp exists
      assert is_binary(config[:timestamp])
    end

    test "uses temporal backbone name when temporal is true" do
      opts =
        [
          temporal: true,
          backbone: :mamba,
          hidden_sizes: [128],
          replays: "/replays",
          checkpoint: "test.axon"
        ] ++ Config.defaults()

      config = Config.build_config_json(opts)

      assert config[:temporal] == true
      assert config[:backbone] == "mamba"
    end

    test "handles nil results gracefully" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")
      config = Config.build_config_json(opts, %{})

      assert config[:embed_size] == nil
      assert config[:training_frames] == nil
    end
  end

  # ============================================================================
  # Defaults Tests
  # ============================================================================

  describe "defaults/0" do
    test "returns expected default values" do
      defaults = Config.defaults()

      assert defaults[:epochs] == 10
      assert defaults[:batch_size] == 64
      assert defaults[:hidden_sizes] == [512, 512, 256]
      assert defaults[:temporal] == false
      assert defaults[:backbone] == :sliding_window
      assert defaults[:window_size] == 60
      # FP32 is default - BF16 is 2x slower on RTX 4090 due to XLA issues
      assert defaults[:precision] == :f32
      assert defaults[:frame_delay] == 0
      assert defaults[:checkpoint] == nil
      assert defaults[:max_files] == nil
    end
  end

  # ============================================================================
  # Edge Cases and Validation Tests
  # ============================================================================

  describe "parse_args/1 edge cases" do
    test "ignores unknown flags" do
      opts = Config.parse_args(["--unknown-flag", "value", "--epochs", "5"])
      assert opts[:epochs] == 5
    end

    test "handles flag at end of args without value" do
      # --epochs at end with no value should use default
      opts = Config.parse_args(["--batch-size", "32", "--epochs"])
      assert opts[:batch_size] == 32
      # epochs tries to parse nil, which would error - but we handle gracefully
    end

    test "parses args in any order" do
      opts =
        Config.parse_args([
          "--temporal",
          "--epochs",
          "3",
          "--wandb",
          "--batch-size",
          "16"
        ])

      assert opts[:epochs] == 3
      assert opts[:batch_size] == 16
      assert opts[:temporal] == true
      assert opts[:wandb] == true
    end

    test "first value wins for duplicate flags" do
      # Current behavior: first occurrence takes precedence
      opts = Config.parse_args(["--epochs", "5", "--epochs", "10"])
      assert opts[:epochs] == 5
    end

    test "parses all backbone types" do
      for backbone <- ~w(sliding_window hybrid lstm gru mamba) do
        opts = Config.parse_args(["--backbone", backbone])
        assert opts[:backbone] == String.to_atom(backbone)
      end
    end

    test "parses --replays path with spaces when quoted" do
      opts = Config.parse_args(["--replays", "/path/with spaces/replays"])
      assert opts[:replays] == "/path/with spaces/replays"
    end

    test "parses zero values correctly" do
      opts = Config.parse_args(["--frame-delay", "0", "--stride", "1"])
      assert opts[:frame_delay] == 0
      assert opts[:stride] == 1
    end

    test "parses large values" do
      opts = Config.parse_args(["--epochs", "1000", "--batch-size", "512"])
      assert opts[:epochs] == 1000
      assert opts[:batch_size] == 512
    end
  end

  # ============================================================================
  # Checkpoint Naming Edge Cases
  # ============================================================================

  describe "ensure_checkpoint_name/1 edge cases" do
    test "generates unique timestamps" do
      opts = [checkpoint: nil, temporal: false, backbone: :mlp]

      result1 = Config.ensure_checkpoint_name(opts)
      # Small delay to ensure different timestamp
      Process.sleep(1100)
      result2 = Config.ensure_checkpoint_name(opts)

      # Timestamps should be different (at least by 1 second)
      assert result1[:checkpoint] != result2[:checkpoint]
    end

    test "handles all temporal backbones in naming" do
      for backbone <- [:lstm, :gru, :mamba, :sliding_window, :lstm_hybrid, :jamba] do
        opts = [checkpoint: nil, temporal: true, backbone: backbone]
        result = Config.ensure_checkpoint_name(opts)

        # Format: checkpoints/backbone_memorablename_timestamp.axon
        assert result[:checkpoint] =~ ~r/checkpoints\/#{backbone}_[a-z0-9_]+_\d{8}_\d{6}\.axon/
        # Should also set the :name option
        assert is_binary(result[:name])
      end
    end

    test "non-temporal always uses mlp in name regardless of backbone setting" do
      opts = [checkpoint: nil, temporal: false, backbone: :mamba]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mlp_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mlp_[a-z0-9_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end
  end

  # ============================================================================
  # Validation Tests (for future validate/1 function)
  # ============================================================================

  describe "validation scenarios" do
    test "epochs must be positive" do
      opts = Config.parse_args(["--epochs", "0"])
      # Currently no validation - this documents expected behavior
      assert opts[:epochs] == 0
    end

    test "batch_size must be positive" do
      opts = Config.parse_args(["--batch-size", "1"])
      assert opts[:batch_size] == 1
    end

    test "window_size should be reasonable" do
      opts = Config.parse_args(["--window-size", "120"])
      assert opts[:window_size] == 120
    end

    test "frame_delay can be large for bad connections" do
      opts = Config.parse_args(["--frame-delay", "36"])
      assert opts[:frame_delay] == 36
    end
  end

  # ============================================================================
  # Config JSON Serialization Tests
  # ============================================================================

  describe "build_config_json/2 serialization" do
    test "all values are JSON-serializable" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")
      results = %{embed_size: 1991, training_frames: 10000}

      config = Config.build_config_json(opts, results)

      # Should not raise
      json = Jason.encode!(config)
      assert is_binary(json)

      # Should round-trip
      decoded = Jason.decode!(json)
      assert decoded["epochs"] == 10
      assert decoded["embed_size"] == 1991
    end

    test "handles float results correctly" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")
      results = %{final_training_loss: 3.14159265359}

      config = Config.build_config_json(opts, results)
      json = Jason.encode!(config)
      decoded = Jason.decode!(json)

      assert_in_delta decoded["final_training_loss"], 3.14159, 0.001
    end

    test "timestamp is ISO8601 format" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")
      config = Config.build_config_json(opts)

      # Should be parseable as ISO8601
      {:ok, _dt, _offset} = DateTime.from_iso8601(config[:timestamp])
    end
  end

  # ============================================================================
  # Training Presets Tests
  # ============================================================================

  describe "preset/1" do
    test "quick preset returns fast iteration config" do
      opts = Config.preset(:quick)

      assert opts[:epochs] == 1
      assert opts[:max_files] == 5
      assert opts[:hidden_sizes] == [32, 32]
      assert opts[:temporal] == false
      assert opts[:preset] == :quick
    end

    test "standard preset returns balanced config" do
      opts = Config.preset(:standard)

      assert opts[:epochs] == 10
      assert opts[:max_files] == 50
      assert opts[:hidden_sizes] == [64, 64]
      assert opts[:temporal] == false
      assert opts[:preset] == :standard
    end

    test "full preset returns maximum quality config" do
      opts = Config.preset(:full)

      assert opts[:epochs] == 50
      assert opts[:max_files] == nil
      assert opts[:hidden_sizes] == [256, 256]
      assert opts[:temporal] == true
      assert opts[:backbone] == :mamba
      assert opts[:preset] == :full
    end

    test "full_cpu preset returns CPU-optimized full config" do
      opts = Config.preset(:full_cpu)

      assert opts[:epochs] == 30
      assert opts[:max_files] == 200
      assert opts[:hidden_sizes] == [128, 128]
      assert opts[:temporal] == false
      assert opts[:preset] == :full_cpu
      # Should include best practices
      assert opts[:ema] == true
      assert opts[:augment] == true
    end

    test "mewtwo preset includes character and longer window" do
      opts = Config.preset(:mewtwo)

      assert opts[:character] == :mewtwo
      assert opts[:window_size] == 90
      assert opts[:temporal] == true
      assert opts[:preset] == :mewtwo
    end

    test "ganondorf preset includes character" do
      opts = Config.preset(:ganondorf)

      assert opts[:character] == :ganondorf
      assert opts[:window_size] == 60
      assert opts[:preset] == :ganondorf
    end

    test "link preset includes character and longer window" do
      opts = Config.preset(:link)

      assert opts[:character] == :link
      assert opts[:window_size] == 75
      assert opts[:preset] == :link
    end

    test "gameandwatch preset includes character and shorter window" do
      opts = Config.preset(:gameandwatch)

      assert opts[:character] == :gameandwatch
      assert opts[:window_size] == 45
      assert opts[:preset] == :gameandwatch
    end

    test "zelda preset includes character" do
      opts = Config.preset(:zelda)

      assert opts[:character] == :zelda
      assert opts[:window_size] == 60
      assert opts[:preset] == :zelda
    end

    test "preset/1 accepts string name" do
      opts = Config.preset("quick")
      assert opts[:preset] == :quick
    end

    test "preset/1 raises on invalid preset" do
      assert_raise ArgumentError, ~r/Unknown preset/, fn ->
        Config.preset(:invalid)
      end
    end

    test "character presets inherit from production" do
      production = Config.preset(:production)
      mewtwo = Config.preset(:mewtwo)

      # Should have same temporal and backbone
      assert mewtwo[:temporal] == production[:temporal]
      assert mewtwo[:backbone] == production[:backbone]
      assert mewtwo[:epochs] == production[:epochs]
      # Should have EMA and other production features
      assert mewtwo[:ema] == production[:ema]
      assert mewtwo[:lr_schedule] == production[:lr_schedule]
    end
  end

  describe "parse_args/1 with presets" do
    test "parses --preset quick" do
      opts = Config.parse_args(["--preset", "quick"])

      assert opts[:epochs] == 1
      assert opts[:max_files] == 5
      assert opts[:hidden_sizes] == [32, 32]
      assert opts[:preset] == :quick
    end

    test "parses --preset standard" do
      opts = Config.parse_args(["--preset", "standard"])

      assert opts[:epochs] == 10
      assert opts[:max_files] == 50
    end

    test "parses --preset full" do
      opts = Config.parse_args(["--preset", "full"])

      assert opts[:epochs] == 50
      assert opts[:temporal] == true
      assert opts[:backbone] == :mamba
    end

    test "parses --preset mewtwo" do
      opts = Config.parse_args(["--preset", "mewtwo"])

      assert opts[:character] == :mewtwo
      assert opts[:window_size] == 90
    end

    test "CLI args override preset values" do
      opts = Config.parse_args(["--preset", "quick", "--epochs", "3"])

      # Overridden
      assert opts[:epochs] == 3
      # From preset
      assert opts[:max_files] == 5
      # From preset
      assert opts[:hidden_sizes] == [32, 32]
    end

    test "multiple CLI args can override preset" do
      opts =
        Config.parse_args([
          "--preset",
          "quick",
          "--epochs",
          "5",
          "--batch-size",
          "128",
          "--hidden-sizes",
          "64,64"
        ])

      assert opts[:epochs] == 5
      assert opts[:batch_size] == 128
      assert opts[:hidden_sizes] == [64, 64]
      # Still from preset
      assert opts[:max_files] == 5
    end

    test "--temporal flag can override preset" do
      # quick preset has temporal: false
      opts = Config.parse_args(["--preset", "quick", "--temporal"])

      assert opts[:temporal] == true
    end

    test "--wandb flag works with preset" do
      opts = Config.parse_args(["--preset", "quick", "--wandb"])

      assert opts[:wandb] == true
      assert opts[:preset] == :quick
    end
  end

  describe "available_presets/0" do
    test "returns list of preset names" do
      presets = Config.available_presets()

      assert :quick in presets
      assert :standard in presets
      assert :full in presets
      assert :full_cpu in presets
      assert :mewtwo in presets
      assert :ganondorf in presets
      assert :link in presets
      assert :gameandwatch in presets
      assert :zelda in presets
    end
  end

  # ============================================================================
  # Model Naming Convention Tests (for future naming functions)
  # ============================================================================

  describe "model naming conventions" do
    test "current naming format" do
      # Current: {backbone}_{timestamp}.axon
      name = "mamba_20260119_123456.axon"
      assert name =~ ~r/^[a-z]+_\d{8}_\d{6}\.axon$/
    end

    test "proposed naming with character" do
      # Proposed: {character}_{backbone}_{timestamp}.axon
      name = "mewtwo_mamba_20260119_123456.axon"
      assert name =~ ~r/^[a-z]+_[a-z]+_\d{8}_\d{6}\.axon$/
    end

    test "proposed naming with hyperparameters" do
      # Proposed: {backbone}_h{hidden}_w{window}_{timestamp}.axon
      name = "mamba_h256_w60_20260119_123456.axon"
      assert name =~ ~r/^[a-z]+_h\d+_w\d+_\d{8}_\d{6}\.axon$/
    end

    test "proposed naming with version" do
      # Proposed: {character}_{backbone}_v{major}.{minor}.axon
      name = "mewtwo_mamba_v1.2.axon"
      assert name =~ ~r/^[a-z]+_[a-z]+_v\d+\.\d+\.axon$/
    end

    test "proposed naming with performance" do
      # Proposed: {backbone}_{timestamp}_loss{val_loss}.axon
      name = "mamba_20260119_123456_loss4.05.axon"
      assert name =~ ~r/^[a-z]+_\d{8}_\d{6}_loss[\d.]+\.axon$/
    end
  end

  # ============================================================================
  # Model Registry Tests (for future registry functionality)
  # ============================================================================

  describe "model registry concepts" do
    test "registry entry structure" do
      # What a registry entry might look like
      entry = %{
        id: "mamba_20260119_123456",
        path: "checkpoints/mamba_20260119_123456.axon",
        policy_path: "checkpoints/mamba_20260119_123456_policy.bin",
        config_path: "checkpoints/mamba_20260119_123456_config.json",
        created_at: ~U[2026-01-19 12:34:56Z],
        # or :mewtwo
        character: nil,
        backbone: :mamba,
        temporal: true,
        hidden_sizes: [256, 256],
        window_size: 60,
        epochs: 10,
        training_frames: 100_000,
        final_val_loss: 4.05,
        # or "lstm_20260118_000000" for fine-tuning
        parent_model: nil,
        tags: ["production", "v1.0"]
      }

      assert entry[:backbone] == :mamba
      assert entry[:final_val_loss] == 4.05
    end

    test "registry search by backbone" do
      # Simulating a search
      models = [
        %{id: "mlp_1", backbone: :mlp, val_loss: 5.0},
        %{id: "mamba_1", backbone: :mamba, val_loss: 4.0},
        %{id: "mamba_2", backbone: :mamba, val_loss: 3.5}
      ]

      mamba_models = Enum.filter(models, &(&1[:backbone] == :mamba))
      assert length(mamba_models) == 2
    end

    test "registry leaderboard by val_loss" do
      models = [
        %{id: "mlp_1", val_loss: 5.0},
        %{id: "mamba_1", val_loss: 4.0},
        %{id: "mamba_2", val_loss: 3.5}
      ]

      leaderboard = Enum.sort_by(models, & &1[:val_loss])
      assert hd(leaderboard)[:id] == "mamba_2"
    end

    test "model lineage tracking" do
      # Track fine-tuning chains
      models = %{
        "base_mlp" => %{parent: nil, epochs: 10},
        "fine_tune_1" => %{parent: "base_mlp", epochs: 5},
        "fine_tune_2" => %{parent: "fine_tune_1", epochs: 3}
      }

      # Get lineage of fine_tune_2
      lineage = get_lineage(models, "fine_tune_2")
      assert lineage == ["base_mlp", "fine_tune_1", "fine_tune_2"]
    end
  end

  # Helper for lineage test
  defp get_lineage(models, id, acc \\ []) do
    model = models[id]

    if model[:parent] do
      get_lineage(models, model[:parent], [id | acc])
    else
      [id | acc]
    end
  end

  # ============================================================================
  # Validation Tests
  # ============================================================================

  describe "validate/1" do
    test "returns {:ok, opts} for valid configuration" do
      opts = [epochs: 10, batch_size: 64, hidden_sizes: [64, 64]]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns {:ok, opts} for default configuration" do
      # Override replays to a valid temp directory since defaults use ./replays
      tmp_dir =
        System.tmp_dir!() |> Path.join("defaults_test_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(tmp_dir)
      on_exit(fn -> File.rm_rf!(tmp_dir) end)

      opts = Config.defaults() |> Keyword.put(:replays, tmp_dir)
      assert {:ok, _} = Config.validate(opts)
    end

    test "returns {:ok, opts} for all presets" do
      for preset <- Config.available_presets() do
        opts = Config.preset(preset)
        assert {:ok, _} = Config.validate(opts), "preset #{preset} should be valid"
      end
    end

    test "returns error for negative epochs" do
      opts = [epochs: -1, batch_size: 64]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "epochs"))
    end

    test "returns error for zero epochs" do
      opts = [epochs: 0, batch_size: 64]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "epochs"))
    end

    test "returns error for negative batch_size" do
      opts = [epochs: 10, batch_size: -32]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "batch_size"))
    end

    test "returns error for zero batch_size" do
      opts = [epochs: 10, batch_size: 0]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "batch_size"))
    end

    test "returns error for invalid max_files" do
      opts = [epochs: 10, batch_size: 64, max_files: -5]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "max_files"))
    end

    test "allows nil max_files" do
      opts = [epochs: 10, batch_size: 64, max_files: nil]
      assert {:ok, _} = Config.validate(opts)
    end

    test "returns error for negative window_size" do
      opts = [epochs: 10, batch_size: 64, window_size: -60]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "window_size"))
    end

    test "returns error for negative frame_delay" do
      opts = [epochs: 10, batch_size: 64, frame_delay: -1]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "frame_delay"))
    end

    test "allows zero frame_delay" do
      opts = [epochs: 10, batch_size: 64, frame_delay: 0]
      assert {:ok, _} = Config.validate(opts)
    end

    test "returns error for invalid hidden_sizes type" do
      opts = [epochs: 10, batch_size: 64, hidden_sizes: "64,64"]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "hidden_sizes"))
    end

    test "returns error for hidden_sizes with negative values" do
      opts = [epochs: 10, batch_size: 64, hidden_sizes: [64, -32]]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "hidden_sizes"))
    end

    test "returns error for hidden_sizes with zero values" do
      opts = [epochs: 10, batch_size: 64, hidden_sizes: [64, 0]]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "hidden_sizes"))
    end

    test "allows nil hidden_sizes" do
      opts = [epochs: 10, batch_size: 64, hidden_sizes: nil]
      assert {:ok, _} = Config.validate(opts)
    end

    test "returns error for invalid backbone with temporal" do
      opts = [epochs: 10, batch_size: 64, temporal: true, backbone: :invalid_backbone]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "backbone"))
    end

    test "allows any backbone without temporal" do
      # Without temporal: true, backbone isn't validated
      opts = [epochs: 10, batch_size: 64, temporal: false, backbone: :whatever]
      assert {:ok, _} = Config.validate(opts)
    end

    test "accepts all valid temporal backbones" do
      for backbone <- [:lstm, :gru, :mamba, :sliding_window, :lstm_hybrid, :jamba] do
        opts = [epochs: 10, batch_size: 64, temporal: true, backbone: backbone]
        assert {:ok, _} = Config.validate(opts), "backbone #{backbone} should be valid"
      end
    end

    test "returns error for invalid precision" do
      opts = [epochs: 10, batch_size: 64, precision: :f16]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "precision"))
    end

    test "accepts valid precision values" do
      for precision <- [:bf16, :f32] do
        opts = [epochs: 10, batch_size: 64, precision: precision]
        assert {:ok, _} = Config.validate(opts)
      end
    end

    test "returns error for non-existent replays directory" do
      opts = [epochs: 10, batch_size: 64, replays: "/nonexistent/path/12345"]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "replays"))
    end

    test "allows nil replays directory" do
      opts = [epochs: 10, batch_size: 64, replays: nil]
      assert {:ok, _} = Config.validate(opts)
    end

    test "collects multiple errors" do
      opts = [epochs: -1, batch_size: -32, hidden_sizes: "bad"]
      assert {:error, errors} = Config.validate(opts)
      assert length(errors) >= 3
    end
  end

  describe "validate!/1" do
    test "returns opts for valid configuration" do
      opts = [epochs: 10, batch_size: 64]
      assert Config.validate!(opts) == opts
    end

    test "raises ArgumentError for invalid configuration" do
      opts = [epochs: -1, batch_size: 64]

      assert_raise ArgumentError, ~r/Invalid training configuration/, fn ->
        Config.validate!(opts)
      end
    end

    test "error message includes all errors" do
      opts = [epochs: -1, batch_size: -32]
      error = catch_error(Config.validate!(opts))
      assert error.message =~ "epochs"
      assert error.message =~ "batch_size"
    end
  end

  describe "validation warnings" do
    import ExUnit.CaptureIO

    test "warns for large window_size" do
      opts = [epochs: 10, batch_size: 64, window_size: 150]

      output =
        capture_io(:stderr, fn ->
          Config.validate(opts)
        end)

      assert output =~ "window_size"
      assert output =~ "memory"
    end

    test "warns for large batch_size on CPU" do
      # Only warns on CPU (GPU handles large batches fine)
      # Skip if EXLA_TARGET=cuda is set
      if System.get_env("EXLA_TARGET") == "cuda" do
        opts = [epochs: 10, batch_size: 512]

        output =
          capture_io(:stderr, fn ->
            Config.validate(opts)
          end)

        # GPU: no warning expected
        refute output =~ "batch_size"
      else
        opts = [epochs: 10, batch_size: 512]

        output =
          capture_io(:stderr, fn ->
            Config.validate(opts)
          end)

        assert output =~ "batch_size"
        assert output =~ "memory"
      end
    end

    test "warns for many epochs without wandb" do
      opts = [epochs: 25, batch_size: 64, wandb: false]

      output =
        capture_io(:stderr, fn ->
          Config.validate(opts)
        end)

      assert output =~ "wandb"
    end

    test "no warning for many epochs with wandb" do
      opts = [epochs: 25, batch_size: 64, wandb: true]

      output =
        capture_io(:stderr, fn ->
          Config.validate(opts)
        end)

      refute output =~ "wandb"
    end

    test "warns for small window_size with temporal" do
      opts = [epochs: 10, batch_size: 64, temporal: true, backbone: :lstm, window_size: 20]

      output =
        capture_io(:stderr, fn ->
          Config.validate(opts)
        end)

      assert output =~ "temporal"
      assert output =~ "window_size"
    end

    test "warnings don't cause validation to fail" do
      # Large window (warning) but otherwise valid
      opts = [epochs: 10, batch_size: 64, window_size: 150]

      capture_io(:stderr, fn ->
        assert {:ok, _} = Config.validate(opts)
      end)
    end
  end

  describe "validate/1 integration with parse_args" do
    setup do
      # Create a temp directory for replays to satisfy validation
      tmp_dir =
        System.tmp_dir!() |> Path.join("config_test_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(tmp_dir)
      on_exit(fn -> File.rm_rf!(tmp_dir) end)
      %{tmp_dir: tmp_dir}
    end

    test "parsed args from preset are valid", %{tmp_dir: tmp_dir} do
      opts = Config.parse_args(["--preset", "full", "--replays", tmp_dir])
      assert {:ok, _} = Config.validate(opts)
    end

    test "parsed args with CLI overrides are valid", %{tmp_dir: tmp_dir} do
      opts =
        Config.parse_args([
          "--preset",
          "quick",
          "--epochs",
          "5",
          "--batch-size",
          "128",
          "--replays",
          tmp_dir
        ])

      assert {:ok, _} = Config.validate(opts)
    end

    test "parsed temporal args are valid", %{tmp_dir: tmp_dir} do
      opts =
        Config.parse_args([
          "--temporal",
          "--backbone",
          "mamba",
          "--window-size",
          "60",
          "--replays",
          tmp_dir
        ])

      assert {:ok, _} = Config.validate(opts)
    end
  end

  # ============================================================================
  # Early Stopping CLI Tests
  # ============================================================================

  describe "parse_args/1 with early stopping" do
    test "parses --early-stopping flag" do
      opts = Config.parse_args(["--early-stopping"])
      assert opts[:early_stopping] == true
    end

    test "early_stopping defaults to false" do
      opts = Config.parse_args([])
      assert opts[:early_stopping] == false
    end

    test "parses --patience" do
      opts = Config.parse_args(["--patience", "10"])
      assert opts[:patience] == 10
    end

    test "patience defaults to 5" do
      opts = Config.parse_args([])
      assert opts[:patience] == 5
    end

    test "parses --min-delta" do
      opts = Config.parse_args(["--min-delta", "0.05"])
      assert opts[:min_delta] == 0.05
    end

    test "min_delta defaults to 0.01" do
      opts = Config.parse_args([])
      assert opts[:min_delta] == 0.01
    end

    test "parses all early stopping options together" do
      opts = Config.parse_args(["--early-stopping", "--patience", "3", "--min-delta", "0.001"])
      assert opts[:early_stopping] == true
      assert opts[:patience] == 3
      assert opts[:min_delta] == 0.001
    end
  end

  describe "validate/1 with early stopping" do
    test "accepts valid early stopping config" do
      opts = [epochs: 10, batch_size: 64, early_stopping: true, patience: 5, min_delta: 0.01]
      assert {:ok, _} = Config.validate(opts)
    end

    test "returns error for negative patience" do
      opts = [epochs: 10, batch_size: 64, patience: -1]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "patience"))
    end

    test "returns error for zero patience" do
      opts = [epochs: 10, batch_size: 64, patience: 0]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "patience"))
    end

    test "returns error for negative min_delta" do
      opts = [epochs: 10, batch_size: 64, min_delta: -0.01]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "min_delta"))
    end

    test "returns error for zero min_delta" do
      opts = [epochs: 10, batch_size: 64, min_delta: 0.0]
      assert {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "min_delta"))
    end
  end

  describe "build_config_json/2 with early stopping" do
    test "includes early stopping fields" do
      opts = [
        epochs: 10,
        batch_size: 64,
        early_stopping: true,
        patience: 5,
        min_delta: 0.01,
        replays: "/path/to/replays",
        temporal: false,
        backbone: :mlp,
        hidden_sizes: [64, 64],
        window_size: 60,
        stride: 1,
        num_layers: 2,
        state_size: 16,
        expand_factor: 2,
        conv_size: 4,
        truncate_bptt: nil,
        precision: :bf16,
        frame_delay: 0,
        checkpoint: "test.axon"
      ]

      results = %{epochs_completed: 7, stopped_early: true}

      json = Config.build_config_json(opts, results)

      assert json[:early_stopping] == true
      assert json[:patience] == 5
      assert json[:min_delta] == 0.01
      assert json[:epochs_completed] == 7
      assert json[:stopped_early] == true
    end
  end

  describe "build_config_json/2 with provenance" do
    test "includes character and stage filters" do
      opts =
        Config.defaults()
        |> Keyword.put(:checkpoint, "test.axon")
        |> Keyword.put(:characters, [:mewtwo, :fox])
        |> Keyword.put(:stages, [:battlefield, :fd])

      json = Config.build_config_json(opts)

      assert json[:characters] == ["mewtwo", "fox"]
      assert json[:stages] == ["battlefield", "fd"]
    end

    test "handles empty character/stage filters" do
      opts =
        Config.defaults()
        |> Keyword.put(:checkpoint, "test.axon")
        |> Keyword.put(:characters, [])
        |> Keyword.put(:stages, [])

      json = Config.build_config_json(opts)

      # Empty lists become nil for cleaner JSON
      assert json[:characters] == nil
      assert json[:stages] == nil
    end

    test "includes replay manifest fields" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")

      results = %{
        replay_count: 150,
        replay_files: ["replay1.slp", "replay2.slp"],
        replay_manifest_hash: "sha256:abc123",
        character_distribution: %{"mewtwo" => 100, "fox" => 50}
      }

      json = Config.build_config_json(opts, results)

      assert json[:replay_count] == 150
      assert json[:replay_files] == ["replay1.slp", "replay2.slp"]
      assert json[:replay_manifest_hash] == "sha256:abc123"
      assert json[:character_distribution] == %{"mewtwo" => 100, "fox" => 50}
    end

    test "handles nil replay manifest fields" do
      opts = Config.defaults() |> Keyword.put(:checkpoint, "test.axon")
      json = Config.build_config_json(opts, %{})

      assert json[:replay_count] == nil
      assert json[:replay_files] == nil
      assert json[:replay_manifest_hash] == nil
      assert json[:character_distribution] == nil
    end
  end

  describe "compute_manifest_hash/1" do
    test "returns nil for empty list" do
      assert Config.compute_manifest_hash([]) == nil
    end

    test "computes consistent hash for same files" do
      files = ["/path/to/a.slp", "/path/to/b.slp", "/path/to/c.slp"]

      hash1 = Config.compute_manifest_hash(files)
      hash2 = Config.compute_manifest_hash(files)

      assert hash1 == hash2
      assert String.starts_with?(hash1, "sha256:")
    end

    test "same hash regardless of input order" do
      files_a = ["/path/to/a.slp", "/path/to/b.slp", "/path/to/c.slp"]
      files_b = ["/path/to/c.slp", "/path/to/a.slp", "/path/to/b.slp"]

      assert Config.compute_manifest_hash(files_a) == Config.compute_manifest_hash(files_b)
    end

    test "different files produce different hashes" do
      files_a = ["/path/to/a.slp", "/path/to/b.slp"]
      files_b = ["/path/to/a.slp", "/path/to/c.slp"]

      assert Config.compute_manifest_hash(files_a) != Config.compute_manifest_hash(files_b)
    end

    test "hash format is sha256:hex" do
      hash = Config.compute_manifest_hash(["/path/to/test.slp"])

      assert String.starts_with?(hash, "sha256:")
      # SHA256 produces 64 hex characters
      # "sha256:" + 64 hex chars
      assert String.length(hash) == 7 + 64
    end
  end

  # ============================================================================
  # Checkpointing CLI Tests
  # ============================================================================

  describe "parse_args/1 with checkpointing" do
    test "save_best defaults to true" do
      opts = Config.parse_args([])
      assert opts[:save_best] == true
    end

    test "parses --save-best flag" do
      opts = Config.parse_args(["--save-best"])
      assert opts[:save_best] == true
    end

    test "save_every defaults to nil" do
      opts = Config.parse_args([])
      assert opts[:save_every] == nil
    end

    test "parses --save-every" do
      opts = Config.parse_args(["--save-every", "5"])
      assert opts[:save_every] == 5
    end
  end

  describe "derive_best_checkpoint_path/1" do
    test "derives best checkpoint path" do
      assert Config.derive_best_checkpoint_path("checkpoints/mlp_20260119.axon") ==
               "checkpoints/mlp_20260119_best.axon"
    end

    test "handles nested paths" do
      assert Config.derive_best_checkpoint_path("checkpoints/mewtwo/mamba_20260119.axon") ==
               "checkpoints/mewtwo/mamba_20260119_best.axon"
    end

    test "returns nil for nil" do
      assert Config.derive_best_checkpoint_path(nil) == nil
    end
  end

  describe "derive_best_policy_path/1" do
    test "derives best policy path" do
      assert Config.derive_best_policy_path("checkpoints/mlp_20260119.axon") ==
               "checkpoints/mlp_20260119_best_policy.bin"
    end

    test "handles nested paths" do
      assert Config.derive_best_policy_path("checkpoints/mewtwo/mamba_20260119.axon") ==
               "checkpoints/mewtwo/mamba_20260119_best_policy.bin"
    end

    test "returns nil for nil" do
      assert Config.derive_best_policy_path(nil) == nil
    end
  end

  # ============================================================================
  # Learning Rate Scheduling CLI Tests
  # ============================================================================

  describe "parse_args/1 with learning rate scheduling" do
    test "learning_rate defaults to 1.0e-4" do
      opts = Config.parse_args([])
      assert opts[:learning_rate] == 1.0e-4
    end

    test "parses --lr" do
      opts = Config.parse_args(["--lr", "0.001"])
      assert opts[:learning_rate] == 0.001
    end

    test "parses --lr with scientific notation" do
      opts = Config.parse_args(["--lr", "5.0e-5"])
      assert opts[:learning_rate] == 5.0e-5
    end

    test "lr_schedule defaults to :constant" do
      opts = Config.parse_args([])
      assert opts[:lr_schedule] == :constant
    end

    test "parses --lr-schedule constant" do
      opts = Config.parse_args(["--lr-schedule", "constant"])
      assert opts[:lr_schedule] == :constant
    end

    test "parses --lr-schedule cosine" do
      opts = Config.parse_args(["--lr-schedule", "cosine"])
      assert opts[:lr_schedule] == :cosine
    end

    test "parses --lr-schedule exponential" do
      opts = Config.parse_args(["--lr-schedule", "exponential"])
      assert opts[:lr_schedule] == :exponential
    end

    test "parses --lr-schedule linear" do
      opts = Config.parse_args(["--lr-schedule", "linear"])
      assert opts[:lr_schedule] == :linear
    end

    test "warmup_steps defaults to 1 (Polaris bug workaround)" do
      opts = Config.parse_args([])
      assert opts[:warmup_steps] == 1
    end

    test "parses --warmup-steps" do
      opts = Config.parse_args(["--warmup-steps", "1000"])
      assert opts[:warmup_steps] == 1000
    end

    test "decay_steps defaults to nil" do
      opts = Config.parse_args([])
      assert opts[:decay_steps] == nil
    end

    test "parses --decay-steps" do
      opts = Config.parse_args(["--decay-steps", "10000"])
      assert opts[:decay_steps] == 10000
    end

    test "parses full learning rate config" do
      opts =
        Config.parse_args([
          "--lr",
          "0.0005",
          "--lr-schedule",
          "cosine",
          "--warmup-steps",
          "500",
          "--decay-steps",
          "5000"
        ])

      assert opts[:learning_rate] == 0.0005
      assert opts[:lr_schedule] == :cosine
      assert opts[:warmup_steps] == 500
      assert opts[:decay_steps] == 5000
    end
  end

  describe "validate/1 with learning rate options" do
    test "accepts valid learning rate config" do
      opts = [
        epochs: 10,
        batch_size: 64,
        learning_rate: 0.001,
        lr_schedule: :cosine,
        warmup_steps: 100
      ]

      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns error for invalid lr_schedule" do
      opts = [epochs: 10, batch_size: 64, lr_schedule: :invalid]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "lr_schedule"))
    end

    test "returns error for negative learning_rate" do
      opts = [epochs: 10, batch_size: 64, learning_rate: -0.001]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "learning_rate"))
    end

    test "returns error for zero learning_rate" do
      opts = [epochs: 10, batch_size: 64, learning_rate: 0.0]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "learning_rate"))
    end

    test "returns error for negative warmup_steps" do
      opts = [epochs: 10, batch_size: 64, warmup_steps: -100]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "warmup_steps"))
    end
  end

  # ============================================================================
  # Training Resumption CLI Tests
  # ============================================================================

  describe "parse_args/1 with resume" do
    test "resume defaults to nil" do
      opts = Config.parse_args([])
      assert opts[:resume] == nil
    end

    test "parses --resume" do
      opts = Config.parse_args(["--resume", "/path/to/checkpoint.axon"])
      assert opts[:resume] == "/path/to/checkpoint.axon"
    end
  end

  describe "validate/1 with resume" do
    test "accepts nil resume" do
      opts = [epochs: 10, batch_size: 64, resume: nil]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns error for non-existent resume checkpoint" do
      opts = [epochs: 10, batch_size: 64, resume: "/nonexistent/path.axon"]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "resume checkpoint does not exist"))
    end

    test "accepts existing resume checkpoint" do
      # Create a temporary file to simulate an existing checkpoint
      path = Path.join(System.tmp_dir!(), "test_resume_#{:rand.uniform(10000)}.axon")
      File.write!(path, "test")

      try do
        opts = [epochs: 10, batch_size: 64, resume: path]
        assert {:ok, ^opts} = Config.validate(opts)
      after
        File.rm(path)
      end
    end
  end

  # ============================================================================
  # Gradient Accumulation Tests
  # ============================================================================

  describe "parse_args/1 with accumulation" do
    test "defaults accumulation_steps to 1" do
      opts = Config.parse_args([])
      assert opts[:accumulation_steps] == 1
    end

    test "parses --accumulation-steps" do
      opts = Config.parse_args(["--accumulation-steps", "4"])
      assert opts[:accumulation_steps] == 4
    end
  end

  describe "validate/1 with accumulation" do
    test "accepts accumulation_steps of 1" do
      opts = [epochs: 10, batch_size: 64, accumulation_steps: 1]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "accepts accumulation_steps greater than 1" do
      opts = [epochs: 10, batch_size: 64, accumulation_steps: 4]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns error for accumulation_steps of 0" do
      opts = [epochs: 10, batch_size: 64, accumulation_steps: 0]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "accumulation_steps must be positive"))
    end

    test "returns error for negative accumulation_steps" do
      opts = [epochs: 10, batch_size: 64, accumulation_steps: -1]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "accumulation_steps must be positive"))
    end
  end

  # ============================================================================
  # Validation Split Tests
  # ============================================================================

  describe "parse_args/1 with val_split" do
    test "defaults val_split to 0.0" do
      opts = Config.parse_args([])
      assert opts[:val_split] == 0.0
    end

    test "parses --val-split" do
      opts = Config.parse_args(["--val-split", "0.1"])
      assert opts[:val_split] == 0.1
    end
  end

  describe "validate/1 with val_split" do
    test "accepts val_split of 0.0 (no validation)" do
      opts = [epochs: 10, batch_size: 64, val_split: 0.0]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "accepts val_split in valid range" do
      opts = [epochs: 10, batch_size: 64, val_split: 0.2]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "accepts val_split close to 1.0 but not equal" do
      opts = [epochs: 10, batch_size: 64, val_split: 0.99]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns error for val_split of 1.0" do
      opts = [epochs: 10, batch_size: 64, val_split: 1.0]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "val_split must be in"))
    end

    test "returns error for val_split greater than 1.0" do
      opts = [epochs: 10, batch_size: 64, val_split: 1.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "val_split must be in"))
    end

    test "returns error for negative val_split" do
      opts = [epochs: 10, batch_size: 64, val_split: -0.1]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "val_split must be in"))
    end
  end

  describe "parse_args/1 with label_smoothing" do
    test "defaults label_smoothing to 0.1" do
      opts = Config.parse_args([])
      assert opts[:label_smoothing] == 0.1
    end

    test "parses --label-smoothing" do
      opts = Config.parse_args(["--label-smoothing", "0.1"])
      assert opts[:label_smoothing] == 0.1
    end

    test "parses --label-smoothing with small value" do
      opts = Config.parse_args(["--label-smoothing", "0.05"])
      assert opts[:label_smoothing] == 0.05
    end
  end

  describe "validate/1 with label_smoothing" do
    test "accepts label_smoothing of 0.0 (no smoothing)" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 0.0]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "accepts label_smoothing in valid range" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 0.1]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "accepts label_smoothing close to 1.0 but not equal" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 0.99]
      assert {:ok, ^opts} = Config.validate(opts)
    end

    test "returns error for label_smoothing of 1.0" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 1.0]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "label_smoothing must be in"))
    end

    test "returns error for label_smoothing greater than 1.0" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 1.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "label_smoothing must be in"))
    end

    test "returns error for negative label_smoothing" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: -0.1]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "label_smoothing must be in"))
    end
  end

  describe "parse_args/1 with stage_mode" do
    test "defaults stage_mode to :one_hot_compact" do
      opts = Config.parse_args([])
      assert opts[:stage_mode] == :one_hot_compact
    end

    test "parses --stage-mode full" do
      opts = Config.parse_args(["--stage-mode", "full"])
      assert opts[:stage_mode] == :one_hot_full
    end

    test "parses --stage-mode compact" do
      opts = Config.parse_args(["--stage-mode", "compact"])
      assert opts[:stage_mode] == :one_hot_compact
    end

    test "parses --stage-mode learned" do
      opts = Config.parse_args(["--stage-mode", "learned"])
      assert opts[:stage_mode] == :learned
    end

    test "parses --stage-mode one_hot_full" do
      opts = Config.parse_args(["--stage-mode", "one_hot_full"])
      assert opts[:stage_mode] == :one_hot_full
    end

    test "parses --stage-mode one_hot_compact" do
      opts = Config.parse_args(["--stage-mode", "one_hot_compact"])
      assert opts[:stage_mode] == :one_hot_compact
    end
  end

  describe "parse_args/1 with num_player_names" do
    test "defaults num_player_names to 112 for backwards compatibility" do
      opts = Config.parse_args([])
      assert opts[:num_player_names] == 112
    end

    test "parses --num-player-names 0 to disable player name embedding" do
      opts = Config.parse_args(["--num-player-names", "0"])
      assert opts[:num_player_names] == 0
    end

    test "parses custom --num-player-names value" do
      opts = Config.parse_args(["--num-player-names", "64"])
      assert opts[:num_player_names] == 64
    end
  end

  describe "validate_args/1 CLI argument validation" do
    test "returns empty warnings for valid args" do
      assert {:ok, []} = Config.validate_args(["--epochs", "10", "--batch-size", "32"])
    end

    test "returns empty warnings for valid flags without values" do
      assert {:ok, []} = Config.validate_args(["--wandb", "--temporal", "--focal-loss"])
    end

    test "suggests correction for typo --ephocs -> --epochs" do
      {:ok, warnings} = Config.validate_args(["--ephocs", "10"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "Unknown flag '--ephocs'"
      assert hd(warnings) =~ "--epochs"
    end

    test "suggests correction for typo --batchsize -> --batch-size" do
      {:ok, warnings} = Config.validate_args(["--batchsize", "32"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "--batch-size"
    end

    test "suggests correction for typo --temporel -> --temporal" do
      {:ok, warnings} = Config.validate_args(["--temporel"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "--temporal"
    end

    test "suggests correction for typo --wanb -> --wandb" do
      {:ok, warnings} = Config.validate_args(["--wanb"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "--wandb"
    end

    test "suggests correction for typo --focalloss -> --focal-loss" do
      {:ok, warnings} = Config.validate_args(["--focalloss"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "--focal-loss"
    end

    test "returns generic message for completely unrecognized flag" do
      {:ok, warnings} = Config.validate_args(["--xyzabc123"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "Unknown flag '--xyzabc123'"
      assert hd(warnings) =~ "--help"
      refute hd(warnings) =~ "Did you mean"
    end

    test "detects multiple typos" do
      {:ok, warnings} = Config.validate_args(["--ephocs", "10", "--batchsize", "32"])
      assert length(warnings) == 2
    end

    test "ignores non-flag arguments" do
      # Values like "10" and "32" should not trigger warnings
      assert {:ok, []} = Config.validate_args(["--epochs", "10", "replay.slp"])
    end

    test "deduplicates repeated invalid flags" do
      {:ok, warnings} = Config.validate_args(["--ephocs", "10", "--ephocs", "20"])
      # Should only warn once, not twice
      assert length(warnings) == 1
    end

    test "valid_flags/0 returns all known flags" do
      flags = Config.valid_flags()
      assert "--epochs" in flags
      assert "--batch-size" in flags
      assert "--temporal" in flags
      assert "--focal-loss" in flags
      assert "--preset" in flags
      assert "--config" in flags
    end
  end

  # ============================================================================
  # YAML Config Tests
  # ============================================================================

  describe "parse_yaml/1" do
    test "parses basic YAML config" do
      yaml = """
      epochs: 20
      batch_size: 128
      temporal: true
      """

      {:ok, opts} = Config.parse_yaml(yaml)

      assert opts[:epochs] == 20
      assert opts[:batch_size] == 128
      assert opts[:temporal] == true
    end

    test "parses hidden_sizes as list" do
      yaml = """
      hidden_sizes: [256, 256]
      """

      {:ok, opts} = Config.parse_yaml(yaml)
      assert opts[:hidden_sizes] == [256, 256]
    end

    test "converts backbone to atom" do
      yaml = """
      backbone: mamba
      lr_schedule: cosine
      optimizer: adamw
      """

      {:ok, opts} = Config.parse_yaml(yaml)

      assert opts[:backbone] == :mamba
      assert opts[:lr_schedule] == :cosine
      assert opts[:optimizer] == :adamw
    end

    test "handles kebab-case keys" do
      yaml = """
      batch-size: 64
      learning-rate: 0.001
      lr-schedule: constant
      """

      {:ok, opts} = Config.parse_yaml(yaml)

      assert opts[:batch_size] == 64
      assert opts[:learning_rate] == 0.001
      assert opts[:lr_schedule] == :constant
    end

    test "converts characters list to atoms" do
      yaml = """
      characters: [mewtwo, fox, falco]
      """

      {:ok, opts} = Config.parse_yaml(yaml)
      assert opts[:characters] == [:mewtwo, :fox, :falco]
    end

    test "returns error for invalid YAML" do
      yaml = """
      epochs: [unclosed bracket
      """

      assert {:error, _} = Config.parse_yaml(yaml)
    end
  end

  describe "load_yaml/1" do
    @test_dir Path.join(System.tmp_dir!(), "config_test_#{:erlang.unique_integer()}")

    setup do
      File.mkdir_p!(@test_dir)
      on_exit(fn -> File.rm_rf!(@test_dir) end)
      :ok
    end

    test "loads config from file" do
      path = Path.join(@test_dir, "config.yaml")

      File.write!(path, """
      epochs: 50
      temporal: true
      backbone: mamba
      """)

      {:ok, opts} = Config.load_yaml(path)

      assert opts[:epochs] == 50
      assert opts[:temporal] == true
      assert opts[:backbone] == :mamba
    end

    test "returns error for missing file" do
      {:error, %YamlError{reason: :file_not_found}} = Config.load_yaml("/nonexistent/config.yaml")
    end
  end

  describe "save_yaml/2" do
    @test_dir Path.join(System.tmp_dir!(), "config_save_test_#{:erlang.unique_integer()}")

    setup do
      File.mkdir_p!(@test_dir)
      on_exit(fn -> File.rm_rf!(@test_dir) end)
      :ok
    end

    test "saves config to YAML file" do
      path = Path.join(@test_dir, "output.yaml")
      opts = [epochs: 10, batch_size: 64, temporal: true]

      :ok = Config.save_yaml(opts, path)

      content = File.read!(path)
      assert content =~ "epochs: 10"
      assert content =~ "batch-size: 64"
      assert content =~ "temporal: true"
    end
  end

  describe "parse_args with --config" do
    @test_dir Path.join(System.tmp_dir!(), "config_args_test_#{:erlang.unique_integer()}")

    setup do
      File.mkdir_p!(@test_dir)
      on_exit(fn -> File.rm_rf!(@test_dir) end)
      :ok
    end

    test "loads config file and merges with CLI args" do
      path = Path.join(@test_dir, "config.yaml")

      File.write!(path, """
      epochs: 50
      batch_size: 256
      temporal: true
      """)

      # CLI args should override YAML
      opts = Config.parse_args(["--config", path, "--epochs", "10"])

      # CLI override
      assert opts[:epochs] == 10
      # From YAML
      assert opts[:batch_size] == 256
      # From YAML
      assert opts[:temporal] == true
    end
  end

  # ============================================================================
  # Verbosity Tests
  # ============================================================================

  describe "parse_args/1 with verbosity flags" do
    test "defaults verbosity to 1 (normal)" do
      opts = Config.parse_args([])
      assert opts[:verbosity] == 1
    end

    test "--quiet sets verbosity to 0" do
      opts = Config.parse_args(["--quiet"])
      assert opts[:verbosity] == 0
    end

    test "--verbose sets verbosity to 2" do
      opts = Config.parse_args(["--verbose"])
      assert opts[:verbosity] == 2
    end

    test "--verbose takes precedence over default" do
      opts = Config.parse_args(["--verbose", "--epochs", "5"])
      assert opts[:verbosity] == 2
      assert opts[:epochs] == 5
    end
  end

  # ============================================================================
  # Seed Tests
  # ============================================================================

  describe "parse_args/1 with seed" do
    test "defaults seed to nil" do
      opts = Config.parse_args([])
      assert opts[:seed] == nil
    end

    test "--seed sets explicit seed" do
      opts = Config.parse_args(["--seed", "12345"])
      assert opts[:seed] == 12345
    end
  end

  describe "init_seed/1" do
    test "returns the seed it was given" do
      seed = Config.init_seed(42)
      assert seed == 42
    end

    test "generates a seed when given nil" do
      seed = Config.init_seed(nil)
      assert is_integer(seed)
      assert seed > 0
    end

    test "generated seeds are different" do
      seed1 = Config.init_seed(nil)
      # Reset entropy
      :rand.seed(:exsss)
      seed2 = Config.init_seed(nil)
      # Seeds should generally be different (probabilistically)
      # This is a weak test but catches obvious bugs
      assert is_integer(seed1)
      assert is_integer(seed2)
    end
  end

  # ============================================================================
  # Checkpoint Safety Tests
  # ============================================================================

  describe "parse_args/1 with checkpoint safety flags" do
    test "defaults overwrite to false" do
      opts = Config.parse_args([])
      assert opts[:overwrite] == false
    end

    test "--overwrite enables overwrite" do
      opts = Config.parse_args(["--overwrite"])
      assert opts[:overwrite] == true
    end

    test "--no-overwrite explicitly disables overwrite" do
      opts = Config.parse_args(["--no-overwrite"])
      assert opts[:overwrite] == false
    end

    test "defaults backup to true" do
      opts = Config.parse_args([])
      assert opts[:backup] == true
    end

    test "--no-backup disables backup" do
      opts = Config.parse_args(["--no-backup"])
      assert opts[:backup] == false
    end

    test "defaults backup_count to 3" do
      opts = Config.parse_args([])
      assert opts[:backup_count] == 3
    end

    test "--backup-count sets custom count" do
      opts = Config.parse_args(["--backup-count", "5"])
      assert opts[:backup_count] == 5
    end
  end

  describe "check_checkpoint_path/2" do
    @test_dir Path.join(System.tmp_dir!(), "checkpoint_test_#{:erlang.unique_integer()}")

    setup do
      File.mkdir_p!(@test_dir)
      on_exit(fn -> File.rm_rf!(@test_dir) end)
      :ok
    end

    test "returns {:ok, :new} for non-existent path" do
      path = Path.join(@test_dir, "new_model.axon")
      assert {:ok, :new} = Config.check_checkpoint_path(path)
    end

    test "returns {:error, :exists, info} for existing path without overwrite" do
      path = Path.join(@test_dir, "existing.axon")
      File.write!(path, "test content")

      assert {:error, :exists, info} = Config.check_checkpoint_path(path)
      assert info.path == path
      assert info.size > 0
    end

    test "returns {:ok, :overwrite, info} for existing path with overwrite" do
      path = Path.join(@test_dir, "existing.axon")
      File.write!(path, "test content")

      assert {:ok, :overwrite, info} = Config.check_checkpoint_path(path, overwrite: true)
      assert info.path == path
    end
  end

  describe "backup_checkpoint/2" do
    @test_dir Path.join(System.tmp_dir!(), "backup_test_#{:erlang.unique_integer()}")

    setup do
      File.mkdir_p!(@test_dir)
      on_exit(fn -> File.rm_rf!(@test_dir) end)
      :ok
    end

    test "creates .bak file" do
      path = Path.join(@test_dir, "model.axon")
      File.write!(path, "original content")

      assert {:ok, backup_path} = Config.backup_checkpoint(path)
      assert backup_path == "#{path}.bak"
      assert File.exists?(backup_path)
      assert File.read!(backup_path) == "original content"
    end

    test "rotates existing backups" do
      path = Path.join(@test_dir, "model.axon")
      File.write!(path, "v3")
      File.write!("#{path}.bak", "v2")
      File.write!("#{path}.bak.1", "v1")

      assert {:ok, _} = Config.backup_checkpoint(path, backup_count: 3)

      assert File.read!("#{path}.bak") == "v3"
      assert File.read!("#{path}.bak.1") == "v2"
      assert File.read!("#{path}.bak.2") == "v1"
    end

    test "returns {:ok, nil} for non-existent file" do
      path = Path.join(@test_dir, "nonexistent.axon")
      assert {:ok, nil} = Config.backup_checkpoint(path)
    end
  end

  describe "format_file_info/1" do
    test "formats bytes correctly" do
      info = %{path: "model.axon", size: 1024, modified: {{2026, 1, 23}, {14, 30, 0}}}
      result = Config.format_file_info(info)
      assert result =~ "1.0 KB"
      assert result =~ "2026-01-23 14:30:00"
    end

    test "formats megabytes correctly" do
      info = %{path: "model.axon", size: 45_200_000, modified: {{2026, 1, 23}, {14, 30, 0}}}
      result = Config.format_file_info(info)
      assert result =~ "43.1 MB"
    end
  end

  # ============================================================================
  # Environment Variable Tests
  # ============================================================================

  describe "environment variable defaults" do
    test "EXPHIL_REPLAYS_DIR overrides default replays path" do
      # Save original
      original = System.get_env("EXPHIL_REPLAYS_DIR")

      try do
        System.put_env("EXPHIL_REPLAYS_DIR", "/custom/replays")
        opts = Config.defaults()
        assert opts[:replays] == "/custom/replays"
      after
        # Restore
        if original,
          do: System.put_env("EXPHIL_REPLAYS_DIR", original),
          else: System.delete_env("EXPHIL_REPLAYS_DIR")
      end
    end

    test "EXPHIL_WANDB_PROJECT overrides default wandb project" do
      original = System.get_env("EXPHIL_WANDB_PROJECT")

      try do
        System.put_env("EXPHIL_WANDB_PROJECT", "my-project")
        opts = Config.defaults()
        assert opts[:wandb_project] == "my-project"
      after
        if original,
          do: System.put_env("EXPHIL_WANDB_PROJECT", original),
          else: System.delete_env("EXPHIL_WANDB_PROJECT")
      end
    end

    test "CLI args override environment variables" do
      original = System.get_env("EXPHIL_REPLAYS_DIR")

      try do
        System.put_env("EXPHIL_REPLAYS_DIR", "/env/replays")
        opts = Config.parse_args(["--replays", "/cli/replays"])
        assert opts[:replays] == "/cli/replays"
      after
        if original,
          do: System.put_env("EXPHIL_REPLAYS_DIR", original),
          else: System.delete_env("EXPHIL_REPLAYS_DIR")
      end
    end
  end

  # ============================================================================
  # Negative Path Tests - YAML Parsing
  # ============================================================================

  describe "YAML parsing errors" do
    test "load_yaml returns error for non-existent file" do
      assert {:error, %YamlError{reason: :file_not_found}} = Config.load_yaml("/nonexistent/config.yaml")
    end

    test "parse_yaml handles malformed YAML" do
      malformed = """
      epochs: 10
      batch_size: [unclosed
      """

      assert {:error, _reason} = Config.parse_yaml(malformed)
    end

    test "parse_yaml handles empty string" do
      assert {:ok, []} = Config.parse_yaml("")
    end

    test "parse_yaml handles non-map YAML" do
      # YAML that parses to a list instead of map
      list_yaml = """
      - item1
      - item2
      """

      assert {:error, %YamlError{reason: :invalid_format}} = Config.parse_yaml(list_yaml)
    end

    test "parse_yaml handles scalar YAML" do
      scalar_yaml = "just a string"
      assert {:error, %YamlError{reason: :invalid_format}} = Config.parse_yaml(scalar_yaml)
    end
  end

  # ============================================================================
  # Negative Path Tests - Argument Validation
  # ============================================================================

  describe "validate_args/1 error cases" do
    test "detects typos and suggests corrections" do
      {:ok, warnings} = Config.validate_args(["--ephocs", "10"])
      assert length(warnings) == 1
      assert hd(warnings) =~ "Did you mean '--epochs'"
    end

    test "detects multiple typos" do
      {:ok, warnings} = Config.validate_args(["--ephocs", "10", "--bach-size", "32"])
      assert length(warnings) == 2
    end

    test "suggests closest flag for near-matches" do
      {:ok, warnings} = Config.validate_args(["--batchsize", "32"])
      assert hd(warnings) =~ "--batch-size"
    end

    test "returns empty warnings for valid flags" do
      {:ok, warnings} = Config.validate_args(["--epochs", "10", "--batch-size", "32"])
      assert warnings == []
    end

    test "ignores flag values, only validates flags" do
      {:ok, warnings} = Config.validate_args(["--epochs", "invalid_number"])
      # The flag --epochs is valid, parsing the value happens elsewhere
      assert warnings == []
    end
  end

  # ============================================================================
  # Negative Path Tests - Frame Delay Validation
  # ============================================================================

  describe "validate/1 frame delay range" do
    test "returns error when frame_delay_min > frame_delay_max" do
      opts = [epochs: 10, batch_size: 64, frame_delay_min: 20, frame_delay_max: 10]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "frame_delay_min"))
    end

    test "returns error for unusually high frame_delay_max" do
      opts = [epochs: 10, batch_size: 64, frame_delay_max: 100]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "frame_delay_max"))
    end

    test "accepts valid frame_delay range" do
      opts = [epochs: 10, batch_size: 64, frame_delay_min: 0, frame_delay_max: 18]
      assert {:ok, _} = Config.validate(opts)
    end
  end

  # ============================================================================
  # Negative Path Tests - Optimizer Validation
  # ============================================================================

  describe "validate/1 optimizer" do
    test "returns error for invalid optimizer" do
      opts = [epochs: 10, batch_size: 64, optimizer: :invalid_optimizer]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "optimizer"))
    end

    test "accepts all valid optimizers" do
      for optimizer <- [:adam, :adamw, :lamb, :radam, :sgd, :rmsprop] do
        opts = [epochs: 10, batch_size: 64, optimizer: optimizer]
        assert {:ok, _} = Config.validate(opts), "optimizer #{optimizer} should be valid"
      end
    end
  end

  # ============================================================================
  # Negative Path Tests - Regularization Validation
  # ============================================================================

  describe "validate/1 regularization options" do
    test "returns error for label_smoothing out of range" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: 1.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "label_smoothing"))
    end

    test "returns error for negative label_smoothing" do
      opts = [epochs: 10, batch_size: 64, label_smoothing: -0.1]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "label_smoothing"))
    end

    test "returns error for ema_decay out of range (too high)" do
      opts = [epochs: 10, batch_size: 64, ema_decay: 1.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "ema_decay"))
    end

    test "returns error for ema_decay out of range (too low)" do
      opts = [epochs: 10, batch_size: 64, ema_decay: -0.1]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "ema_decay"))
    end

    test "returns error for mirror_prob out of range" do
      opts = [epochs: 10, batch_size: 64, mirror_prob: 1.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "mirror_prob"))
    end

    test "returns error for noise_prob out of range" do
      opts = [epochs: 10, batch_size: 64, noise_prob: -0.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "noise_prob"))
    end
  end

  # ============================================================================
  # Negative Path Tests - Cosine Restarts Validation
  # ============================================================================

  describe "validate/1 cosine restarts" do
    test "returns error for restart_mult less than 1" do
      opts = [epochs: 10, batch_size: 64, restart_mult: 0.5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "restart_mult"))
    end

    test "accepts restart_mult of exactly 1" do
      opts = [epochs: 10, batch_size: 64, restart_mult: 1.0]
      assert {:ok, _} = Config.validate(opts)
    end

    test "accepts restart_mult greater than 1" do
      opts = [epochs: 10, batch_size: 64, restart_mult: 2.0]
      assert {:ok, _} = Config.validate(opts)
    end
  end

  # ============================================================================
  # Negative Path Tests - Gradient Clipping Validation
  # ============================================================================

  describe "validate/1 gradient clipping" do
    test "returns error for negative max_grad_norm" do
      opts = [epochs: 10, batch_size: 64, max_grad_norm: -1.0]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "max_grad_norm"))
    end

    test "accepts zero max_grad_norm (disables clipping)" do
      opts = [epochs: 10, batch_size: 64, max_grad_norm: 0.0]
      assert {:ok, _} = Config.validate(opts)
    end

    test "accepts positive max_grad_norm" do
      opts = [epochs: 10, batch_size: 64, max_grad_norm: 1.0]
      assert {:ok, _} = Config.validate(opts)
    end
  end

  # ============================================================================
  # Negative Path Tests - Streaming Validation
  # ============================================================================

  describe "validate/1 streaming options" do
    test "returns error for invalid stream_chunk_size" do
      opts = [epochs: 10, batch_size: 64, stream_chunk_size: -5]
      {:error, errors} = Config.validate(opts)
      assert Enum.any?(errors, &String.contains?(&1, "stream_chunk_size"))
    end

    test "accepts nil stream_chunk_size" do
      opts = [epochs: 10, batch_size: 64, stream_chunk_size: nil]
      assert {:ok, _} = Config.validate(opts)
    end

    test "accepts positive stream_chunk_size" do
      opts = [epochs: 10, batch_size: 64, stream_chunk_size: 100]
      assert {:ok, _} = Config.validate(opts)
    end
  end

  # ============================================================================
  # Smart Defaults Inference Tests
  # ============================================================================

  describe "infer_smart_defaults/1" do
    test "auto-enables temporal for LSTM backbone" do
      opts = [backbone: :lstm, temporal: false]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      assert new_opts[:temporal] == true
      assert length(inferences) == 1
      assert hd(inferences) =~ "LSTM"
    end

    test "auto-enables temporal for Mamba backbone" do
      opts = [backbone: :mamba, temporal: false]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      assert new_opts[:temporal] == true
      assert Enum.any?(inferences, &(&1 =~ "MAMBA"))
    end

    test "auto-sets val_split for early_stopping when val_split not specified" do
      # Don't include val_split - let the function see it as unset
      opts = [early_stopping: true]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      assert new_opts[:val_split] == 0.1
      assert Enum.any?(inferences, &(&1 =~ "val-split"))
    end

    test "respects explicit val_split of 0 (no auto-set)" do
      # When user explicitly includes val_split in opts, respect it
      # even if it's 0.0 - the key's presence indicates intent
      opts = [early_stopping: true, val_split: 0.0]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      # val_split key is present, so treated as explicit user choice
      assert new_opts[:val_split] == 0.0
      refute Enum.any?(inferences, &(&1 =~ "val-split"))
    end

    test "disables cache with augmentation" do
      opts = [augment: true, cache_embeddings: true, cache_augmented: false]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      assert new_opts[:cache_embeddings] == false
      assert Enum.any?(inferences, &(&1 =~ "cache"))
    end

    test "does not disable cache when cache_augmented is true" do
      opts = [augment: true, cache_embeddings: true, cache_augmented: true]
      {_new_opts, inferences} = Config.infer_smart_defaults(opts)

      # Should not disable cache since cache_augmented handles it
      refute Enum.any?(inferences, &(&1 =~ "cache-embeddings"))
    end

    test "returns empty inferences when no changes needed" do
      opts = [backbone: :mlp, temporal: false]
      {new_opts, inferences} = Config.infer_smart_defaults(opts)

      assert new_opts[:temporal] == false
      assert inferences == []
    end
  end
end
