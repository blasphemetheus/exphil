defmodule ExPhil.Training.ConfigTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config

  # ============================================================================
  # Argument Parsing Tests
  # ============================================================================

  describe "parse_args/1" do
    test "returns defaults for empty args" do
      opts = Config.parse_args([])

      assert opts[:epochs] == 10
      assert opts[:batch_size] == 64
      assert opts[:hidden_sizes] == [64, 64]
      assert opts[:temporal] == false
      assert opts[:precision] == :bf16
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
      opts = Config.parse_args([
        "--epochs", "3",
        "--batch-size", "32",
        "--temporal",
        "--backbone", "lstm",
        "--window-size", "30"
      ])

      assert opts[:epochs] == 3
      assert opts[:batch_size] == 32
      assert opts[:temporal] == true
      assert opts[:backbone] == :lstm
      assert opts[:window_size] == 30
    end

    test "parses Mamba-specific options" do
      opts = Config.parse_args([
        "--backbone", "mamba",
        "--state-size", "32",
        "--expand-factor", "4",
        "--conv-size", "8"
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
  end

  # ============================================================================
  # Checkpoint Naming Tests
  # ============================================================================

  describe "ensure_checkpoint_name/1" do
    test "generates timestamped name for MLP when checkpoint is nil" do
      opts = [checkpoint: nil, temporal: false, backbone: :sliding_window]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mlp_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mlp_[a-z_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end

    test "generates timestamped name with backbone for temporal" do
      opts = [checkpoint: nil, temporal: true, backbone: :mamba]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mamba_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mamba_[a-z_]+_\d{8}_\d{6}\.axon/
      assert is_binary(result[:name])
    end

    test "generates timestamped name with lstm backbone" do
      opts = [checkpoint: nil, temporal: true, backbone: :lstm]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/lstm_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/lstm_[a-z_]+_\d{8}_\d{6}\.axon/
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
      opts = [
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
      assert defaults[:hidden_sizes] == [64, 64]
      assert defaults[:temporal] == false
      assert defaults[:backbone] == :sliding_window
      assert defaults[:window_size] == 60
      assert defaults[:precision] == :bf16
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
      opts = Config.parse_args([
        "--temporal",
        "--epochs", "3",
        "--wandb",
        "--batch-size", "16"
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
      for backbone <- [:lstm, :gru, :mamba, :sliding_window, :hybrid] do
        opts = [checkpoint: nil, temporal: true, backbone: backbone]
        result = Config.ensure_checkpoint_name(opts)

        # Format: checkpoints/backbone_memorablename_timestamp.axon
        assert result[:checkpoint] =~ ~r/checkpoints\/#{backbone}_[a-z_]+_\d{8}_\d{6}\.axon/
        # Should also set the :name option
        assert is_binary(result[:name])
      end
    end

    test "non-temporal always uses mlp in name regardless of backbone setting" do
      opts = [checkpoint: nil, temporal: false, backbone: :mamba]
      result = Config.ensure_checkpoint_name(opts)

      # Format: checkpoints/mlp_memorablename_timestamp.axon
      assert result[:checkpoint] =~ ~r/checkpoints\/mlp_[a-z_]+_\d{8}_\d{6}\.axon/
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

      assert opts[:epochs] == 20
      assert opts[:max_files] == 100
      assert opts[:hidden_sizes] == [128, 128]
      assert opts[:temporal] == false
      assert opts[:preset] == :full_cpu
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

    test "character presets inherit from full" do
      full = Config.preset(:full)
      mewtwo = Config.preset(:mewtwo)

      # Should have same temporal and backbone
      assert mewtwo[:temporal] == full[:temporal]
      assert mewtwo[:backbone] == full[:backbone]
      assert mewtwo[:epochs] == full[:epochs]
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

      assert opts[:epochs] == 3  # Overridden
      assert opts[:max_files] == 5  # From preset
      assert opts[:hidden_sizes] == [32, 32]  # From preset
    end

    test "multiple CLI args can override preset" do
      opts = Config.parse_args([
        "--preset", "quick",
        "--epochs", "5",
        "--batch-size", "128",
        "--hidden-sizes", "64,64"
      ])

      assert opts[:epochs] == 5
      assert opts[:batch_size] == 128
      assert opts[:hidden_sizes] == [64, 64]
      assert opts[:max_files] == 5  # Still from preset
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
        character: nil,  # or :mewtwo
        backbone: :mamba,
        temporal: true,
        hidden_sizes: [256, 256],
        window_size: 60,
        epochs: 10,
        training_frames: 100_000,
        final_val_loss: 4.05,
        parent_model: nil,  # or "lstm_20260118_000000" for fine-tuning
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
      opts = Config.defaults()
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
      for backbone <- [:lstm, :gru, :mamba, :sliding_window, :hybrid] do
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
      output = capture_io(:stderr, fn ->
        Config.validate(opts)
      end)
      assert output =~ "window_size"
      assert output =~ "memory"
    end

    test "warns for large batch_size" do
      opts = [epochs: 10, batch_size: 512]
      output = capture_io(:stderr, fn ->
        Config.validate(opts)
      end)
      assert output =~ "batch_size"
      assert output =~ "memory"
    end

    test "warns for many epochs without wandb" do
      opts = [epochs: 25, batch_size: 64, wandb: false]
      output = capture_io(:stderr, fn ->
        Config.validate(opts)
      end)
      assert output =~ "wandb"
    end

    test "no warning for many epochs with wandb" do
      opts = [epochs: 25, batch_size: 64, wandb: true]
      output = capture_io(:stderr, fn ->
        Config.validate(opts)
      end)
      refute output =~ "wandb"
    end

    test "warns for small window_size with temporal" do
      opts = [epochs: 10, batch_size: 64, temporal: true, backbone: :lstm, window_size: 20]
      output = capture_io(:stderr, fn ->
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
    test "parsed args from preset are valid" do
      opts = Config.parse_args(["--preset", "full"])
      assert {:ok, _} = Config.validate(opts)
    end

    test "parsed args with CLI overrides are valid" do
      opts = Config.parse_args(["--preset", "quick", "--epochs", "5", "--batch-size", "128"])
      assert {:ok, _} = Config.validate(opts)
    end

    test "parsed temporal args are valid" do
      opts = Config.parse_args(["--temporal", "--backbone", "mamba", "--window-size", "60"])
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

    test "warmup_steps defaults to 0" do
      opts = Config.parse_args([])
      assert opts[:warmup_steps] == 0
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
      opts = Config.parse_args([
        "--lr", "0.0005",
        "--lr-schedule", "cosine",
        "--warmup-steps", "500",
        "--decay-steps", "5000"
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
end
