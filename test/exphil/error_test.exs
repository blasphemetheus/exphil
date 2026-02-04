defmodule ExPhil.ErrorTest do
  use ExUnit.Case, async: true

  alias ExPhil.Error
  alias ExPhil.Error.{CheckpointError, ReplayError, ConfigError, GPUError, BridgeError, EmbeddingError, DataError, AgentError, ValidationError, LeagueError, SelfPlayError, RegistryError}

  # ============================================================================
  # CheckpointError Tests
  # ============================================================================

  describe "CheckpointError" do
    test "creates error with reason only" do
      error = CheckpointError.new(:corrupted)
      assert error.reason == :corrupted
      assert error.message =~ "corrupted"
    end

    test "creates error with path" do
      error = CheckpointError.new(:not_found, path: "/path/to/model.bin")
      assert error.path == "/path/to/model.bin"
      assert error.message =~ "not found"
      assert error.message =~ "/path/to/model.bin"
    end

    test "creates error with context for incompatible" do
      error = CheckpointError.new(:incompatible,
        path: "model.bin",
        context: %{expected: 512, actual: 256}
      )
      assert error.message =~ "incompatible"
      assert error.message =~ "expected 512"
      assert error.message =~ "got 256"
    end

    test "creates error with details context" do
      error = CheckpointError.new(:write_failed,
        context: %{details: "disk full"}
      )
      assert error.message =~ "disk full"
    end

    test "is raiseable as exception" do
      error = CheckpointError.new(:corrupted, path: "bad.bin")
      assert_raise CheckpointError, ~r/corrupted/, fn ->
        raise error
      end
    end
  end

  # ============================================================================
  # ReplayError Tests
  # ============================================================================

  describe "ReplayError" do
    test "creates NIF panic error" do
      error = ReplayError.new(:nif_panic, path: "corrupted.slp")
      assert error.reason == :nif_panic
      assert error.message =~ "crashed"
      assert error.message =~ "corrupted.slp"
    end

    test "creates too_short error with frame count" do
      error = ReplayError.new(:too_short,
        path: "short.slp",
        context: %{min_frames: 1000, actual: 50}
      )
      assert error.message =~ "too short"
      assert error.message =~ "need 1000 frames"
      assert error.message =~ "got 50"
    end

    test "creates missing_player error with port" do
      error = ReplayError.new(:missing_player,
        context: %{port: 2}
      )
      assert error.message =~ "player not found"
      assert error.message =~ "port 2"
    end

    test "creates unsupported version error" do
      error = ReplayError.new(:unsupported_version,
        context: %{version: "0.1.0"}
      )
      assert error.message =~ "Unsupported"
      assert error.message =~ "0.1.0"
    end
  end

  # ============================================================================
  # ConfigError Tests
  # ============================================================================

  describe "ConfigError" do
    test "creates invalid preset error" do
      error = ConfigError.new(:invalid_preset,
        field: "fast",
        context: %{valid: [:quick, :standard, :full]}
      )
      assert error.field == "fast"
      assert error.message =~ "Invalid preset"
      assert error.message =~ "'fast'"
      assert error.message =~ "quick"
    end

    test "creates invalid flag error" do
      error = ConfigError.new(:invalid_flag, field: "--unknown-flag")
      assert error.message =~ "Unknown flag"
      assert error.message =~ "--unknown-flag"
    end

    test "creates incompatible options error" do
      error = ConfigError.new(:incompatible,
        field: :temporal,
        context: %{conflict: :single_frame}
      )
      assert error.message =~ "Incompatible"
      assert error.message =~ "temporal"
      assert error.message =~ "single_frame"
    end

    test "creates missing required error" do
      error = ConfigError.new(:missing_required, field: :replays)
      assert error.message =~ "Missing required"
      assert error.message =~ "replays"
    end
  end

  # ============================================================================
  # GPUError Tests
  # ============================================================================

  describe "GPUError" do
    test "creates not found error" do
      error = GPUError.new(:not_found)
      assert error.message =~ "No GPU found"
    end

    test "creates OOM error with batch size hint" do
      error = GPUError.new(:oom, context: %{batch_size: 512})
      assert error.message =~ "out of memory"
      assert error.message =~ "reducing batch size from 512"
    end

    test "creates OOM error with memory usage" do
      error = GPUError.new(:oom,
        context: %{memory_used: 7.8, memory_total: 8.0}
      )
      assert error.message =~ "out of memory"
      assert error.message =~ "7.8/8.0 GB"
    end

    test "creates driver mismatch error" do
      error = GPUError.new(:driver_mismatch,
        context: %{cuda_version: "12.0", driver_version: "11.5"}
      )
      assert error.message =~ "mismatch"
      assert error.message =~ "CUDA 12.0"
      assert error.message =~ "driver 11.5"
    end
  end

  # ============================================================================
  # BridgeError Tests
  # ============================================================================

  describe "BridgeError" do
    test "creates not running error" do
      error = BridgeError.new(:not_running, bridge: :pytorch_port)
      assert error.bridge == :pytorch_port
      assert error.message =~ "not running"
      assert error.message =~ "pytorch_port"
    end

    test "creates timeout error" do
      error = BridgeError.new(:timeout,
        bridge: :flash_attention,
        context: %{timeout_ms: 30000}
      )
      assert error.message =~ "timed out"
      assert error.message =~ "30000ms"
    end

    test "creates crashed error with operation" do
      error = BridgeError.new(:crashed,
        bridge: :selective_scan,
        context: %{operation: :forward}
      )
      assert error.message =~ "crashed"
      assert error.message =~ "forward"
    end
  end

  # ============================================================================
  # EmbeddingError Tests
  # ============================================================================

  describe "EmbeddingError" do
    test "creates shape mismatch error" do
      error = EmbeddingError.new(:shape_mismatch,
        context: %{expected: {32, 512}, actual: {32, 256}}
      )
      assert error.message =~ "shape mismatch"
      assert error.message =~ "{32, 512}"
      assert error.message =~ "{32, 256}"
    end

    test "creates size mismatch error" do
      error = EmbeddingError.new(:size_mismatch,
        context: %{expected: 1208, actual: 288}
      )
      assert error.message =~ "size mismatch"
    end

    test "creates invalid input error with field" do
      error = EmbeddingError.new(:invalid_input,
        context: %{field: "player.action"}
      )
      assert error.message =~ "Invalid"
      assert error.message =~ "player.action"
    end
  end

  # ============================================================================
  # DataError Tests
  # ============================================================================

  describe "DataError" do
    test "creates insufficient data error" do
      error = DataError.new(:insufficient_data,
        context: %{required: 100, actual: 50}
      )
      assert error.reason == :insufficient_data
      assert error.message =~ "Insufficient data"
      assert error.message =~ "need 100"
      assert error.message =~ "got 50"
    end

    test "creates python not found error" do
      error = DataError.new(:python_not_found)
      assert error.message =~ "Python not found"
    end

    test "creates script failed error with exit code" do
      error = DataError.new(:script_failed,
        context: %{exit_code: 1}
      )
      assert error.message =~ "script failed"
      assert error.message =~ "exit code 1"
    end

    test "creates empty dataset error" do
      error = DataError.new(:empty_dataset,
        context: %{path: "/data/train.bin"}
      )
      assert error.message =~ "empty"
      assert error.message =~ "/data/train.bin"
    end

    test "creates parse failed error with details" do
      error = DataError.new(:parse_failed,
        context: %{details: "invalid JSON"}
      )
      assert error.message =~ "parse"
      assert error.message =~ "invalid JSON"
    end
  end

  # ============================================================================
  # AgentError Tests
  # ============================================================================

  describe "AgentError" do
    test "creates no_policy_loaded error" do
      error = AgentError.new(:no_policy_loaded)
      assert error.reason == :no_policy_loaded
      assert error.message =~ "No policy loaded"
    end

    test "creates error with agent name" do
      error = AgentError.new(:not_found, agent: :player1)
      assert error.agent == :player1
      assert error.message =~ "not found"
      assert error.message =~ "player1"
    end

    test "creates initialization_failed error with details" do
      error = AgentError.new(:initialization_failed,
        agent: :main_agent,
        context: %{details: "missing model weights"}
      )
      assert error.message =~ "initialization failed"
      assert error.message =~ "missing model weights"
    end
  end

  # ============================================================================
  # ValidationError Tests
  # ============================================================================

  describe "ValidationError" do
    test "creates not_found error with path" do
      error = ValidationError.new(:not_found, path: "/path/to/file.slp")
      assert error.reason == :not_found
      assert error.message =~ "not found"
      assert error.message =~ "/path/to/file.slp"
    end

    test "creates permission_denied error" do
      error = ValidationError.new(:permission_denied, path: "/protected/file.slp")
      assert error.message =~ "Permission denied"
      assert error.message =~ "/protected/file.slp"
    end

    test "creates file_too_small error with size info" do
      error = ValidationError.new(:file_too_small,
        path: "small.slp",
        context: %{min_size: 1024, actual_size: 100}
      )
      assert error.message =~ "too small"
      assert error.message =~ "minimum 1024 bytes"
      assert error.message =~ "got 100"
    end

    test "creates timeout error" do
      error = ValidationError.new(:timeout)
      assert error.message =~ "timed out"
    end
  end

  # ============================================================================
  # LeagueError Tests
  # ============================================================================

  describe "LeagueError" do
    test "creates not_found error with agent_id" do
      error = LeagueError.new(:not_found, agent_id: "agent_123")
      assert error.reason == :not_found
      assert error.agent_id == "agent_123"
      assert error.message =~ "not found"
      assert error.message =~ "agent_123"
    end

    test "creates already_registered error" do
      error = LeagueError.new(:already_registered, agent_id: "mewtwo_v2")
      assert error.message =~ "already registered"
      assert error.message =~ "mewtwo_v2"
    end

    test "creates no_model error" do
      error = LeagueError.new(:no_model, agent_id: "new_agent")
      assert error.message =~ "No model available"
    end
  end

  # ============================================================================
  # SelfPlayError Tests
  # ============================================================================

  describe "SelfPlayError" do
    test "creates game_not_started error" do
      error = SelfPlayError.new(:game_not_started)
      assert error.reason == :game_not_started
      assert error.message =~ "Game not started"
    end

    test "creates game_finished error with game_id" do
      error = SelfPlayError.new(:game_finished, game_id: "game_456")
      assert error.game_id == "game_456"
      assert error.message =~ "already finished"
      assert error.message =~ "game_456"
    end

    test "creates no_current_policy error" do
      error = SelfPlayError.new(:no_current_policy)
      assert error.message =~ "No current policy"
    end

    test "creates dolphin_failed error with exit code" do
      error = SelfPlayError.new(:dolphin_failed,
        context: %{exit_code: 1}
      )
      assert error.message =~ "Dolphin"
      assert error.message =~ "exit code 1"
    end
  end

  # ============================================================================
  # RegistryError Tests
  # ============================================================================

  describe "RegistryError" do
    test "creates not_found error with model_id" do
      error = RegistryError.new(:not_found, model_id: "model_v1")
      assert error.reason == :not_found
      assert error.model_id == "model_v1"
      assert error.message =~ "not found"
      assert error.message =~ "model_v1"
    end

    test "creates missing_required_field error" do
      error = RegistryError.new(:missing_required_field,
        context: %{field: :checkpoint_path}
      )
      assert error.message =~ "Missing required field"
      assert error.message =~ "checkpoint_path"
    end

    test "creates no_models_with_metric error" do
      error = RegistryError.new(:no_models_with_metric,
        context: %{metric: :val_loss}
      )
      assert error.message =~ "No models with"
      assert error.message =~ "val_loss"
    end
  end

  # ============================================================================
  # Error Module Functions
  # ============================================================================

  describe "Error.error?/1" do
    test "returns true for all error types" do
      assert Error.error?(%CheckpointError{reason: :corrupted, message: ""})
      assert Error.error?(%ReplayError{reason: :nif_panic, message: ""})
      assert Error.error?(%ConfigError{reason: :invalid_preset, message: ""})
      assert Error.error?(%GPUError{reason: :oom, message: ""})
      assert Error.error?(%BridgeError{reason: :timeout, message: ""})
      assert Error.error?(%EmbeddingError{reason: :shape_mismatch, message: ""})
      assert Error.error?(%DataError{reason: :insufficient_data, message: ""})
      assert Error.error?(%AgentError{reason: :no_policy_loaded, message: ""})
      assert Error.error?(%ValidationError{reason: :not_found, message: ""})
      assert Error.error?(%LeagueError{reason: :not_found, message: ""})
      assert Error.error?(%SelfPlayError{reason: :game_not_started, message: ""})
      assert Error.error?(%RegistryError{reason: :not_found, message: ""})
    end

    test "returns false for non-errors" do
      refute Error.error?("string error")
      refute Error.error?(:atom_error)
      refute Error.error?({:error, :reason})
      refute Error.error?(nil)
    end
  end

  describe "Error.wrap/3" do
    test "wraps checkpoint errors" do
      assert %CheckpointError{reason: :not_found} = Error.wrap(:not_found, :checkpoint)
      assert %CheckpointError{reason: :corrupted} = Error.wrap(:corrupted, :checkpoint)
      assert %CheckpointError{reason: :incompatible} = Error.wrap(:incompatible, :checkpoint)
    end

    test "wraps replay errors" do
      assert %ReplayError{reason: :not_found} = Error.wrap(:not_found, :replay)
      assert %ReplayError{reason: :nif_panic} = Error.wrap(:nif_panic, :replay)
      assert %ReplayError{reason: :nif_panic} = Error.wrap(:nif_panicked, :replay)
    end

    test "wraps GPU errors" do
      assert %GPUError{reason: :not_found} = Error.wrap(:not_found, :gpu)
      assert %GPUError{reason: :not_found} = Error.wrap(:nvidia_smi_not_found, :gpu)
      assert %GPUError{reason: :oom} = Error.wrap(:oom, :gpu)
    end

    test "wraps bridge errors" do
      assert %BridgeError{reason: :not_running} = Error.wrap(:not_running, :bridge)
      assert %BridgeError{reason: :timeout} = Error.wrap(:timeout, :bridge)
      assert %BridgeError{reason: :timeout} = Error.wrap(:queue_full, :bridge)
    end

    test "wraps data errors" do
      assert %DataError{reason: :insufficient_data} = Error.wrap(:insufficient_data, :data)
      assert %DataError{reason: :python_not_found} = Error.wrap(:python_not_found, :data)
      assert %DataError{reason: :script_failed} = Error.wrap(:script_failed, :data)
      assert %DataError{reason: :empty_dataset} = Error.wrap(:empty_dataset, :data)
    end

    test "wraps agent errors" do
      assert %AgentError{reason: :no_policy_loaded} = Error.wrap(:no_policy_loaded, :agent)
      assert %AgentError{reason: :not_found} = Error.wrap(:not_found, :agent)
      assert %AgentError{reason: :initialization_failed} = Error.wrap(:initialization_failed, :agent)
    end

    test "wraps validation errors" do
      assert %ValidationError{reason: :not_found} = Error.wrap(:not_found, :validation)
      assert %ValidationError{reason: :not_regular_file} = Error.wrap(:not_regular_file, :validation)
      assert %ValidationError{reason: :permission_denied} = Error.wrap(:permission_denied, :validation)
      assert %ValidationError{reason: :file_too_small} = Error.wrap(:file_too_small, :validation)
      # Also wraps common POSIX errors
      assert %ValidationError{reason: :not_found} = Error.wrap(:enoent, :validation)
      assert %ValidationError{reason: :permission_denied} = Error.wrap(:eacces, :validation)
    end

    test "wraps league errors" do
      assert %LeagueError{reason: :not_found} = Error.wrap(:not_found, :league)
      assert %LeagueError{reason: :already_registered} = Error.wrap(:already_registered, :league)
      assert %LeagueError{reason: :no_model} = Error.wrap(:no_model, :league)
    end

    test "wraps self_play errors" do
      assert %SelfPlayError{reason: :game_not_started} = Error.wrap(:game_not_started, :self_play)
      assert %SelfPlayError{reason: :game_finished} = Error.wrap(:game_finished, :self_play)
      assert %SelfPlayError{reason: :no_current_policy} = Error.wrap(:no_current_policy, :self_play)
      assert %SelfPlayError{reason: :not_found} = Error.wrap(:not_found, :self_play)
    end

    test "wraps registry errors" do
      assert %RegistryError{reason: :not_found} = Error.wrap(:not_found, :registry)
      assert %RegistryError{reason: :missing_required_field} = Error.wrap(:missing_required_field, :registry)
      assert %RegistryError{reason: :no_models_with_metric} = Error.wrap(:no_models_with_metric, :registry)
    end

    test "preserves options in wrapped error" do
      error = Error.wrap(:not_found, :checkpoint, path: "/foo/bar.bin")
      assert error.path == "/foo/bar.bin"
    end

    test "returns original value for unknown errors" do
      assert "unknown" == Error.wrap("unknown", :checkpoint)
      assert :weird_error == Error.wrap(:weird_error, :unknown)
    end
  end
end
