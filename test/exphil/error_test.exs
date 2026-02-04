defmodule ExPhil.ErrorTest do
  use ExUnit.Case, async: true

  alias ExPhil.Error
  alias ExPhil.Error.{CheckpointError, ReplayError, ConfigError, GPUError, BridgeError, EmbeddingError, DataError}

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
