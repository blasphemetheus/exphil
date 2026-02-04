defmodule ExPhil.Error do
  @moduledoc """
  Structured error types for ExPhil.

  Provides typed errors with context instead of generic strings or atoms.
  Each error type includes relevant context for debugging and recovery.

  ## Error Categories

  | Module | Description |
  |--------|-------------|
  | `CheckpointError` | Checkpoint loading, saving, validation |
  | `ReplayError` | Replay file parsing, validation |
  | `ConfigError` | Training configuration issues |
  | `GPUError` | GPU/hardware detection and OOM |
  | `BridgeError` | Python/NIF port communication |
  | `EmbeddingError` | Tensor shape mismatches |

  ## Usage

      case load_checkpoint(path) do
        {:ok, checkpoint} -> use(checkpoint)
        {:error, %CheckpointError{reason: :corrupted}} ->
          Logger.error("Checkpoint corrupted, starting fresh")
        {:error, %CheckpointError{reason: :incompatible, context: ctx}} ->
          Logger.error("Model incompatible: expected \#{ctx.expected}, got \#{ctx.actual}")
      end

  ## See Also

  - `ExPhil.Training` - Uses checkpoint errors
  - `ExPhil.Data.Peppi` - Uses replay errors
  - `ExPhil.Training.GPUUtils` - Uses GPU errors
  """

  # ============================================================================
  # Checkpoint Errors
  # ============================================================================

  defmodule CheckpointError do
    @moduledoc """
    Errors during checkpoint operations.

    ## Reasons

    - `:not_found` - Checkpoint file doesn't exist
    - `:corrupted` - File exists but can't be deserialized
    - `:incompatible` - Model architecture mismatch
    - `:version_mismatch` - Checkpoint from different ExPhil version
    - `:write_failed` - Could not save checkpoint
    """

    
    defexception [:reason, :path, :context, :message]

    @type reason ::
            :not_found
            | :corrupted
            | :incompatible
            | :version_mismatch
            | :write_failed

    @type t :: %__MODULE__{
            reason: reason(),
            path: String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_found -> "Checkpoint not found"
          :corrupted -> "Checkpoint corrupted or malformed"
          :incompatible -> "Checkpoint incompatible with current model"
          :version_mismatch -> "Checkpoint from incompatible ExPhil version"
          :write_failed -> "Failed to write checkpoint"
        end

      parts = [base]
      parts = if error.path, do: parts ++ ["at #{error.path}"], else: parts

      parts =
        if error.context do
          case error.context do
            %{expected: exp, actual: act} ->
              parts ++ ["(expected #{inspect(exp)}, got #{inspect(act)})"]

            %{details: details} ->
              parts ++ ["(#{details})"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new checkpoint error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        path: Keyword.get(opts, :path),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Replay Errors
  # ============================================================================

  defmodule ReplayError do
    @moduledoc """
    Errors during replay file operations.

    ## Reasons

    - `:not_found` - Replay file doesn't exist
    - `:nif_panic` - Native parser crashed (corrupted file)
    - `:invalid_format` - Not a valid SLP file
    - `:too_short` - Game too short for training
    - `:missing_player` - Required player port not found
    - `:unsupported_version` - SLP version not supported
    """

    
    defexception [:reason, :path, :context, :message]

    @type reason ::
            :not_found
            | :nif_panic
            | :invalid_format
            | :too_short
            | :missing_player
            | :unsupported_version

    @type t :: %__MODULE__{
            reason: reason(),
            path: String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_found -> "Replay file not found"
          :nif_panic -> "Replay parser crashed (file may be corrupted)"
          :invalid_format -> "Invalid SLP file format"
          :too_short -> "Game too short for training"
          :missing_player -> "Required player not found in replay"
          :unsupported_version -> "Unsupported SLP version"
        end

      parts = [base]
      parts = if error.path, do: parts ++ ["at #{error.path}"], else: parts

      parts =
        if error.context do
          case error.context do
            %{min_frames: min, actual: actual} ->
              parts ++ ["(need #{min} frames, got #{actual})"]

            %{port: port} ->
              parts ++ ["(port #{port})"]

            %{version: version} ->
              parts ++ ["(version #{version})"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new replay error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        path: Keyword.get(opts, :path),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Configuration Errors
  # ============================================================================

  defmodule ConfigError do
    @moduledoc """
    Errors in training configuration.

    ## Reasons

    - `:invalid_preset` - Unknown preset name
    - `:invalid_flag` - Unknown CLI flag
    - `:invalid_value` - Flag value out of range or wrong type
    - `:incompatible` - Conflicting options
    - `:missing_required` - Required option not provided
    """

    
    defexception [:reason, :field, :context, :message]

    @type reason ::
            :invalid_preset
            | :invalid_flag
            | :invalid_value
            | :incompatible
            | :missing_required

    @type t :: %__MODULE__{
            reason: reason(),
            field: atom() | String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :invalid_preset -> "Invalid preset"
          :invalid_flag -> "Unknown flag"
          :invalid_value -> "Invalid value"
          :incompatible -> "Incompatible options"
          :missing_required -> "Missing required option"
        end

      parts = [base]
      parts = if error.field, do: parts ++ ["'#{error.field}'"], else: parts

      parts =
        if error.context do
          case error.context do
            %{valid: valid} ->
              parts ++ ["(valid: #{Enum.join(valid, ", ")})"]

            %{expected: exp, actual: act} ->
              parts ++ ["(expected #{inspect(exp)}, got #{inspect(act)})"]

            %{conflict: conflict} ->
              parts ++ ["conflicts with '#{conflict}'"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new config error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        field: Keyword.get(opts, :field),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # GPU/Hardware Errors
  # ============================================================================

  defmodule GPUError do
    @moduledoc """
    GPU and hardware-related errors.

    ## Reasons

    - `:not_found` - No GPU detected
    - `:nvidia_smi_failed` - nvidia-smi command failed
    - `:oom` - Out of GPU memory
    - `:driver_mismatch` - CUDA/driver version mismatch
    - `:unsupported` - GPU not supported for operation
    """

    
    defexception [:reason, :context, :message]

    @type reason ::
            :not_found
            | :nvidia_smi_failed
            | :oom
            | :driver_mismatch
            | :unsupported

    @type t :: %__MODULE__{
            reason: reason(),
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_found -> "No GPU found"
          :nvidia_smi_failed -> "nvidia-smi command failed"
          :oom -> "GPU out of memory"
          :driver_mismatch -> "CUDA/driver version mismatch"
          :unsupported -> "GPU operation not supported"
        end

      parts = [base]

      parts =
        if error.context do
          case error.context do
            %{batch_size: bs} ->
              parts ++ ["(try reducing batch size from #{bs})"]

            %{memory_used: used, memory_total: total} ->
              parts ++ ["(#{used}/#{total} GB used)"]

            %{cuda_version: cv, driver_version: dv} ->
              parts ++ ["(CUDA #{cv}, driver #{dv})"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new GPU error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Bridge/Port Errors
  # ============================================================================

  defmodule BridgeError do
    @moduledoc """
    Errors communicating with external processes (Python, NIFs).

    ## Reasons

    - `:not_running` - Process not started
    - `:timeout` - Call timed out
    - `:crashed` - Process crashed during call
    - `:protocol_error` - Invalid response from process
    """

    
    defexception [:reason, :bridge, :context, :message]

    @type reason :: :not_running | :timeout | :crashed | :protocol_error

    @type t :: %__MODULE__{
            reason: reason(),
            bridge: atom() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_running -> "Bridge not running"
          :timeout -> "Bridge call timed out"
          :crashed -> "Bridge process crashed"
          :protocol_error -> "Invalid response from bridge"
        end

      parts = [base]
      parts = if error.bridge, do: parts ++ ["(#{error.bridge})"], else: parts

      parts =
        if error.context do
          case error.context do
            %{timeout_ms: ms} ->
              parts ++ ["after #{ms}ms"]

            %{operation: op} ->
              parts ++ ["during #{op}"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new bridge error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        bridge: Keyword.get(opts, :bridge),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Embedding Errors
  # ============================================================================

  defmodule EmbeddingError do
    @moduledoc """
    Errors in tensor embedding operations.

    ## Reasons

    - `:shape_mismatch` - Tensor shapes don't match
    - `:size_mismatch` - Embedding size differs from expected
    - `:type_mismatch` - Tensor dtype differs
    - `:invalid_input` - Input data malformed
    """

    
    defexception [:reason, :context, :message]

    @type reason :: :shape_mismatch | :size_mismatch | :type_mismatch | :invalid_input

    @type t :: %__MODULE__{
            reason: reason(),
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :shape_mismatch -> "Tensor shape mismatch"
          :size_mismatch -> "Embedding size mismatch"
          :type_mismatch -> "Tensor type mismatch"
          :invalid_input -> "Invalid embedding input"
        end

      parts = [base]

      parts =
        if error.context do
          case error.context do
            %{expected: exp, actual: act} ->
              parts ++ ["(expected #{inspect(exp)}, got #{inspect(act)})"]

            %{field: field} ->
              parts ++ ["in #{field}"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new embedding error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Data/Dataset Errors
  # ============================================================================

  defmodule DataError do
    @moduledoc """
    Errors during data loading and processing.

    ## Reasons

    - `:insufficient_data` - Not enough data for operation
    - `:parse_failed` - Failed to parse data file
    - `:python_not_found` - Python interpreter not found
    - `:script_failed` - External script failed
    - `:empty_dataset` - Dataset contains no valid samples
    """

    defexception [:reason, :context, :message]

    @type reason ::
            :insufficient_data
            | :parse_failed
            | :python_not_found
            | :script_failed
            | :empty_dataset

    @type t :: %__MODULE__{
            reason: reason(),
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :insufficient_data -> "Insufficient data"
          :parse_failed -> "Failed to parse data"
          :python_not_found -> "Python not found in PATH"
          :script_failed -> "External script failed"
          :empty_dataset -> "Dataset is empty"
        end

      parts = [base]

      parts =
        if error.context do
          case error.context do
            %{required: req, actual: act} ->
              parts ++ ["(need #{req}, got #{act})"]

            %{exit_code: code} ->
              parts ++ ["(exit code #{code})"]

            %{path: path} ->
              parts ++ ["at #{path}"]

            %{details: details} ->
              parts ++ ["(#{details})"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new data error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Convenience Functions
  # ============================================================================

  @doc """
  Check if a value is an ExPhil error struct.

  ## Examples

      iex> ExPhil.Error.error?(%ExPhil.Error.ConfigError{reason: :invalid_preset})
      true

      iex> ExPhil.Error.error?("some string")
      false
  """
  @spec error?(term()) :: boolean()
  def error?(%CheckpointError{}), do: true
  def error?(%ReplayError{}), do: true
  def error?(%ConfigError{}), do: true
  def error?(%GPUError{}), do: true
  def error?(%BridgeError{}), do: true
  def error?(%EmbeddingError{}), do: true
  def error?(%DataError{}), do: true
  def error?(_), do: false

  @doc """
  Wrap a legacy error (atom or string) into a structured error if possible.

  Returns the original value if it can't be wrapped.

  ## Examples

      iex> ExPhil.Error.wrap(:not_found, :checkpoint, path: "foo.bin")
      %ExPhil.Error.CheckpointError{reason: :not_found, path: "foo.bin"}

      iex> ExPhil.Error.wrap("unknown error", :unknown)
      "unknown error"
  """
  @spec wrap(term(), atom(), keyword()) :: struct() | term()
  def wrap(error, category, opts \\ [])

  def wrap(:not_found, :checkpoint, opts), do: CheckpointError.new(:not_found, opts)
  def wrap(:corrupted, :checkpoint, opts), do: CheckpointError.new(:corrupted, opts)
  def wrap(:incompatible, :checkpoint, opts), do: CheckpointError.new(:incompatible, opts)
  def wrap(:write_failed, :checkpoint, opts), do: CheckpointError.new(:write_failed, opts)

  def wrap(:not_found, :replay, opts), do: ReplayError.new(:not_found, opts)
  def wrap(:nif_panic, :replay, opts), do: ReplayError.new(:nif_panic, opts)
  def wrap(:nif_panicked, :replay, opts), do: ReplayError.new(:nif_panic, opts)
  def wrap(:invalid_format, :replay, opts), do: ReplayError.new(:invalid_format, opts)
  def wrap(:too_short, :replay, opts), do: ReplayError.new(:too_short, opts)

  def wrap(:not_found, :gpu, opts), do: GPUError.new(:not_found, opts)
  def wrap(:nvidia_smi_failed, :gpu, opts), do: GPUError.new(:nvidia_smi_failed, opts)
  def wrap(:nvidia_smi_not_found, :gpu, opts), do: GPUError.new(:not_found, opts)
  def wrap(:oom, :gpu, opts), do: GPUError.new(:oom, opts)

  def wrap(:not_running, :bridge, opts), do: BridgeError.new(:not_running, opts)
  def wrap(:timeout, :bridge, opts), do: BridgeError.new(:timeout, opts)
  def wrap(:queue_full, :bridge, opts), do: BridgeError.new(:timeout, opts)

  def wrap(:insufficient_data, :data, opts), do: DataError.new(:insufficient_data, opts)
  def wrap(:parse_failed, :data, opts), do: DataError.new(:parse_failed, opts)
  def wrap(:python_not_found, :data, opts), do: DataError.new(:python_not_found, opts)
  def wrap(:script_failed, :data, opts), do: DataError.new(:script_failed, opts)
  def wrap(:empty_dataset, :data, opts), do: DataError.new(:empty_dataset, opts)

  def wrap(error, _category, _opts), do: error
end
