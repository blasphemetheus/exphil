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
  | `DataError` | Data loading and processing |
  | `AgentError` | Agent/policy runtime issues |
  | `ValidationError` | File validation errors |
  | `LeagueError` | League/matchmaking operations |
  | `SelfPlayError` | Self-play game management |
  | `RegistryError` | Model registry operations |
  | `WandBError` | Weights & Biases integration |
  | `TelemetryError` | Metrics collection |
  | `CacheError` | Embedding/mmap cache operations |
  | `RecoveryError` | Training recovery/checkpointing |
  | `YamlError` | YAML configuration parsing |

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
            | :queue_full

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
          :queue_full -> "Checkpoint save queue full"
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

    @type reason :: :not_running | :timeout | :crashed | :protocol_error | :port_closed

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
          :port_closed -> "Bridge port closed unexpectedly"
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
  # Agent Errors
  # ============================================================================

  defmodule AgentError do
    @moduledoc """
    Errors in agent operations.

    ## Reasons

    - `:no_policy_loaded` - Agent has no policy loaded
    - `:not_found` - Agent not found
    - `:initialization_failed` - Agent failed to initialize
    - `:inference_failed` - Policy inference failed
    """

    defexception [:reason, :agent, :context, :message]

    @type reason :: :no_policy_loaded | :not_found | :initialization_failed | :inference_failed

    @type t :: %__MODULE__{
            reason: reason(),
            agent: atom() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :no_policy_loaded -> "No policy loaded"
          :not_found -> "Agent not found"
          :initialization_failed -> "Agent initialization failed"
          :inference_failed -> "Policy inference failed"
        end

      parts = [base]
      parts = if error.agent, do: parts ++ ["(#{error.agent})"], else: parts

      parts =
        if error.context do
          case error.context do
            %{details: details} ->
              parts ++ ["- #{details}"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new agent error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        agent: Keyword.get(opts, :agent),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Validation Errors
  # ============================================================================

  defmodule ValidationError do
    @moduledoc """
    Errors during file/data validation.

    ## Reasons

    - `:not_found` - File does not exist
    - `:not_regular_file` - Path is not a regular file
    - `:permission_denied` - No permission to access file
    - `:file_too_small` - File below minimum size
    - `:empty_file` - File is empty
    - `:invalid_format` - File format is invalid
    - `:timeout` - Validation timed out
    """

    defexception [:reason, :path, :context, :message]

    @type reason ::
            :not_found
            | :not_regular_file
            | :permission_denied
            | :file_too_small
            | :empty_file
            | :invalid_format
            | :timeout

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
          :not_found -> "File not found"
          :not_regular_file -> "Not a regular file"
          :permission_denied -> "Permission denied"
          :file_too_small -> "File too small"
          :empty_file -> "File is empty"
          :invalid_format -> "Invalid file format"
          :timeout -> "Validation timed out"
        end

      parts = [base]
      parts = if error.path, do: parts ++ ["at #{error.path}"], else: parts

      parts =
        if error.context do
          case error.context do
            %{min_size: min, actual_size: actual} ->
              parts ++ ["(minimum #{min} bytes, got #{actual})"]

            %{expected_magic: exp} ->
              parts ++ ["(expected #{exp} magic bytes)"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new validation error.
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
  # League Errors
  # ============================================================================

  defmodule LeagueError do
    @moduledoc """
    Errors in league/matchmaking operations.

    ## Reasons

    - `:not_found` - Agent or entry not found
    - `:already_registered` - Agent already registered
    - `:no_model` - No model available for agent
    - `:match_failed` - Match creation failed
    """

    defexception [:reason, :agent_id, :context, :message]

    @type reason :: :not_found | :already_registered | :no_model | :match_failed

    @type t :: %__MODULE__{
            reason: reason(),
            agent_id: String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_found -> "Agent not found in league"
          :already_registered -> "Agent already registered"
          :no_model -> "No model available"
          :match_failed -> "Match creation failed"
        end

      parts = [base]
      parts = if error.agent_id, do: parts ++ ["(#{error.agent_id})"], else: parts

      parts =
        if error.context do
          case error.context do
            %{details: details} ->
              parts ++ ["- #{details}"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new league error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        agent_id: Keyword.get(opts, :agent_id),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Self-Play Errors
  # ============================================================================

  defmodule SelfPlayError do
    @moduledoc """
    Errors in self-play game management.

    ## Reasons

    - `:game_not_started` - Game has not started yet
    - `:game_finished` - Game is already finished
    - `:no_current_policy` - No current policy available
    - `:not_found` - Game or resource not found
    - `:dolphin_failed` - Dolphin process failed
    """

    defexception [:reason, :game_id, :context, :message]

    @type reason ::
            :game_not_started
            | :game_finished
            | :no_current_policy
            | :not_found
            | :dolphin_failed

    @type t :: %__MODULE__{
            reason: reason(),
            game_id: String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :game_not_started -> "Game not started"
          :game_finished -> "Game already finished"
          :no_current_policy -> "No current policy available"
          :not_found -> "Game not found"
          :dolphin_failed -> "Dolphin process failed"
        end

      parts = [base]
      parts = if error.game_id, do: parts ++ ["(game #{error.game_id})"], else: parts

      parts =
        if error.context do
          case error.context do
            %{details: details} ->
              parts ++ ["- #{details}"]

            %{exit_code: code} ->
              parts ++ ["(exit code #{code})"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new self-play error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        game_id: Keyword.get(opts, :game_id),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # Registry Errors
  # ============================================================================

  defmodule RegistryError do
    @moduledoc """
    Errors in model registry operations.

    ## Reasons

    - `:not_found` - Model not found
    - `:missing_required_field` - Required field missing from entry
    - `:no_models_with_metric` - No models have requested metric
    - `:write_failed` - Failed to write registry file
    """

    defexception [:reason, :model_id, :context, :message]

    @type reason :: :not_found | :missing_required_field | :no_models_with_metric | :write_failed

    @type t :: %__MODULE__{
            reason: reason(),
            model_id: String.t() | nil,
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :not_found -> "Model not found in registry"
          :missing_required_field -> "Missing required field"
          :no_models_with_metric -> "No models with requested metric"
          :write_failed -> "Failed to write registry"
        end

      parts = [base]
      parts = if error.model_id, do: parts ++ ["(#{error.model_id})"], else: parts

      parts =
        if error.context do
          case error.context do
            %{field: field} ->
              parts ++ ["'#{field}'"]

            %{metric: metric} ->
              parts ++ ["'#{metric}'"]

            _ ->
              parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new registry error.
    """
    @spec new(reason(), keyword()) :: t()
    def new(reason, opts \\ []) do
      error = %__MODULE__{
        reason: reason,
        model_id: Keyword.get(opts, :model_id),
        context: Keyword.get(opts, :context),
        message: ""
      }

      %{error | message: message(error)}
    end
  end

  # ============================================================================
  # WandB Errors
  # ============================================================================

  defmodule WandBError do
    @moduledoc """
    Errors in Weights & Biases integration.

    ## Reasons

    - `:no_api_key` - WANDB_API_KEY not set
    - `:run_already_active` - Tried to start run when one is active
    - `:no_active_run` - Tried to log without active run
    - `:initialization_failed` - Failed to initialize WandB
    """

    defexception [:reason, :context, :message]

    @type reason :: :no_api_key | :run_already_active | :no_active_run | :initialization_failed

    @type t :: %__MODULE__{
            reason: reason(),
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :no_api_key -> "WANDB_API_KEY environment variable not set"
          :run_already_active -> "WandB run already active"
          :no_active_run -> "No active WandB run"
          :initialization_failed -> "Failed to initialize WandB"
        end

      parts = [base]

      parts =
        if error.context do
          case error.context do
            %{details: details} -> parts ++ ["(#{details})"]
            _ -> parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new WandB error.
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
  # Telemetry Errors
  # ============================================================================

  defmodule TelemetryError do
    @moduledoc """
    Errors in telemetry/metrics collection.

    ## Reasons

    - `:collector_not_started` - Telemetry collector not running
    - `:invalid_metric` - Invalid metric name or value
    """

    defexception [:reason, :context, :message]

    @type reason :: :collector_not_started | :invalid_metric

    @type t :: %__MODULE__{
            reason: reason(),
            context: map() | nil,
            message: String.t()
          }

    @doc false
    def message(%__MODULE__{} = error) do
      base =
        case error.reason do
          :collector_not_started -> "Telemetry collector not started"
          :invalid_metric -> "Invalid metric"
        end

      parts = [base]

      parts =
        if error.context do
          case error.context do
            %{metric: metric} -> parts ++ ["'#{metric}'"]
            _ -> parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new telemetry error.
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
  # Cache Errors
  # ============================================================================

  defmodule CacheError do
    @moduledoc """
    Errors in embedding/mmap cache operations.

    ## Reasons

    - `:not_found` - Cache entry not found
    - `:invalid_header` - Invalid cache file header
    - `:corrupted` - Cache file corrupted
    - `:write_failed` - Failed to write cache
    """

    defexception [:reason, :path, :context, :message]

    @type reason :: :not_found | :invalid_header | :corrupted | :write_failed

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
          :not_found -> "Cache entry not found"
          :invalid_header -> "Invalid cache header"
          :corrupted -> "Cache file corrupted"
          :write_failed -> "Failed to write cache"
        end

      parts = [base]
      parts = if error.path, do: parts ++ ["at #{error.path}"], else: parts

      parts =
        if error.context do
          case error.context do
            %{details: details} -> parts ++ ["(#{details})"]
            _ -> parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new cache error.
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
  # Recovery Errors
  # ============================================================================

  defmodule RecoveryError do
    @moduledoc """
    Errors in training recovery operations.

    ## Reasons

    - `:invalid_marker` - Invalid recovery marker file
    - `:no_checkpoint` - No checkpoint to recover from
    - `:state_mismatch` - Recovery state doesn't match current training
    """

    defexception [:reason, :path, :context, :message]

    @type reason :: :invalid_marker | :no_checkpoint | :state_mismatch

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
          :invalid_marker -> "Invalid recovery marker"
          :no_checkpoint -> "No checkpoint to recover from"
          :state_mismatch -> "Recovery state mismatch"
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
    Create a new recovery error.
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
  # YAML Errors
  # ============================================================================

  defmodule YamlError do
    @moduledoc """
    Errors in YAML configuration parsing.

    ## Reasons

    - `:file_not_found` - YAML file not found
    - `:invalid_format` - Invalid YAML syntax
    - `:invalid_value` - Valid YAML but invalid config value
    - `:missing_field` - Required field missing
    """

    defexception [:reason, :path, :context, :message]

    @type reason :: :file_not_found | :invalid_format | :invalid_value | :missing_field

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
          :file_not_found -> "YAML file not found"
          :invalid_format -> "Invalid YAML format"
          :invalid_value -> "Invalid configuration value"
          :missing_field -> "Missing required field"
        end

      parts = [base]
      parts = if error.path, do: parts ++ ["at #{error.path}"], else: parts

      parts =
        if error.context do
          case error.context do
            %{field: field} -> parts ++ ["'#{field}'"]
            %{line: line} -> parts ++ ["(line #{line})"]
            %{details: details} -> parts ++ ["(#{details})"]
            _ -> parts
          end
        else
          parts
        end

      Enum.join(parts, " ")
    end

    @doc """
    Create a new YAML error.
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
  def error?(%AgentError{}), do: true
  def error?(%ValidationError{}), do: true
  def error?(%LeagueError{}), do: true
  def error?(%SelfPlayError{}), do: true
  def error?(%RegistryError{}), do: true
  def error?(%WandBError{}), do: true
  def error?(%TelemetryError{}), do: true
  def error?(%CacheError{}), do: true
  def error?(%RecoveryError{}), do: true
  def error?(%YamlError{}), do: true
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
  def wrap(:queue_full, :checkpoint, opts), do: CheckpointError.new(:queue_full, opts)

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
  def wrap(:port_closed, :bridge, opts), do: BridgeError.new(:port_closed, opts)
  def wrap(:protocol_error, :bridge, opts), do: BridgeError.new(:protocol_error, opts)
  def wrap(:unexpected_response, :bridge, opts), do: BridgeError.new(:protocol_error, opts)

  def wrap(:insufficient_data, :data, opts), do: DataError.new(:insufficient_data, opts)
  def wrap(:parse_failed, :data, opts), do: DataError.new(:parse_failed, opts)
  def wrap(:python_not_found, :data, opts), do: DataError.new(:python_not_found, opts)
  def wrap(:script_failed, :data, opts), do: DataError.new(:script_failed, opts)
  def wrap(:empty_dataset, :data, opts), do: DataError.new(:empty_dataset, opts)

  def wrap(:no_policy_loaded, :agent, opts), do: AgentError.new(:no_policy_loaded, opts)
  def wrap(:not_found, :agent, opts), do: AgentError.new(:not_found, opts)
  def wrap(:initialization_failed, :agent, opts), do: AgentError.new(:initialization_failed, opts)
  def wrap(:inference_failed, :agent, opts), do: AgentError.new(:inference_failed, opts)

  def wrap(:not_found, :validation, opts), do: ValidationError.new(:not_found, opts)
  def wrap(:not_regular_file, :validation, opts), do: ValidationError.new(:not_regular_file, opts)
  def wrap(:permission_denied, :validation, opts), do: ValidationError.new(:permission_denied, opts)
  def wrap(:file_too_small, :validation, opts), do: ValidationError.new(:file_too_small, opts)
  def wrap(:empty_file, :validation, opts), do: ValidationError.new(:empty_file, opts)
  def wrap(:invalid_format, :validation, opts), do: ValidationError.new(:invalid_format, opts)
  def wrap(:timeout, :validation, opts), do: ValidationError.new(:timeout, opts)
  def wrap(:enoent, :validation, opts), do: ValidationError.new(:not_found, opts)
  def wrap(:eacces, :validation, opts), do: ValidationError.new(:permission_denied, opts)

  def wrap(:not_found, :league, opts), do: LeagueError.new(:not_found, opts)
  def wrap(:already_registered, :league, opts), do: LeagueError.new(:already_registered, opts)
  def wrap(:no_model, :league, opts), do: LeagueError.new(:no_model, opts)
  def wrap(:match_failed, :league, opts), do: LeagueError.new(:match_failed, opts)

  def wrap(:game_not_started, :self_play, opts), do: SelfPlayError.new(:game_not_started, opts)
  def wrap(:game_finished, :self_play, opts), do: SelfPlayError.new(:game_finished, opts)
  def wrap(:no_current_policy, :self_play, opts), do: SelfPlayError.new(:no_current_policy, opts)
  def wrap(:not_found, :self_play, opts), do: SelfPlayError.new(:not_found, opts)
  def wrap(:dolphin_failed, :self_play, opts), do: SelfPlayError.new(:dolphin_failed, opts)

  def wrap(:not_found, :registry, opts), do: RegistryError.new(:not_found, opts)
  def wrap(:missing_required_field, :registry, opts), do: RegistryError.new(:missing_required_field, opts)
  def wrap(:no_models_with_metric, :registry, opts), do: RegistryError.new(:no_models_with_metric, opts)
  def wrap(:write_failed, :registry, opts), do: RegistryError.new(:write_failed, opts)

  def wrap(:no_api_key, :wandb, opts), do: WandBError.new(:no_api_key, opts)
  def wrap(:run_already_active, :wandb, opts), do: WandBError.new(:run_already_active, opts)
  def wrap(:no_active_run, :wandb, opts), do: WandBError.new(:no_active_run, opts)
  def wrap(:initialization_failed, :wandb, opts), do: WandBError.new(:initialization_failed, opts)

  def wrap(:collector_not_started, :telemetry, opts), do: TelemetryError.new(:collector_not_started, opts)
  def wrap(:invalid_metric, :telemetry, opts), do: TelemetryError.new(:invalid_metric, opts)

  def wrap(:not_found, :cache, opts), do: CacheError.new(:not_found, opts)
  def wrap(:invalid_header, :cache, opts), do: CacheError.new(:invalid_header, opts)
  def wrap(:corrupted, :cache, opts), do: CacheError.new(:corrupted, opts)
  def wrap(:write_failed, :cache, opts), do: CacheError.new(:write_failed, opts)

  def wrap(:invalid_marker, :recovery, opts), do: RecoveryError.new(:invalid_marker, opts)
  def wrap(:no_checkpoint, :recovery, opts), do: RecoveryError.new(:no_checkpoint, opts)
  def wrap(:state_mismatch, :recovery, opts), do: RecoveryError.new(:state_mismatch, opts)

  def wrap(:file_not_found, :yaml, opts), do: YamlError.new(:file_not_found, opts)
  def wrap(:invalid_format, :yaml, opts), do: YamlError.new(:invalid_format, opts)
  def wrap(:invalid_yaml_format, :yaml, opts), do: YamlError.new(:invalid_format, opts)
  def wrap(:invalid_value, :yaml, opts), do: YamlError.new(:invalid_value, opts)
  def wrap(:missing_field, :yaml, opts), do: YamlError.new(:missing_field, opts)

  # ConfigError can also handle validation errors from atom_safety
  def wrap(:invalid_value, :config, opts), do: ConfigError.new(:invalid_value, opts)
  def wrap(:not_existing, :config, opts), do: ConfigError.new(:invalid_value, opts)

  def wrap(error, _category, _opts), do: error
end
