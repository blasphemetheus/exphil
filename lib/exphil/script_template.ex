defmodule ExPhil.ScriptTemplate do
  @moduledoc """
  Shared utilities for ExPhil scripts to reduce boilerplate.

  Provides standardized environment setup, validation, and startup routines
  that complement the `ExPhil.CLI` module for argument parsing.

  ## Quick Start

  ```elixir
  alias ExPhil.ScriptTemplate
  alias ExPhil.CLI
  alias ExPhil.Training.Output

  # 1. Setup environment FIRST (before any EXLA operations)
  ScriptTemplate.setup_environment()

  # 2. Parse arguments
  opts = CLI.parse_args(System.argv(),
    flags: [:verbosity, :replay, :checkpoint],
    extra: [custom_flag: :string]
  )

  # 3. Setup verbosity (before any output)
  CLI.setup_verbosity(opts)

  # 4. Show help if requested
  CLI.maybe_show_help(opts, "my_script.exs", [:verbosity, :replay, :checkpoint])

  # 5. Validate inputs
  ScriptTemplate.validate_dir!(opts[:replays], "Replays directory")
  ScriptTemplate.validate_file!(opts[:checkpoint], "Checkpoint")

  # 6. Print startup banner and config
  ScriptTemplate.print_startup("My Script", [
    {"Replays", opts[:replays]},
    {"Checkpoint", opts[:checkpoint]}
  ])

  # Now do your work...
  ```

  ## Simplified One-Liner

  For simple scripts, use `setup!/2` which combines steps 1-5:

  ```elixir
  opts = ScriptTemplate.setup!(System.argv(),
    name: "my_script.exs",
    flags: [:verbosity, :replay],
    extra: [custom: :string],
    validate: [
      dir: [:replays],
      file: [:checkpoint]
    ]
  )
  ```

  ## See Also

  - `ExPhil.CLI` - Argument parsing and flag definitions
  - `ExPhil.Training.Output` - Formatted output utilities
  """

  alias ExPhil.CLI
  alias ExPhil.Training.Output
  alias ExPhil.Training.GPUUtils

  require Logger

  # ============================================================================
  # Environment Setup
  # ============================================================================

  @doc """
  Setup environment variables for optimal performance.

  **IMPORTANT**: Call this BEFORE any EXLA operations to ensure environment
  variables take effect. EXLA reads these at compile/startup time.

  ## What it does

  1. Enables XLA multi-threading for CPU operations
  2. Suppresses noisy TensorFlow/XLA logs (unless `--verbose`)
  3. Configures GPU memory allocation (if available)

  ## Options

  - `:verbose` - If true, don't suppress TF/XLA logs (default: false)
  - `:gpu_prealloc` - Pre-allocate GPU memory fraction (default: nil)

  ## Example

      # At the very start of your script:
      ScriptTemplate.setup_environment()

      # Or with options:
      ScriptTemplate.setup_environment(verbose: "--verbose" in System.argv())
  """
  @spec setup_environment(keyword()) :: :ok
  def setup_environment(opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)
    gpu_prealloc = Keyword.get(opts, :gpu_prealloc)

    # Enable XLA multi-threading for CPU
    setup_xla_threading()

    # Suppress TF/XLA logs unless verbose
    unless verbose do
      suppress_xla_logs()
    end

    # GPU memory allocation
    if gpu_prealloc do
      System.put_env("TF_GPU_ALLOCATOR", "cuda_malloc_async")
      System.put_env("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    end

    # Set Elixir inspect limits for cleaner output
    Application.put_env(:elixir, :inspect, limit: 10, printable_limit: 100)

    :ok
  end

  defp setup_xla_threading do
    xla_flags = System.get_env("XLA_FLAGS", "")

    unless String.contains?(xla_flags, "xla_cpu_multi_thread_eigen") do
      new_flags = "#{xla_flags} --xla_cpu_multi_thread_eigen=true" |> String.trim()
      System.put_env("XLA_FLAGS", new_flags)
    end
  end

  defp suppress_xla_logs do
    # Suppress TensorFlow C++ logs
    System.put_env("TF_CPP_MIN_LOG_LEVEL", "2")

    # Suppress PTXAS warnings (CUDA compiler)
    System.put_env("PTXAS_OPTIONS", "--warning-level 0")
  end

  # ============================================================================
  # Validation Helpers
  # ============================================================================

  @doc """
  Validate that a directory exists, halt with error if not.

  ## Example

      ScriptTemplate.validate_dir!(opts[:replays], "Replays directory")
      # If missing, prints:
      # ❌ Replays directory not found: ./replays
      # and exits with code 1
  """
  @spec validate_dir!(String.t() | nil, String.t()) :: :ok | no_return()
  def validate_dir!(nil, description) do
    Output.error("#{description} not specified")
    System.halt(1)
  end

  def validate_dir!(path, description) do
    unless File.dir?(path) do
      Output.error("#{description} not found: #{path}")
      System.halt(1)
    end

    :ok
  end

  @doc """
  Validate that a file exists, halt with error if not.

  ## Example

      ScriptTemplate.validate_file!(opts[:checkpoint], "Checkpoint")
  """
  @spec validate_file!(String.t() | nil, String.t()) :: :ok | no_return()
  def validate_file!(nil, description) do
    Output.error("#{description} not specified")
    System.halt(1)
  end

  def validate_file!(path, description) do
    unless File.exists?(path) do
      Output.error("#{description} not found: #{path}")
      System.halt(1)
    end

    :ok
  end

  @doc """
  Validate that a value is present (not nil), halt with error if not.

  ## Example

      ScriptTemplate.validate_required!(opts[:output], "--output")
  """
  @spec validate_required!(term(), String.t()) :: :ok | no_return()
  def validate_required!(nil, name) do
    Output.error("Required argument missing: #{name}")
    System.halt(1)
  end

  def validate_required!(_value, _name), do: :ok

  @doc """
  Validate that a numeric value is positive, halt with error if not.

  ## Example

      ScriptTemplate.validate_positive!(opts[:epochs], "--epochs")
  """
  @spec validate_positive!(number() | nil, String.t()) :: :ok | no_return()
  def validate_positive!(nil, name), do: validate_required!(nil, name)

  def validate_positive!(value, name) when value <= 0 do
    Output.error("#{name} must be positive, got: #{value}")
    System.halt(1)
  end

  def validate_positive!(_value, _name), do: :ok

  @doc """
  Validate that a value is in a range, halt with error if not.

  ## Example

      ScriptTemplate.validate_range!(opts[:temperature], 0.0, 2.0, "--temperature")
  """
  @spec validate_range!(number() | nil, number(), number(), String.t()) :: :ok | no_return()
  def validate_range!(nil, _min, _max, name), do: validate_required!(nil, name)

  def validate_range!(value, min, max, name) when value < min or value > max do
    Output.error("#{name} must be between #{min} and #{max}, got: #{value}")
    System.halt(1)
  end

  def validate_range!(_value, _min, _max, _name), do: :ok

  # ============================================================================
  # Startup Helpers
  # ============================================================================

  @doc """
  Print standardized script startup with banner, config, and GPU status.

  ## Example

      ScriptTemplate.print_startup("Model Evaluation", [
        {"Checkpoint", opts[:checkpoint]},
        {"Replays", opts[:replays]},
        {"Batch size", opts[:batch_size]}
      ])

  ## Output

      ╔════════════════════════════════════════════════════════════╗
      ║                    Model Evaluation                        ║
      ╚════════════════════════════════════════════════════════════╝

      Configuration:
        Checkpoint: checkpoints/model.axon
        Replays: ./replays
        Batch size: 64

        GPU: 4.2/8.0 GB (52%) | Util: 45%
  """
  @spec print_startup(String.t(), [{String.t(), term()}], keyword()) :: :ok
  def print_startup(title, config_items, opts \\ []) do
    show_gpu = Keyword.get(opts, :show_gpu, true)

    Output.banner(title)
    Output.config(config_items)

    if show_gpu do
      Output.puts("")
      Output.puts("  #{GPUUtils.memory_status_string()}")
    end

    Output.puts("")
    :ok
  end

  # ============================================================================
  # Combined Setup (Convenience)
  # ============================================================================

  @doc """
  All-in-one setup for simple scripts.

  Combines environment setup, argument parsing, verbosity setup, help display,
  and validation into a single call. Returns parsed options.

  ## Options

  - `:name` - Script name for help text (required)
  - `:flags` - CLI flag groups (default: `[:verbosity, :common]`)
  - `:extra` - Extra OptionParser flags
  - `:defaults` - Override default values
  - `:validate` - Validation checks:
    - `:dir` - List of keys that must be valid directories
    - `:file` - List of keys that must be valid files
    - `:required` - List of keys that must be present

  ## Example

      opts = ScriptTemplate.setup!(System.argv(),
        name: "train.exs",
        flags: [:verbosity, :replay, :training],
        extra: [custom: :string],
        defaults: [batch_size: 128],
        validate: [
          dir: [:replays],
          file: [:checkpoint],
          required: [:output]
        ]
      )

  ## Returns

  Parsed options keyword list, or exits if help is requested or validation fails.
  """
  @spec setup!(list(String.t()), keyword()) :: keyword() | no_return()
  def setup!(argv, options) do
    name = Keyword.fetch!(options, :name)
    flags = Keyword.get(options, :flags, [:verbosity, :common])
    extra = Keyword.get(options, :extra, [])
    defaults = Keyword.get(options, :defaults, [])
    validations = Keyword.get(options, :validate, [])

    # 1. Setup environment (must be before EXLA)
    verbose_early = "--verbose" in argv or "-v" in argv
    setup_environment(verbose: verbose_early)

    # 2. Parse arguments
    opts = CLI.parse_args(argv,
      flags: flags,
      extra: extra,
      defaults: defaults
    )

    # 3. Setup verbosity
    CLI.setup_verbosity(opts)

    # 4. Show help if requested
    CLI.maybe_show_help(opts, name, flags)

    # 5. Run validations
    run_validations!(opts, validations)

    opts
  end

  defp run_validations!(opts, validations) do
    # Validate directories
    for key <- Keyword.get(validations, :dir, []) do
      value = opts[key]
      description = key |> to_string() |> String.replace("_", " ") |> String.capitalize()
      validate_dir!(value, description)
    end

    # Validate files
    for key <- Keyword.get(validations, :file, []) do
      value = opts[key]
      # Skip nil values unless also in required
      if value do
        description = key |> to_string() |> String.replace("_", " ") |> String.capitalize()
        validate_file!(value, description)
      end
    end

    # Validate required
    for key <- Keyword.get(validations, :required, []) do
      value = opts[key]
      flag = "--#{key |> to_string() |> String.replace("_", "-")}"
      validate_required!(value, flag)
    end

    :ok
  end

  # ============================================================================
  # Progress and Timing Helpers
  # ============================================================================

  @doc """
  Run a block with timing and display the duration.

  ## Example

      {result, duration_ms} = ScriptTemplate.timed("Loading replays") do
        Data.load_replays(...)
      end

  ## Output

      [12:34:56] Loading replays... done! (2.3s)
  """
  defmacro timed(description, do: block) do
    quote do
      require ExPhil.Training.Output
      ExPhil.Training.Output.timed(unquote(description)) do
        unquote(block)
      end
    end
  end

  @doc """
  Print a step indicator for multi-phase scripts.

  ## Example

      ScriptTemplate.step(1, 4, "Loading data")
      ScriptTemplate.step(2, 4, "Training model")
      ScriptTemplate.step(3, 4, "Evaluating")
      ScriptTemplate.step(4, 4, "Saving results")
  """
  @spec step(pos_integer(), pos_integer(), String.t()) :: :ok
  def step(current, total, description) do
    Output.step(current, total, description)
  end

  @doc """
  Print a success message.

  ## Example

      ScriptTemplate.success("Training complete!")
  """
  @spec success(String.t()) :: :ok
  def success(message) do
    Output.success(message)
  end

  @doc """
  Print an error message (does not exit).

  ## Example

      ScriptTemplate.error("Failed to load checkpoint")
  """
  @spec error(String.t()) :: :ok
  def error(message) do
    Output.error(message)
  end
end
