defmodule ExPhil.CLI do
  @moduledoc """
  Shared utilities for CLI scripts.

  Provides standardized argument parsing, verbosity control, and common flag
  definitions to ensure consistent UX across all ExPhil scripts.

  ## Usage

      # In a script:
      alias ExPhil.CLI

      # Parse with common flag groups
      opts = CLI.parse_args(args,
        flags: [:verbosity, :replay, :checkpoint],
        extra: [custom_flag: :string]
      )

      # Setup verbosity (call early, before any EXLA operations)
      CLI.setup_verbosity(opts)

      # Access parsed options
      replays_dir = opts[:replays]

  ## Flag Groups

  - `:verbosity` - `--quiet`, `--verbose` flags
  - `:replay` - `--replays`, `--max-files`, `--character` flags
  - `:checkpoint` - `--checkpoint`, `--policy` flags
  - `:training` - `--batch-size`, `--epochs`, `--lr` flags
  - `:evaluation` - `--batch-size`, `--player`, `--detailed` flags
  - `:analysis` - `--top-actions`, `--show-stages`, `--show-positions`, `--show-buttons` flags
  """

  require Logger
  alias ExPhil.Training.Output

  # ============================================================================
  # Flag Definitions
  # ============================================================================

  # Flag definitions with metadata for parsing and help generation.
  #
  # Each flag has:
  # - :name - Keyword key for the option
  # - :flag - CLI flag string (e.g., "--replays")
  # - :type - :string, :integer, :float, :boolean, :atom
  # - :short - Optional single-letter alias
  # - :default - Default value
  # - :desc - Description for help text
  # - :group - Flag group(s) this belongs to
  @flag_definitions [
    # Verbosity flags
    %{name: :quiet, flag: "--quiet", type: :boolean, short: "q", default: false,
      desc: "Suppress non-essential output", group: [:verbosity]},
    %{name: :verbose, flag: "--verbose", type: :boolean, short: "v", default: false,
      desc: "Show debug output including XLA logs", group: [:verbosity]},

    # Replay flags
    %{name: :replays, flag: "--replays", type: :string, short: "r", default: "./replays",
      desc: "Path to replay files or directory", group: [:replay]},
    %{name: :max_files, flag: "--max-files", type: :integer, short: "m", default: nil,
      desc: "Maximum replay files to process", group: [:replay]},
    %{name: :character, flag: "--character", type: :atom, short: nil, default: nil,
      desc: "Filter replays by character", group: [:replay]},
    %{name: :player_port, flag: "--player", type: :integer, short: nil, default: 1,
      desc: "Player port (1-4)", group: [:replay, :evaluation]},

    # Checkpoint flags
    %{name: :checkpoint, flag: "--checkpoint", type: :string, short: "c", default: nil,
      desc: "Path to checkpoint (.axon) file", group: [:checkpoint]},
    %{name: :policy, flag: "--policy", type: :string, short: "p", default: nil,
      desc: "Path to exported policy (.bin) file", group: [:checkpoint]},

    # Training flags
    %{name: :batch_size, flag: "--batch-size", type: :integer, short: "b", default: 64,
      desc: "Batch size", group: [:training, :evaluation]},
    %{name: :epochs, flag: "--epochs", type: :integer, short: "e", default: 10,
      desc: "Number of training epochs", group: [:training]},
    %{name: :learning_rate, flag: "--lr", type: :float, short: nil, default: 1.0e-4,
      desc: "Learning rate", group: [:training]},

    # Evaluation flags
    %{name: :detailed, flag: "--detailed", type: :boolean, short: nil, default: false,
      desc: "Show detailed per-component metrics", group: [:evaluation]},
    %{name: :output, flag: "--output", type: :string, short: "o", default: nil,
      desc: "Save results to JSON file", group: [:evaluation]},

    # Common flags
    %{name: :help, flag: "--help", type: :boolean, short: "h", default: false,
      desc: "Show help message", group: [:common]},

    # Analysis flags (for analyze_replays.exs)
    %{name: :top_actions, flag: "--top-actions", type: :integer, short: "n", default: 15,
      desc: "Show top N actions in report", group: [:analysis]},
    %{name: :show_stages, flag: "--show-stages", type: :boolean, short: nil, default: true,
      desc: "Show stage breakdown", group: [:analysis]},
    %{name: :show_positions, flag: "--show-positions", type: :boolean, short: nil, default: true,
      desc: "Show position analysis", group: [:analysis]},
    %{name: :show_buttons, flag: "--show-buttons", type: :boolean, short: nil, default: true,
      desc: "Show button press analysis", group: [:analysis]}
  ]

  @doc "Get all flag definitions"
  def flag_definitions, do: @flag_definitions

  @doc "Get flag definitions for specific groups"
  def flags_for_groups(groups) when is_list(groups) do
    @flag_definitions
    |> Enum.filter(fn flag ->
      Enum.any?(flag.group, &(&1 in groups))
    end)
  end

  # ============================================================================
  # Argument Parsing
  # ============================================================================

  @doc """
  Parse command-line arguments with standardized flag groups.

  ## Options

  - `:flags` - List of flag groups to include (e.g., `[:verbosity, :replay]`)
  - `:extra` - Additional flags specific to this script (OptionParser format)
  - `:defaults` - Override default values

  ## Examples

      # Basic usage with flag groups
      opts = CLI.parse_args(args, flags: [:verbosity, :replay])

      # With extra script-specific flags
      opts = CLI.parse_args(args,
        flags: [:verbosity, :checkpoint],
        extra: [temporal: :boolean, backbone: :string]
      )

      # Override defaults
      opts = CLI.parse_args(args,
        flags: [:training],
        defaults: [batch_size: 512]
      )
  """
  def parse_args(args, options \\ []) do
    groups = Keyword.get(options, :flags, [:verbosity, :common])
    extra_flags = Keyword.get(options, :extra, [])
    default_overrides = Keyword.get(options, :defaults, [])

    # Build OptionParser spec from flag groups
    group_flags = flags_for_groups(groups)
    {strict_spec, aliases} = build_option_spec(group_flags, extra_flags)

    # Parse arguments
    {parsed, positional, invalid} = OptionParser.parse(args,
      strict: strict_spec,
      aliases: aliases
    )

    # Warn about invalid flags
    for {flag, _} <- invalid do
      IO.puts(:stderr, "Warning: Unknown flag #{flag}")
    end

    # Build defaults from flag definitions
    defaults =
      group_flags
      |> Enum.map(fn flag -> {flag.name, flag.default} end)
      |> Keyword.new()
      |> Keyword.merge(default_overrides)

    # Merge parsed over defaults
    opts =
      defaults
      |> Keyword.merge(parsed)
      |> Keyword.put(:_positional, positional)

    opts
  end

  @doc """
  Parse arguments using raw arg list (like Config.parse_args).

  Lower-level parsing that works directly with the arg list instead of
  OptionParser. Useful when you need more control or compatibility with
  existing Config-style parsing.
  """
  def parse_raw(args, defaults \\ []) when is_list(args) do
    defaults
    |> parse_string_arg(args, "--replays", :replays)
    |> parse_int_arg(args, "--max-files", :max_files)
    |> parse_int_arg(args, "--batch-size", :batch_size)
    |> parse_int_arg(args, "--epochs", :epochs)
    |> parse_int_arg(args, "--player", :player_port)
    |> parse_float_arg(args, "--lr", :learning_rate)
    |> parse_atom_arg(args, "--character", :character)
    |> parse_string_arg(args, "--checkpoint", :checkpoint)
    |> parse_string_arg(args, "--policy", :policy)
    |> parse_string_arg(args, "--output", :output)
    |> parse_flag(args, "--quiet", :quiet)
    |> parse_flag(args, "--verbose", :verbose)
    |> parse_flag(args, "--detailed", :detailed)
    |> parse_flag(args, "--help", :help)
  end

  # Build OptionParser spec from flag definitions
  defp build_option_spec(flags, extra_flags) do
    strict =
      flags
      |> Enum.map(fn flag ->
        type = case flag.type do
          :atom -> :string  # OptionParser doesn't have :atom
          other -> other
        end
        {flag.name, type}
      end)
      |> Keyword.new()
      |> Keyword.merge(extra_flags)

    aliases =
      flags
      |> Enum.filter(& &1.short)
      |> Enum.map(fn flag ->
        {String.to_atom(flag.short), flag.name}
      end)
      |> Keyword.new()

    {strict, aliases}
  end

  # ============================================================================
  # Public Parsing Helpers
  # ============================================================================

  @doc "Check if a flag is present in args"
  def has_flag?(args, flag) do
    flag in args
  end

  @doc "Get the value after a flag, or nil if not present"
  def get_arg_value(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> nil
      idx ->
        case Enum.at(args, idx + 1) do
          nil -> nil
          value when is_binary(value) ->
            if String.starts_with?(value, "--"), do: nil, else: value
        end
    end
  end

  @doc "Parse a string argument"
  def parse_string_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, value)
    end
  end

  @doc "Parse an integer argument"
  def parse_int_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_integer(value))
    end
  end

  @doc "Parse a float argument (supports scientific notation like 1e-4)"
  def parse_float_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value ->
        case Float.parse(value) do
          {float, ""} -> Keyword.put(opts, key, float)
          _ -> raise ArgumentError, "Invalid float for #{flag}: #{value}"
        end
    end
  end

  @doc "Parse an atom argument"
  def parse_atom_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value -> Keyword.put(opts, key, String.to_atom(value))
    end
  end

  @doc "Parse a boolean flag (presence = true)"
  def parse_flag(opts, args, flag, key) do
    if has_flag?(args, flag) do
      Keyword.put(opts, key, true)
    else
      opts
    end
  end

  @doc "Parse a comma-separated list of integers"
  def parse_int_list_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value ->
        list = value |> String.split(",") |> Enum.map(&String.to_integer(String.trim(&1)))
        Keyword.put(opts, key, list)
    end
  end

  @doc "Parse a comma-separated list of atoms"
  def parse_atom_list_arg(opts, args, flag, key) do
    case get_arg_value(args, flag) do
      nil -> opts
      value ->
        list = value |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
        Keyword.put(opts, key, list)
    end
  end

  # ============================================================================
  # Verbosity Control
  # ============================================================================

  @doc """
  Setup verbosity based on --quiet and --verbose flags.

  IMPORTANT: Call this early in the script, BEFORE any EXLA operations,
  to suppress XLA initialization logs.

  ## Levels

  - `--quiet` (level 0): Errors only, suppress XLA logs
  - default (level 1): Normal output, suppress XLA info logs
  - `--verbose` (level 2): Full debug output including XLA logs

  ## Example

      opts = CLI.parse_args(args, flags: [:verbosity])
      CLI.setup_verbosity(opts)
      # Now safe to do EXLA operations
  """
  def setup_verbosity(opts) do
    cond do
      opts[:quiet] ->
        Logger.configure(level: :error)
        Output.set_verbosity(0)
        :quiet

      opts[:verbose] ->
        # Don't change Logger level - show all logs
        Output.set_verbosity(2)
        :verbose

      true ->
        # Default: suppress XLA info logs but show warnings
        Logger.configure(level: :warning)
        Output.set_verbosity(1)
        :normal
    end
  end

  @doc "Get verbosity level from opts (0=quiet, 1=normal, 2=verbose)"
  def verbosity_level(opts) do
    cond do
      opts[:quiet] -> 0
      opts[:verbose] -> 2
      true -> 1
    end
  end

  # ============================================================================
  # Help Text Generation
  # ============================================================================

  @doc """
  Generate help text for specified flag groups.

  ## Example

      help = CLI.help_text([:verbosity, :replay, :checkpoint])
      IO.puts(help)
  """
  def help_text(groups) do
    flags = flags_for_groups(groups)

    flags
    |> Enum.map(&format_flag_help/1)
    |> Enum.join("\n")
  end

  defp format_flag_help(flag) do
    short = if flag.short, do: "-#{flag.short}, ", else: "    "
    default = if flag.default != nil && flag.default != false do
      " (default: #{inspect(flag.default)})"
    else
      ""
    end

    "  #{short}#{flag.flag}\t#{flag.desc}#{default}"
  end

  @doc """
  Print help and exit if --help flag is present.

  ## Example

      CLI.maybe_show_help(opts, "eval_model.exs", [:verbosity, :replay], fn ->
        IO.puts("Additional examples...")
      end)
  """
  def maybe_show_help(opts, script_name, groups, extra_fn \\ nil) do
    if opts[:help] do
      IO.puts("""
      Usage: mix run scripts/#{script_name} [options]

      Options:
      #{help_text(groups)}
      """)

      if extra_fn, do: extra_fn.()

      System.halt(0)
    end
  end

  # ============================================================================
  # Script Utilities
  # ============================================================================

  @doc """
  Standard script banner.

  ## Example

      CLI.banner("Model Evaluation")
      # Outputs:
      # ╔════════════════════════════════════════════════════════════╗
      # ║                    Model Evaluation                        ║
      # ╚════════════════════════════════════════════════════════════╝
  """
  def banner(title) do
    Output.banner(title)
  end

  @doc """
  Print configuration summary.

  ## Example

      CLI.config([
        {"Replays", opts[:replays]},
        {"Batch size", opts[:batch_size]},
        {"Output", opts[:output] || "stdout"}
      ])
  """
  def config(items) do
    Output.config(items)
  end

  @doc """
  Validate required options are present.

  ## Example

      CLI.require_options!(opts, [:checkpoint, :replays])
      # Raises if either is nil
  """
  def require_options!(opts, required_keys) do
    missing =
      required_keys
      |> Enum.filter(fn key -> opts[key] == nil end)

    if length(missing) > 0 do
      flags = Enum.map(missing, fn key ->
        case Enum.find(@flag_definitions, &(&1.name == key)) do
          nil -> "--#{key}"
          flag -> flag.flag
        end
      end)

      raise ArgumentError, "Missing required options: #{Enum.join(flags, ", ")}"
    end

    :ok
  end

  @doc """
  Require at least one of the given options.

  ## Example

      CLI.require_one_of!(opts, [:checkpoint, :policy])
  """
  def require_one_of!(opts, keys) do
    present = Enum.filter(keys, fn key -> opts[key] != nil end)

    if length(present) == 0 do
      flags = Enum.map(keys, fn key ->
        case Enum.find(@flag_definitions, &(&1.name == key)) do
          nil -> "--#{key}"
          flag -> flag.flag
        end
      end)

      raise ArgumentError, "Requires at least one of: #{Enum.join(flags, ", ")}"
    end

    :ok
  end
end
