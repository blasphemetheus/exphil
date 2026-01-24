defmodule ExPhil.Training.Stacktrace do
  @moduledoc """
  Stack trace simplification for better error readability.

  Filters out internal Nx/EXLA/Axon frames to show only user-relevant
  code in error messages. This makes debugging much easier since you
  see your code instead of pages of internal framework calls.

  ## Usage

      try do
        some_training_code()
      rescue
        e ->
          Stacktrace.format_exception(e, __STACKTRACE__)
          |> IO.puts()
      end

  ## What Gets Filtered

  - Nx.Defn internal compilation frames
  - EXLA JIT/compilation frames
  - Axon internal frames
  - Erlang/OTP internals (gen_server, supervisor, etc.)
  - Anonymous function wrappers

  ## Configuration

  Set `EXPHIL_FULL_STACKTRACE=1` to disable filtering and see full traces.
  """

  alias ExPhil.Training.Output

  # Modules to filter from stack traces (internal implementation details)
  @filtered_modules [
    # Nx internals
    Nx.Defn,
    Nx.Defn.Compiler,
    Nx.Defn.Evaluator,
    Nx.Defn.Expr,
    Nx.Defn.Grad,
    Nx.Defn.Tree,
    Nx.LinAlg,
    Nx.BinaryBackend,
    Nx.Shared,

    # EXLA internals
    EXLA,
    EXLA.Backend,
    EXLA.Defn,
    EXLA.Defn.Buffers,
    EXLA.Defn.Expr,
    EXLA.Defn.Outfeed,
    EXLA.Executable,
    EXLA.MLIR.Module,
    EXLA.MLIR.Value,

    # Axon internals
    Axon,
    Axon.Compiler,
    Axon.Loop,
    Axon.Loop.State,
    Axon.ModelState,

    # Polaris (optimizer)
    Polaris.Optimizers,
    Polaris.Updates,

    # Erlang/OTP internals
    :gen_server,
    :gen,
    :proc_lib,
    :supervisor,
    :application_master,
    :application_controller,

    # Elixir internals
    Enum,
    Stream,
    Task,
    Task.Supervised,
    Agent,
    GenServer,
    Supervisor
  ]

  # File path patterns to filter (as strings, converted to regex at runtime)
  @filtered_path_patterns [
    "deps/nx/",
    "deps/exla/",
    "deps/axon/",
    "deps/polaris/",
    "lib/elixir/",
    "_build/.+/deps/",
    "otp.*/lib/"
  ]

  @doc """
  Format an exception with a simplified stack trace.

  Returns a string suitable for printing to stderr.
  """
  @spec format_exception(Exception.t(), list()) :: String.t()
  def format_exception(exception, stacktrace) do
    if show_full_trace?() do
      Exception.format(:error, exception, stacktrace)
    else
      format_simplified(exception, stacktrace)
    end
  end

  @doc """
  Print an exception with simplified stack trace to stderr.
  """
  @spec print_exception(Exception.t(), list()) :: :ok
  def print_exception(exception, stacktrace) do
    Output.puts_raw("")
    Output.puts_raw(Output.colorize("Error: #{Exception.message(exception)}", :red))

    simplified = simplify_stacktrace(stacktrace)

    if simplified != [] do
      Output.puts_raw("")
      Output.puts_raw(Output.colorize("Stacktrace (simplified):", :bold))

      Enum.each(simplified, fn frame ->
        Output.puts_raw("  #{format_frame(frame)}")
      end)

      hidden = length(stacktrace) - length(simplified)
      if hidden > 0 do
        Output.puts_raw(Output.colorize("  ... #{hidden} internal frames hidden", :dim))
        Output.puts_raw(Output.colorize("  (set EXPHIL_FULL_STACKTRACE=1 for full trace)", :dim))
      end
    end

    :ok
  end

  @doc """
  Simplify a stack trace by removing internal frames.

  Keeps frames from:
  - ExPhil modules
  - User's project modules
  - Top-level scripts

  Filters out:
  - Nx/EXLA/Axon internals
  - Erlang/OTP internals
  - Standard library internals
  """
  @spec simplify_stacktrace(list()) :: list()
  def simplify_stacktrace(stacktrace) do
    stacktrace
    |> Enum.filter(&keep_frame?/1)
    |> Enum.take(10)  # Limit to 10 most relevant frames
  end

  @doc """
  Check if a stack frame should be kept (not filtered).
  """
  @spec keep_frame?(tuple()) :: boolean()
  def keep_frame?({module, _function, _arity, location}) do
    keep_module?(module) and keep_location?(location)
  end

  def keep_frame?(_), do: true

  @doc """
  Check if full stack trace should be shown.
  """
  @spec show_full_trace?() :: boolean()
  def show_full_trace? do
    System.get_env("EXPHIL_FULL_STACKTRACE") == "1"
  end

  # ============================================================
  # Private Helpers
  # ============================================================

  defp format_simplified(exception, stacktrace) do
    message = Exception.message(exception)
    simplified = simplify_stacktrace(stacktrace)

    frames_str = simplified
    |> Enum.map(&format_frame/1)
    |> Enum.join("\n    ")

    hidden = length(stacktrace) - length(simplified)
    hidden_note = if hidden > 0 do
      "\n    ... #{hidden} internal frames hidden (set EXPHIL_FULL_STACKTRACE=1 for full trace)"
    else
      ""
    end

    """
    ** (#{inspect(exception.__struct__)}) #{message}
        #{frames_str}#{hidden_note}
    """
  end

  defp format_frame({module, function, arity, location}) do
    file = Keyword.get(location, :file, "nofile") |> to_string()
    line = Keyword.get(location, :line, 0)

    # Shorten paths
    file = shorten_path(file)

    arity_str = if is_list(arity), do: length(arity), else: arity

    "#{inspect(module)}.#{function}/#{arity_str} (#{file}:#{line})"
  end

  defp format_frame(other), do: inspect(other)

  defp keep_module?(module) when is_atom(module) do
    module_str = to_string(module)

    cond do
      # Always keep ExPhil modules
      String.starts_with?(module_str, "Elixir.ExPhil") -> true

      # Keep user scripts
      String.starts_with?(module_str, "Elixir.Mix.Tasks") -> true

      # Keep anonymous functions (usually user code)
      module_str =~ ~r/^Elixir\.-/ -> true

      # Filter known internal modules
      module in @filtered_modules -> false

      # Filter Nx.* and EXLA.* submodules
      String.starts_with?(module_str, "Elixir.Nx.") -> false
      String.starts_with?(module_str, "Elixir.EXLA.") -> false
      String.starts_with?(module_str, "Elixir.Axon.") -> false

      # Keep everything else
      true -> true
    end
  end

  defp keep_module?(_), do: true

  defp keep_location?(location) do
    file = Keyword.get(location, :file, "") |> to_string()

    # Keep if not matching any filtered path pattern
    not Enum.any?(@filtered_path_patterns, fn pattern ->
      String.contains?(file, pattern) or
        (String.contains?(pattern, ".+") and Regex.match?(~r/#{pattern}/, file))
    end)
  end

  defp shorten_path(path) do
    path
    |> String.replace(~r/.*\/lib\/exphil\//, "lib/exphil/")
    |> String.replace(~r/.*\/scripts\//, "scripts/")
    |> String.replace(~r/.*\/test\//, "test/")
  end
end
