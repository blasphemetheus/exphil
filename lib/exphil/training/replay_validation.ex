defmodule ExPhil.Training.ReplayValidation do
  @moduledoc """
  Replay file validation before training.

  Validates replay files to catch issues early:
  - File existence and readability
  - SLP file format validation
  - Minimum file size checks
  - Quick header parsing for corruption detection

  Designed to run fast (parallel, minimal parsing) to catch issues
  before expensive full replay parsing.
  """

  alias ExPhil.Training.Output

  # Minimum valid SLP file size (header + some frames)
  # A completely empty game with no frames is still ~8KB
  @min_file_size 4096

  @doc """
  Validate a list of replay file paths.

  Returns `{:ok, valid_paths, stats}` with validation statistics,
  or `{:error, reason}` for critical failures.

  ## Options
  - `:parallel` - Run validation in parallel (default: true)
  - `:max_workers` - Max parallel workers (default: schedulers_online)
  - `:show_progress` - Show progress bar (default: true)
  - `:verbose` - Show individual file errors (default: false)

  ## Statistics returned
  - `:total` - Total files checked
  - `:valid` - Number of valid files
  - `:invalid` - Number of invalid files
  - `:errors` - List of `{path, reason}` tuples for invalid files
  - `:skipped` - Files skipped for other reasons

  ## Examples

      iex> ReplayValidation.validate(["replay1.slp", "replay2.slp"])
      {:ok, ["replay1.slp", "replay2.slp"], %{total: 2, valid: 2, invalid: 0, errors: []}}

      iex> ReplayValidation.validate(["missing.slp", "valid.slp"])
      {:ok, ["valid.slp"], %{total: 2, valid: 1, invalid: 1, errors: [{"missing.slp", :not_found}]}}
  """
  @spec validate([String.t()], keyword()) :: {:ok, [String.t()], map()} | {:error, any()}
  def validate(paths, opts \\ []) do
    parallel = Keyword.get(opts, :parallel, true)
    max_workers = Keyword.get(opts, :max_workers, System.schedulers_online())
    show_progress = Keyword.get(opts, :show_progress, true)
    verbose = Keyword.get(opts, :verbose, false)

    total = length(paths)

    if show_progress do
      Output.puts("Validating #{total} replay files...")
    end

    # Validate files
    results =
      if parallel do
        validate_parallel(paths, max_workers)
      else
        Enum.map(paths, &validate_file/1)
      end

    # Partition results
    {valid, invalid} =
      Enum.split_with(results, fn {_path, result} ->
        result == :ok
      end)

    valid_paths = Enum.map(valid, fn {path, _} -> path end)
    errors = Enum.map(invalid, fn {path, {:error, reason}} -> {path, reason} end)

    stats = %{
      total: total,
      valid: length(valid_paths),
      invalid: length(errors),
      errors: errors,
      skipped: 0
    }

    # Report results
    if show_progress do
      report_validation_results(stats, verbose)
    end

    {:ok, valid_paths, stats}
  end

  @doc """
  Validate a single replay file.

  Performs quick validation checks without full parsing:
  1. File exists and is readable
  2. File is not too small
  3. File has valid SLP magic bytes

  Returns `:ok` or `{:error, reason}`.
  """
  @spec validate_file(String.t()) :: {String.t(), :ok | {:error, atom()}}
  def validate_file(path) do
    result =
      with :ok <- check_exists(path),
           :ok <- check_size(path),
           :ok <- check_format(path) do
        :ok
      end

    {path, result}
  end

  @doc """
  Quick validation that just checks if files exist and are readable.

  Faster than full validation, useful for very large datasets.
  """
  @spec quick_validate([String.t()]) :: {:ok, [String.t()], [String.t()]}
  def quick_validate(paths) do
    {valid, invalid} = Enum.split_with(paths, &File.exists?/1)
    {:ok, valid, invalid}
  end

  # Private helpers

  defp validate_parallel(paths, max_workers) do
    paths
    |> Task.async_stream(
      &validate_file/1,
      max_concurrency: max_workers,
      timeout: 10_000
    )
    |> Enum.map(fn
      {:ok, result} -> result
      {:exit, _reason} -> {nil, {:error, :timeout}}
    end)
    |> Enum.reject(fn {path, _} -> is_nil(path) end)
  end

  defp check_exists(path) do
    case File.stat(path) do
      {:ok, %{type: :regular}} -> :ok
      {:ok, %{type: _other}} -> {:error, :not_regular_file}
      {:error, :enoent} -> {:error, :not_found}
      {:error, :eacces} -> {:error, :permission_denied}
      {:error, reason} -> {:error, reason}
    end
  end

  defp check_size(path) do
    case File.stat(path) do
      {:ok, %{size: size}} when size >= @min_file_size -> :ok
      {:ok, %{size: size}} when size < @min_file_size -> {:error, :file_too_small}
      {:error, reason} -> {:error, reason}
    end
  end

  defp check_format(path) do
    case File.open(path, [:read, :binary]) do
      {:ok, file} ->
        result =
          case IO.binread(file, 4) do
            # SLP raw format starts with 0x7B 0x55 (ASCII "{U")
            <<0x7B, 0x55, _, _>> -> :ok
            :eof -> {:error, :empty_file}
            {:error, reason} -> {:error, reason}
            _other -> {:error, :invalid_format}
          end

        File.close(file)
        result

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp report_validation_results(stats, verbose) do
    if stats.invalid == 0 do
      Output.success("All #{stats.valid} replays are valid")
    else
      Output.puts("Validation: #{stats.valid}/#{stats.total} valid, #{stats.invalid} invalid")

      if verbose and stats.invalid > 0 do
        Output.puts_raw("")
        Output.puts_raw("  Invalid files:")

        stats.errors
        # Show first 20
        |> Enum.take(20)
        |> Enum.each(fn {path, reason} ->
          Output.puts_raw("    - #{Path.basename(path)}: #{format_error(reason)}")
        end)

        if length(stats.errors) > 20 do
          Output.puts_raw("    ... and #{length(stats.errors) - 20} more")
        end
      end
    end
  end

  defp format_error(:not_found), do: "file not found"
  defp format_error(:permission_denied), do: "permission denied"
  defp format_error(:file_too_small), do: "file too small (< #{div(@min_file_size, 1024)} KB)"
  defp format_error(:invalid_format), do: "not a valid SLP file"
  defp format_error(:empty_file), do: "empty file"
  defp format_error(:not_regular_file), do: "not a regular file"
  defp format_error(:timeout), do: "validation timeout"
  defp format_error(reason), do: "#{inspect(reason)}"
end
