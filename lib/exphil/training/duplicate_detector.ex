defmodule ExPhil.Training.DuplicateDetector do
  @moduledoc """
  Detect and filter duplicate replay files by content hash.

  Uses MD5 hash of file contents to identify duplicates. This catches:
  - Exact copies with different filenames
  - Files downloaded multiple times
  - Replays in different directories

  ## Usage

      # Filter a list of files
      {unique_files, stats} = DuplicateDetector.filter_duplicates(files)

      # Or use streaming for large file lists
      DuplicateDetector.filter_duplicates_stream(files)
      |> Stream.each(&process/1)
      |> Stream.run()

  ## Performance

  - Uses streaming file reads for large files
  - Processes files in parallel for speed
  - Memory efficient: only stores 16-byte hashes per file

  ## Options

  The `--skip-duplicates` flag (default: true) enables this filtering.
  Use `--no-skip-duplicates` to disable if you want all files.
  """

  alias ExPhil.Training.Output

  @hash_algorithm :md5
  @chunk_size 65_536  # 64KB chunks for streaming hash

  @doc """
  Filter duplicate files from a list, keeping the first occurrence.

  Returns `{unique_files, stats}` where stats contains:
  - `:total` - Total files processed
  - `:unique` - Number of unique files
  - `:duplicates` - Number of duplicates removed
  - `:duplicate_groups` - Map of hash -> list of duplicate paths

  ## Options

    * `:show_progress` - Show progress bar (default: false)
    * `:parallel` - Use parallel processing (default: true)

  """
  @spec filter_duplicates([Path.t()], keyword()) :: {[Path.t()], map()}
  def filter_duplicates(files, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, false)
    parallel = Keyword.get(opts, :parallel, true)

    total = length(files)

    # Compute hashes (optionally in parallel)
    hashed_files = if parallel and total > 10 do
      hash_files_parallel(files, show_progress)
    else
      hash_files_sequential(files, show_progress)
    end

    # Group by hash, keeping track of all files with same hash
    grouped = Enum.group_by(hashed_files, fn {hash, _path} -> hash end, fn {_hash, path} -> path end)

    # Keep first file from each group
    unique_files = grouped
    |> Enum.map(fn {_hash, [first | _rest]} -> first end)
    |> Enum.sort()  # Deterministic ordering

    # Find duplicates (groups with more than one file)
    duplicate_groups = grouped
    |> Enum.filter(fn {_hash, paths} -> length(paths) > 1 end)
    |> Map.new()

    duplicates_count = total - length(unique_files)

    stats = %{
      total: total,
      unique: length(unique_files),
      duplicates: duplicates_count,
      duplicate_groups: duplicate_groups
    }

    {unique_files, stats}
  end

  @doc """
  Stream version that yields unique files as they're found.

  More memory efficient for very large file lists, but doesn't
  provide full duplicate statistics.
  """
  @spec filter_duplicates_stream(Enumerable.t()) :: Enumerable.t()
  def filter_duplicates_stream(files) do
    Stream.transform(files, MapSet.new(), fn file, seen_hashes ->
      case hash_file(file) do
        {:ok, hash} ->
          if MapSet.member?(seen_hashes, hash) do
            {[], seen_hashes}  # Skip duplicate
          else
            {[file], MapSet.put(seen_hashes, hash)}
          end

        {:error, _reason} ->
          # Include files we can't hash (let downstream handle errors)
          {[file], seen_hashes}
      end
    end)
  end

  @doc """
  Compute hash for a single file.
  """
  @spec hash_file(Path.t()) :: {:ok, binary()} | {:error, term()}
  def hash_file(path) do
    try do
      hash = File.stream!(path, @chunk_size)
      |> Enum.reduce(:crypto.hash_init(@hash_algorithm), fn chunk, acc ->
        :crypto.hash_update(acc, chunk)
      end)
      |> :crypto.hash_final()

      {:ok, hash}
    rescue
      e -> {:error, e}
    end
  end

  @doc """
  Print a summary of duplicate detection results.
  """
  @spec print_summary(map()) :: :ok
  def print_summary(stats) do
    if stats.duplicates > 0 do
      pct = Float.round(stats.duplicates / stats.total * 100, 1)
      Output.puts("Duplicate detection: removed #{stats.duplicates}/#{stats.total} files (#{pct}%)")

      if map_size(stats.duplicate_groups) <= 5 do
        # Show details for small number of duplicate groups
        Enum.each(stats.duplicate_groups, fn {_hash, paths} ->
          Output.puts_raw("    #{length(paths)} copies: #{Path.basename(hd(paths))}")
        end)
      else
        Output.puts_raw("    (#{map_size(stats.duplicate_groups)} duplicate groups)")
      end
    else
      Output.puts("Duplicate detection: no duplicates found in #{stats.total} files")
    end

    :ok
  end

  # ============================================================
  # Private Helpers
  # ============================================================

  defp hash_files_sequential(files, show_progress) do
    total = length(files)

    files
    |> Enum.with_index(1)
    |> Enum.map(fn {file, idx} ->
      if show_progress and rem(idx, 100) == 0 do
        Output.progress_bar(idx, total, label: "Hashing")
      end

      case hash_file(file) do
        {:ok, hash} -> {hash, file}
        {:error, _} -> {make_ref(), file}  # Unique ref for unhashable files
      end
    end)
    |> tap(fn _ -> if show_progress, do: Output.progress_done() end)
  end

  defp hash_files_parallel(files, show_progress) do
    total = length(files)

    # Use Task.async_stream for parallel hashing
    files
    |> Task.async_stream(
      fn file ->
        case hash_file(file) do
          {:ok, hash} -> {hash, file}
          {:error, _} -> {make_ref(), file}
        end
      end,
      max_concurrency: System.schedulers_online() * 2,
      ordered: false
    )
    |> Stream.with_index(1)
    |> Enum.map(fn {{:ok, result}, idx} ->
      if show_progress and rem(idx, 100) == 0 do
        Output.progress_bar(idx, total, label: "Hashing")
      end
      result
    end)
    |> tap(fn _ -> if show_progress, do: Output.progress_done() end)
  end
end
