defmodule ExPhil.Training.Config.Checkpoint do
  @moduledoc """
  Checkpoint file safety functions for training configuration.

  Provides utilities for:
  - Checking if checkpoint paths would overwrite existing files
  - Creating backups before overwriting
  - Rotating backup files to prevent accumulation
  - Formatting file metadata for display

  ## Usage

  Before saving a checkpoint, check if the path exists:

      case Checkpoint.check_checkpoint_path(path, overwrite: allow_overwrite) do
        {:ok, :new} ->
          # Safe to save, no existing file
          save_checkpoint(path, model)

        {:ok, :overwrite, info} ->
          # File exists but overwrite allowed
          Checkpoint.backup_checkpoint(path)
          save_checkpoint(path, model)

        {:error, :exists, info} ->
          # File exists and overwrite not allowed
          IO.warn("Checkpoint exists: \#{Checkpoint.format_file_info(info)}")
      end

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Imitation` - Training loop that uses checkpointing
  """

  @doc """
  Check if a checkpoint path would overwrite an existing file.

  Returns `{:ok, :new}` if path doesn't exist,
  `{:ok, :overwrite, info}` if exists and overwrite allowed,
  `{:error, :exists, info}` if exists and overwrite not allowed.

  The `info` map contains file metadata for warning display.

  ## Options

  - `:overwrite` - Whether to allow overwriting (default: false)

  ## Examples

      iex> Checkpoint.check_checkpoint_path("/nonexistent/path.axon")
      {:ok, :new}

  """
  @spec check_checkpoint_path(Path.t(), keyword()) ::
          {:ok, :new} | {:ok, :overwrite, map()} | {:error, :exists, map()}
  def check_checkpoint_path(path, opts \\ []) do
    overwrite = Keyword.get(opts, :overwrite, false)

    case File.stat(path) do
      {:error, :enoent} ->
        {:ok, :new}

      {:ok, stat} ->
        info = %{
          path: path,
          size: stat.size,
          modified: stat.mtime
        }

        if overwrite do
          {:ok, :overwrite, info}
        else
          {:error, :exists, info}
        end
    end
  end

  @doc """
  Format file info for display in collision warnings.

  ## Examples

      iex> info = %{path: "model.axon", size: 45_200_000, modified: {{2026, 1, 23}, {14, 30, 0}}}
      iex> Checkpoint.format_file_info(info)
      "Size: 43.1 MB, Modified: 2026-01-23 14:30:00"

  """
  @spec format_file_info(map()) :: String.t()
  def format_file_info(info) do
    size_str = format_bytes(info.size)
    time_str = format_datetime(info.modified)
    "Size: #{size_str}, Modified: #{time_str}"
  end

  @doc """
  Backup an existing checkpoint before overwriting.

  Creates backups with rotation: file.bak, file.bak.1, file.bak.2, etc.
  Keeps at most `backup_count` versions.

  ## Options

  - `:backup_count` - Maximum number of backups to keep (default: 3)

  ## Returns

  - `{:ok, backup_path}` on success
  - `{:ok, nil}` if nothing to backup (file doesn't exist)
  - `{:error, reason}` on failure

  ## Examples

      iex> Checkpoint.backup_checkpoint("/nonexistent/path.axon")
      {:ok, nil}

  """
  @spec backup_checkpoint(Path.t(), keyword()) :: {:ok, Path.t() | nil} | {:error, term()}
  def backup_checkpoint(path, opts \\ []) do
    backup_count = Keyword.get(opts, :backup_count, 3)

    if File.exists?(path) do
      # Rotate existing backups
      rotate_backups(path, backup_count)

      # Create new backup
      backup_path = "#{path}.bak"

      case File.copy(path, backup_path) do
        {:ok, _} -> {:ok, backup_path}
        {:error, reason} -> {:error, reason}
      end
    else
      # Nothing to backup
      {:ok, nil}
    end
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp rotate_backups(path, count) when count > 0 do
    # Delete the oldest backup if it exists
    oldest = "#{path}.bak.#{count - 1}"
    File.rm(oldest)

    # Rotate .bak.N -> .bak.N+1 (from highest to lowest)
    Enum.each((count - 2)..0//-1, fn n ->
      src = if n == 0, do: "#{path}.bak", else: "#{path}.bak.#{n}"
      dst = "#{path}.bak.#{n + 1}"
      if File.exists?(src), do: File.rename(src, dst)
    end)
  end

  defp rotate_backups(_path, _count), do: :ok

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"

  defp format_bytes(bytes) when bytes < 1024 * 1024 * 1024,
    do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"

  defp format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"

  defp format_datetime({{y, m, d}, {h, min, s}}) do
    "#{y}-#{pad(m)}-#{pad(d)} #{pad(h)}:#{pad(min)}:#{pad(s)}"
  end

  defp pad(n) when n < 10, do: "0#{n}"
  defp pad(n), do: "#{n}"
end
