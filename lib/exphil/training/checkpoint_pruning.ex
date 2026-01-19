defmodule ExPhil.Training.CheckpointPruning do
  @moduledoc """
  Manages checkpoint pruning to keep only the best N models.

  During training, multiple checkpoints may be saved (per-epoch, best model, etc.).
  This module tracks checkpoint quality and prunes older/worse checkpoints to save disk space.

  ## Usage

      # Create a pruning tracker
      pruner = CheckpointPruning.new(keep_best: 3)

      # Track a checkpoint with its loss
      pruner = CheckpointPruning.track(pruner, "checkpoints/epoch_5.axon", 1.234)

      # Prune checkpoints (keeps best 3, deletes rest)
      {pruner, deleted} = CheckpointPruning.prune(pruner)

  ## Configuration

    * `:keep_best` - Number of best checkpoints to keep (default: 5)
    * `:metric` - :loss (lower is better) or :accuracy (higher is better)

  """

  defstruct [
    :keep_best,
    :metric,
    checkpoints: []
  ]

  @type checkpoint :: %{
          path: String.t(),
          value: float(),
          epoch: non_neg_integer() | nil,
          timestamp: DateTime.t()
        }

  @type t :: %__MODULE__{
          keep_best: pos_integer(),
          metric: :loss | :accuracy,
          checkpoints: [checkpoint()]
        }

  @doc """
  Create a new checkpoint pruner.

  ## Options

    * `:keep_best` - Number of best checkpoints to retain (default: 5)
    * `:metric` - Comparison metric: :loss (minimize) or :accuracy (maximize)

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      keep_best: Keyword.get(opts, :keep_best, 5),
      metric: Keyword.get(opts, :metric, :loss),
      checkpoints: []
    }
  end

  @doc """
  Track a checkpoint with its metric value.

  Returns updated pruner with the checkpoint added to tracking list.
  """
  @spec track(t(), String.t(), float(), keyword()) :: t()
  def track(%__MODULE__{} = pruner, path, value, opts \\ []) do
    checkpoint = %{
      path: path,
      value: value,
      epoch: Keyword.get(opts, :epoch),
      timestamp: DateTime.utc_now()
    }

    %{pruner | checkpoints: [checkpoint | pruner.checkpoints]}
  end

  @doc """
  Prune checkpoints, keeping only the best N.

  Returns `{updated_pruner, deleted_paths}` where `deleted_paths` is a list
  of checkpoint paths that were removed from disk.

  ## Options

    * `:dry_run` - If true, don't actually delete files (default: false)

  """
  @spec prune(t(), keyword()) :: {t(), [String.t()]}
  def prune(%__MODULE__{} = pruner, opts \\ []) do
    dry_run = Keyword.get(opts, :dry_run, false)

    # Sort checkpoints by value (best first based on metric)
    sorted = sort_checkpoints(pruner.checkpoints, pruner.metric)

    # Split into keepers and to-delete
    {keep, delete} = Enum.split(sorted, pruner.keep_best)

    # Delete the files (unless dry run)
    deleted_paths =
      if dry_run do
        Enum.map(delete, & &1.path)
      else
        Enum.flat_map(delete, fn checkpoint ->
          case delete_checkpoint_files(checkpoint.path) do
            :ok -> [checkpoint.path]
            {:error, _} -> []
          end
        end)
      end

    {%{pruner | checkpoints: keep}, deleted_paths}
  end

  @doc """
  Get the current list of tracked checkpoints, sorted by quality.
  """
  @spec list(t()) :: [checkpoint()]
  def list(%__MODULE__{} = pruner) do
    sort_checkpoints(pruner.checkpoints, pruner.metric)
  end

  @doc """
  Get the best checkpoint currently tracked.
  """
  @spec best(t()) :: checkpoint() | nil
  def best(%__MODULE__{checkpoints: []}) do
    nil
  end

  def best(%__MODULE__{} = pruner) do
    pruner
    |> list()
    |> List.first()
  end

  @doc """
  Get the number of checkpoints being tracked.
  """
  @spec count(t()) :: non_neg_integer()
  def count(%__MODULE__{checkpoints: checkpoints}) do
    length(checkpoints)
  end

  @doc """
  Check if pruning is needed (more checkpoints than keep_best).
  """
  @spec needs_pruning?(t()) :: boolean()
  def needs_pruning?(%__MODULE__{} = pruner) do
    length(pruner.checkpoints) > pruner.keep_best
  end

  @doc """
  Remove a specific checkpoint from tracking (useful if manually deleted).
  """
  @spec untrack(t(), String.t()) :: t()
  def untrack(%__MODULE__{} = pruner, path) do
    checkpoints = Enum.reject(pruner.checkpoints, &(&1.path == path))
    %{pruner | checkpoints: checkpoints}
  end

  @doc """
  Prune checkpoints from a directory matching a pattern.

  This is a convenience function for pruning checkpoints outside of the
  tracking system (e.g., cleaning up old training runs).

  ## Options

    * `:pattern` - Glob pattern for checkpoint files (default: "*.axon")
    * `:keep_best` - Number to keep (default: 5)
    * `:dry_run` - If true, don't delete (default: false)

  """
  @spec prune_directory(String.t(), keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def prune_directory(dir, opts \\ []) do
    pattern = Keyword.get(opts, :pattern, "*.axon")
    keep = Keyword.get(opts, :keep_best, 5)
    dry_run = Keyword.get(opts, :dry_run, false)

    full_pattern = Path.join(dir, pattern)

    case Path.wildcard(full_pattern) do
      [] ->
        {:ok, []}

      files ->
        # Sort by modification time (newest first)
        # Note: mtime is a tuple {{y,m,d},{h,m,s}} which compares correctly
        sorted =
          files
          |> Enum.map(fn path ->
            stat = File.stat!(path)
            {path, stat.mtime}
          end)
          |> Enum.sort_by(fn {_, mtime} -> mtime end, :desc)
          |> Enum.map(fn {path, _} -> path end)

        # Keep newest N, delete rest
        {_keep, delete} = Enum.split(sorted, keep)

        deleted =
          if dry_run do
            delete
          else
            Enum.flat_map(delete, fn path ->
              case delete_checkpoint_files(path) do
                :ok -> [path]
                {:error, _} -> []
              end
            end)
          end

        {:ok, deleted}
    end
  end

  # Private functions

  defp sort_checkpoints(checkpoints, :loss) do
    # Lower is better
    Enum.sort_by(checkpoints, & &1.value)
  end

  defp sort_checkpoints(checkpoints, :accuracy) do
    # Higher is better
    Enum.sort_by(checkpoints, & &1.value, :desc)
  end

  defp delete_checkpoint_files(path) do
    # Delete the main checkpoint and related files
    related_files = [
      path,
      String.replace(path, ".axon", "_policy.bin"),
      String.replace(path, ".axon", "_config.json")
    ]

    Enum.each(related_files, fn file ->
      if File.exists?(file) do
        File.rm(file)
      end
    end)

    :ok
  rescue
    e -> {:error, e}
  end
end
