defmodule ExPhil.Training.Recovery do
  @moduledoc """
  Training recovery utilities for detecting and resuming incomplete training runs.

  Creates and manages `.incomplete` marker files that track training progress.
  If training crashes or is interrupted, the marker allows automatic detection
  and resumption.
  """

  alias ExPhil.Error.RecoveryError

  @marker_suffix ".incomplete"

  @doc """
  Get the incomplete marker path for a checkpoint.

  ## Examples

      iex> Recovery.marker_path("checkpoints/model.axon")
      "checkpoints/model.axon.incomplete"
  """
  @spec marker_path(String.t()) :: String.t()
  def marker_path(checkpoint_path) do
    checkpoint_path <> @marker_suffix
  end

  @doc """
  Check if there's an incomplete training run for this checkpoint.

  Returns `{:incomplete, state}` if found, `:ok` if no incomplete run.
  """
  @spec check_incomplete(String.t()) :: {:incomplete, map()} | :ok
  def check_incomplete(checkpoint_path) do
    marker = marker_path(checkpoint_path)

    if File.exists?(marker) do
      case File.read(marker) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, state} -> {:incomplete, state}
            _ -> :ok
          end

        _ ->
          :ok
      end
    else
      :ok
    end
  end

  @doc """
  Create an incomplete marker at training start.

  Records initial training configuration for recovery.
  """
  @spec mark_started(String.t(), keyword()) :: :ok | {:error, term()}
  def mark_started(checkpoint_path, opts) do
    marker = marker_path(checkpoint_path)

    state = %{
      started_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      checkpoint: checkpoint_path,
      epochs_target: opts[:epochs],
      epochs_completed: 0,
      preset: opts[:preset],
      last_epoch_loss: nil,
      last_update: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    write_marker(marker, state)
  end

  @doc """
  Update the marker after each epoch completes.
  """
  @spec mark_epoch_complete(String.t(), non_neg_integer(), float()) :: :ok | {:error, term()}
  def mark_epoch_complete(checkpoint_path, epoch, loss) do
    marker = marker_path(checkpoint_path)

    case File.read(marker) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, state} ->
            updated =
              Map.merge(state, %{
                "epochs_completed" => epoch,
                "last_epoch_loss" => loss,
                "last_update" => DateTime.utc_now() |> DateTime.to_iso8601()
              })

            write_marker(marker, updated)

          _ ->
            {:error, RecoveryError.new(:invalid_marker)}
        end

      error ->
        error
    end
  end

  @doc """
  Remove the incomplete marker when training completes successfully.
  """
  @spec mark_complete(String.t()) :: :ok
  def mark_complete(checkpoint_path) do
    marker = marker_path(checkpoint_path)

    case File.rm(marker) do
      :ok -> :ok
      # Already removed, that's fine
      {:error, :enoent} -> :ok
      error -> error
    end
  end

  @doc """
  Format incomplete state for display.
  """
  @spec format_incomplete_info(map()) :: String.t()
  def format_incomplete_info(state) do
    epochs_done = state["epochs_completed"] || 0
    epochs_target = state["epochs_target"] || "?"
    started = state["started_at"] || "unknown"
    last_loss = state["last_epoch_loss"]
    preset = state["preset"]

    loss_str = if last_loss, do: " (loss: #{Float.round(last_loss, 4)})", else: ""
    preset_str = if preset, do: " [preset: #{preset}]", else: ""

    "Incomplete training detected#{preset_str}\n" <>
      "  Started: #{started}\n" <>
      "  Progress: #{epochs_done}/#{epochs_target} epochs#{loss_str}"
  end

  @doc """
  Get the checkpoint path to resume from.

  Returns the best checkpoint if it exists, otherwise the base checkpoint.
  """
  @spec get_resume_checkpoint(String.t()) :: String.t() | nil
  def get_resume_checkpoint(checkpoint_path) do
    # Try best checkpoint first
    best_path = String.replace(checkpoint_path, ".axon", "_best.axon")

    cond do
      File.exists?(best_path) -> best_path
      File.exists?(checkpoint_path) -> checkpoint_path
      true -> nil
    end
  end

  defp write_marker(path, state) do
    case File.write(path, Jason.encode!(state, pretty: true)) do
      :ok -> :ok
      error -> error
    end
  end
end
