defmodule ExPhil.Training.Callbacks.Checkpoint do
  @moduledoc """
  Save best and periodic model checkpoints.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Config, Imitation, Output}

  @impl true
  def init(opts) do
    %{
      save_best: Keyword.get(opts, :save_best, true),
      save_every: Keyword.get(opts, :save_every, nil),
      save_every_batches: Keyword.get(opts, :save_every_batches, nil),
      checkpoint_path: Keyword.get(opts, :checkpoint_path),
      overwrite: Keyword.get(opts, :overwrite, false),
      backup: Keyword.get(opts, :backup, false),
      best_val_loss: nil
    }
  end

  @impl true
  def on_train_begin(state, cb) do
    path = cb.checkpoint_path
    if path && File.exists?(path) && !cb.overwrite do
      if cb.backup do
        backup_path = path <> ".bak"
        File.cp!(path, backup_path)
        Output.puts("  Backed up existing checkpoint to #{backup_path}")
      else
        Output.warning("Checkpoint #{path} exists. Use --overwrite or --backup.")
      end
    end
    {:cont, state, cb}
  end

  @impl true
  def on_batch_end(state, cb) do
    if cb.save_every_batches != nil && cb.checkpoint_path != nil && state.step > 0 && rem(state.step, cb.save_every_batches) == 0 do
      batch_path = String.replace(cb.checkpoint_path, ".axon", "_batch#{state.step}.axon")
      case Imitation.save_checkpoint(state.trainer, batch_path) do
        :ok -> Output.puts("\n  Batch #{state.step} checkpoint saved")
        {:error, _} -> :ok
      end
    end
    {:cont, state, cb}
  end

  @impl true
  def on_epoch_end(state, cb) do
    val_loss = state.val_loss || state.train_loss
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]

    # Save best model
    cb =
      if cb.save_best and checkpoint_path != nil and is_number(val_loss) do
        is_best = cb.best_val_loss == nil or val_loss < cb.best_val_loss

        if is_best do
          best_path = String.replace(checkpoint_path, ".axon", "_best.axon")

          case Imitation.save_checkpoint(state.trainer, best_path) do
            :ok ->
              Output.puts("    * New best model saved (val_loss=#{Float.round(val_loss * 1.0, 4)})")
              # Also export a runnable policy from the SAME (best) weights, so the
              # model you ship/play is the best one — not the final-epoch weights
              # that PolicyExport captures at train end. Without this, --save-best
              # + --early-stopping silently ships your second-best model.
              best_policy_path = Config.derive_policy_path(best_path)

              case Imitation.export_policy(state.trainer, best_policy_path) do
                :ok -> Output.puts("      Best policy exported to #{best_policy_path}")
                {:error, reason} -> Output.warning("Failed to export best policy: #{inspect(reason)}")
              end

              %{cb | best_val_loss: val_loss}

            {:error, reason} ->
              Output.warning("Failed to save best model: #{inspect(reason)}")
              cb
          end
        else
          cb
        end
      else
        cb
      end

    # Update state with best val loss
    state = %{state | best_val_loss: cb.best_val_loss}

    # Periodic checkpoint
    if cb.save_every && checkpoint_path && rem(state.epoch, cb.save_every) == 0 do
      epoch_path = String.replace(checkpoint_path, ".axon", "_epoch#{state.epoch}.axon")
      case Imitation.save_checkpoint(state.trainer, epoch_path) do
        :ok -> Output.puts("    Epoch #{state.epoch} checkpoint saved")
        {:error, reason} -> Output.warning("Failed to save epoch checkpoint: #{inspect(reason)}")
      end
    end

    {:cont, state, cb}
  end

  @impl true
  def on_train_end(state, cb) do
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]

    if checkpoint_path do
      case Imitation.save_checkpoint(state.trainer, checkpoint_path) do
        :ok -> Output.puts("  Final checkpoint saved to #{checkpoint_path}")
        {:error, reason} -> Output.warning("Failed to save final checkpoint: #{inspect(reason)}")
      end
    end

    {:cont, state, cb}
  end
end
