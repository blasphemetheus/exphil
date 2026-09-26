defmodule ExPhil.Training.Callbacks.RollingCheckpoint do
  @moduledoc "Early, bounded recovery checkpoints independent of epoch selection."
  use ExPhil.Training.Callback
  alias ExPhil.Training.{Imitation, Output}

  @impl true
  def init(opts) do
    every = Keyword.get(opts, :every, 500)
    unless is_integer(every) and every > 0, do: raise(ArgumentError, "every must be positive")
    path = Keyword.fetch!(opts, :checkpoint_path)
    unless String.ends_with?(path, ".axon"),
      do: raise(ArgumentError, "recovery checkpoint path must end in .axon")
    %{path: path, every: every}
  end

  @impl true
  def on_train_begin(state, cb) do
    save!(state, cb, "initial")
    {:cont, state, cb}
  end

  @impl true
  def on_batch_end(state, cb) do
    if state.step == 1 or rem(state.step, cb.every) == 0 do
      save!(state, cb, Integer.to_string(rem(div(state.step, cb.every), 2)))
    end

    {:cont, state, cb}
  end

  defp save!(state, cb, slot) do
    path = String.replace_suffix(cb.path, ".axon", "_resume_#{slot}.axon")

    meta = %{
      epoch: state.epoch,
      batch_idx: state.batch_idx,
      step: state.step,
      seed: state.opts[:seed],
      recovery_kind: :weights_optimizer,
      data_cursor_restored: false
    }

    case Imitation.save_checkpoint(state.trainer, path, meta: meta) do
      :ok -> Output.puts("Recovery checkpoint step #{state.step}: #{path}")
      {:error, reason} -> raise "Recovery checkpoint failed: #{inspect(reason)}"
    end
  end
end
