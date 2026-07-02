defmodule ExPhil.Training.Callbacks.GracefulShutdown do
  @moduledoc """
  Save checkpoint on Ctrl+C / SIGTERM for crash recovery.

  IMPORTANT: Never sends GPU tensors (EXLA NIF refs) across process boundaries.
  The Agent stores only metadata (epoch, step, checkpoint path). On interrupt,
  it sets a flag that the training loop checks and saves from the main process.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.Output

  @impl true
  def init(opts) do
    %{
      checkpoint_path: Keyword.get(opts, :checkpoint_path),
      agent_pid: nil,
      interrupted: false
    }
  end

  @impl true
  def on_train_begin(state, cb) do
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]

    if checkpoint_path do
      # Agent stores ONLY metadata — never GPU tensors
      {:ok, agent} = Agent.start_link(fn ->
        %{epoch: 0, step: 0, checkpoint_path: checkpoint_path, interrupted: false}
      end, name: :trainer_state)

      # Trap SIGTERM — set interrupted flag instead of saving directly
      for signal <- [:sigterm] do
        try do
          System.trap_signal(signal, fn ->
            Agent.update(:trainer_state, fn state ->
              %{state | interrupted: true}
            end)
            Output.puts("\n  Interrupt received — will save checkpoint after current batch")
          end)
        rescue
          _ -> :ok
        end
      end

      {:cont, state, %{cb | agent_pid: agent, checkpoint_path: checkpoint_path}}
    else
      {:cont, state, cb}
    end
  end

  @impl true
  def on_batch_end(state, cb) do
    if cb.agent_pid do
      # Update metadata only (no GPU tensors) — cheap, every 1000 steps
      if state.step > 0 && rem(state.step, 1000) == 0 do
        Agent.update(:trainer_state, fn meta ->
          %{meta | epoch: state.epoch, step: state.step}
        end)
      end

      # Check if interrupt was requested — save from MAIN process (has GPU access)
      interrupted = Agent.get(:trainer_state, & &1.interrupted)

      if interrupted do
        Output.puts("  Saving interrupt checkpoint from main process...")
        interrupt_path = String.replace(cb.checkpoint_path, ".axon", "_interrupt.axon")

        case ExPhil.Training.Imitation.save_checkpoint(state.trainer, interrupt_path) do
          :ok -> Output.puts("  Saved to #{interrupt_path}")
          {:error, reason} -> Output.puts("  Save failed: #{inspect(reason)}")
        end

        {:halt, state, %{cb | interrupted: true}}
      else
        {:cont, state, cb}
      end
    else
      {:cont, state, cb}
    end
  end

  @impl true
  def on_train_end(state, cb) do
    if cb.agent_pid do
      try do
        Agent.stop(:trainer_state)
      rescue
        _ -> :ok
      end
    end

    {:cont, state, %{cb | agent_pid: nil}}
  end
end
