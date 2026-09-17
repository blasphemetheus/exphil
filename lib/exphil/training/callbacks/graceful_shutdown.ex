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
      signal_id: nil,
      interrupted: false
    }
  end

  @impl true
  def on_train_begin(state, cb) do
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]

    if checkpoint_path do
      # Agent stores ONLY metadata — never GPU tensors
      {:ok, agent} =
        Agent.start_link(
          fn ->
            %{
              epoch: 0,
              step: 0,
              checkpoint_path: checkpoint_path,
              interrupted: false,
              signal_waiter: nil
            }
          end,
          name: :trainer_state
        )

      # Trap SIGTERM — set interrupted flag instead of saving directly
      owner = self()

      signal_id =
        case System.trap_signal(:sigterm, fn ->
               waiter = self()
               monitor = Process.monitor(owner)
               Agent.update(agent, &%{&1 | interrupted: true, signal_waiter: waiter})
               Output.puts("\n  Interrupt received — will save checkpoint after current batch")
               # Erlang's default SIGTERM handler runs after this callback.
               # Hold it until checkpoint/export cleanup finishes in the owner.
               receive do
                 {:training_shutdown_complete, ^owner} -> :ok
                 {:DOWN, ^monitor, :process, ^owner, _} -> :ok
               end

               Process.demonitor(monitor, [:flush])
               :ok
             end) do
          {:ok, id} -> id
          {:error, :not_sup} -> nil
        end

      {:cont, state,
       %{cb | agent_pid: agent, checkpoint_path: checkpoint_path, signal_id: signal_id}}
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

  def finish_shutdown do
    if waiter = Process.delete({__MODULE__, :signal_waiter}),
      do: send(waiter, {:training_shutdown_complete, self()})

    :ok
  end

  @impl true
  def on_train_end(state, cb) do
    waiter = if cb.agent_pid, do: Agent.get(cb.agent_pid, & &1.signal_waiter)

    if waiter do
      Process.put({__MODULE__, :signal_waiter}, waiter)
    else
      if cb.signal_id, do: System.untrap_signal(:sigterm, cb.signal_id)
    end

    if cb.agent_pid do
      try do
        Agent.stop(:trainer_state)
      rescue
        _ -> :ok
      end
    end

    {:cont, state, %{cb | agent_pid: nil, signal_id: nil}}
  end
end
