defmodule ExPhil.Training.Callbacks.LossPlot do
  @moduledoc """
  Display ASCII loss graph and save HTML loss plot on training completion.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Output, Plots}

  @impl true
  def init(opts) do
    %{
      checkpoint_path: Keyword.get(opts, :checkpoint_path)
    }
  end

  @impl true
  def on_train_end(state, cb) do
    history = state.history

    if length(history) > 1 do
      # ASCII terminal graph
      Output.puts_raw("")
      Output.terminal_loss_graph(history, title: "Training Loss", width: 60)
    end

    # HTML plot
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]
    if checkpoint_path && length(history) > 0 do
      plot_path = String.replace(checkpoint_path, ".axon", "_loss.html")
      try do
        Plots.save_report!(history, plot_path,
          title: "Training Report",
          metadata: [
            epochs: state.epoch,
            batch_size: state.opts[:batch_size],
            temporal: state.opts[:temporal],
            backbone: state.opts[:backbone]
          ]
        )
        Output.puts("  Loss plot saved to #{plot_path}")
      rescue
        _ -> Output.puts("  Loss plot generation failed")
      end
    end

    {:cont, state, cb}
  end
end
