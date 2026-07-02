defmodule ExPhil.Training.Callbacks.EpochSummary do
  @moduledoc """
  Display epoch summary box with sparkline after each epoch.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Output, GPUUtils}

  @impl true
  def init(_opts), do: %{}

  @impl true
  def on_epoch_end(state, cb) do
    gpu_status = GPUUtils.memory_status_string()

    is_best = state.best_val_loss == nil or state.val_loss == state.best_val_loss
    highlight = if is_best, do: "* New best model", else: nil

    entries = [
      {"train_loss", safe_round(state.train_loss)},
      {"val_loss", if(state.val_loss, do: safe_round(state.val_loss), else: "n/a")},
      {"time", "#{state.epoch_time}s"},
      {"GPU", gpu_status}
    ]

    Output.puts_raw(Output.summary_box("Epoch #{state.epoch}/#{state.epochs}", entries, highlight: highlight))

    # Sparkline
    all_losses = Enum.map(state.history ++ [%{train_loss: state.train_loss}], & &1.train_loss)
    |> Enum.filter(&is_number/1)

    if length(all_losses) >= 2 do
      Output.puts_raw(Output.sparkline_with_label("  Loss", all_losses))
    end

    {:cont, state, cb}
  end

  defp safe_round(val) when is_number(val), do: Float.round(val * 1.0, 4)
  defp safe_round(val), do: inspect(val)
end
