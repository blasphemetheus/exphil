defmodule ExPhil.Training.Metrics do
  @moduledoc """
  Training metrics beyond simple loss - per-action accuracy, confusion analysis, etc.

  Key insight: Overall loss can be low while rare but important actions (Z, L-cancel triggers)
  are poorly predicted. Per-action metrics reveal this.
  """

  import Nx.Defn

  @button_names [:a, :b, :x, :y, :z, :l, :r, :start]
  @stick_names [:main_x, :main_y, :c_x, :c_y]

  @doc """
  Compute per-action accuracy metrics from predictions and targets.

  Returns a map with:
  - `:button_accuracy` - Per-button accuracy (8 buttons)
  - `:stick_mse` - Per-stick mean squared error (4 sticks)
  - `:overall_button_accuracy` - Weighted average button accuracy
  - `:rare_action_accuracy` - Accuracy on rare actions (Z, L, R)

  ## Parameters

  - `predictions` - Map with button logits and stick values
  - `targets` - Map with target button values and stick positions
  """
  @spec compute(map(), map()) :: map()
  def compute(predictions, targets) do
    button_acc = compute_button_accuracy(predictions, targets)
    stick_mse = compute_stick_mse(predictions, targets)
    rare_acc = compute_rare_action_accuracy(predictions, targets)

    %{
      button_accuracy: button_acc,
      stick_mse: stick_mse,
      overall_button_accuracy: mean_accuracy(button_acc),
      rare_action_accuracy: rare_acc,
      overall_stick_mse: mean_mse(stick_mse)
    }
  end

  @doc """
  Compute accuracy for each button head.
  """
  def compute_button_accuracy(predictions, targets) do
    @button_names
    |> Enum.with_index()
    |> Enum.map(fn {name, idx} ->
      pred_key = :"button_#{idx}"
      target_key = :"buttons"

      if Map.has_key?(predictions, pred_key) and Map.has_key?(targets, target_key) do
        pred = predictions[pred_key]
        target = Nx.slice_along_axis(targets.buttons, idx, 1, axis: -1) |> Nx.squeeze(axes: [-1])

        # Convert logits to predictions (> 0.5 after sigmoid, or > 0 for logits)
        pred_binary = Nx.greater(pred, 0)
        target_binary = Nx.greater(target, 0.5)

        correct = Nx.equal(pred_binary, target_binary)
        accuracy = Nx.mean(correct) |> Nx.to_number()

        {name, accuracy}
      else
        {name, nil}
      end
    end)
    |> Enum.reject(fn {_, v} -> is_nil(v) end)
    |> Map.new()
  end

  @doc """
  Compute MSE for each stick axis.
  """
  def compute_stick_mse(predictions, targets) do
    stick_pairs = [
      {:main_x, :main_stick_x},
      {:main_y, :main_stick_y},
      {:c_x, :c_stick_x},
      {:c_y, :c_stick_y}
    ]

    stick_pairs
    |> Enum.map(fn {name, target_key} ->
      if Map.has_key?(predictions, name) and Map.has_key?(targets, target_key) do
        pred = predictions[name]
        target = targets[target_key]

        mse = Nx.mean(Nx.pow(Nx.subtract(pred, target), 2)) |> Nx.to_number()
        {name, mse}
      else
        {name, nil}
      end
    end)
    |> Enum.reject(fn {_, v} -> is_nil(v) end)
    |> Map.new()
  end

  @doc """
  Compute accuracy specifically on rare actions (Z, L, R triggers).

  These are important for tech skill (L-cancels, grabs, shields) but rare in data.
  """
  def compute_rare_action_accuracy(predictions, targets) do
    rare_buttons = [:z, :l, :r]
    rare_indices = [4, 5, 6]  # Z=4, L=5, R=6 in button order

    if Map.has_key?(targets, :buttons) do
      accuracies = Enum.zip(rare_buttons, rare_indices)
      |> Enum.map(fn {name, idx} ->
        pred_key = :"button_#{idx}"

        if Map.has_key?(predictions, pred_key) do
          pred = predictions[pred_key]
          target = Nx.slice_along_axis(targets.buttons, idx, 1, axis: -1) |> Nx.squeeze(axes: [-1])

          # Only compute accuracy where target is pressed (positive examples)
          target_pressed = Nx.greater(target, 0.5)
          num_pressed = Nx.sum(target_pressed) |> Nx.to_number()

          if num_pressed > 0 do
            pred_binary = Nx.greater(pred, 0)
            correct_when_pressed = Nx.logical_and(Nx.equal(pred_binary, target_pressed), target_pressed)
            recall = Nx.sum(correct_when_pressed) |> Nx.to_number() |> Kernel./(num_pressed)
            {name, recall}
          else
            {name, nil}
          end
        else
          {name, nil}
        end
      end)
      |> Enum.reject(fn {_, v} -> is_nil(v) end)
      |> Map.new()

      accuracies
    else
      %{}
    end
  end

  @doc """
  Format metrics for display.
  """
  @spec format(map()) :: String.t()
  def format(metrics) do
    lines = []

    # Button accuracy
    if Map.has_key?(metrics, :button_accuracy) and map_size(metrics.button_accuracy) > 0 do
      button_str = metrics.button_accuracy
      |> Enum.map(fn {k, v} -> "#{k}=#{Float.round(v * 100, 1)}%" end)
      |> Enum.join(" ")
      lines = ["Buttons: #{button_str}" | lines]
    end

    # Stick MSE
    if Map.has_key?(metrics, :stick_mse) and map_size(metrics.stick_mse) > 0 do
      stick_str = metrics.stick_mse
      |> Enum.map(fn {k, v} -> "#{k}=#{Float.round(v, 4)}" end)
      |> Enum.join(" ")
      lines = ["Sticks: #{stick_str}" | lines]
    end

    # Rare action accuracy
    if Map.has_key?(metrics, :rare_action_accuracy) and map_size(metrics.rare_action_accuracy) > 0 do
      rare_str = metrics.rare_action_accuracy
      |> Enum.map(fn {k, v} -> "#{k}=#{Float.round(v * 100, 1)}%" end)
      |> Enum.join(" ")
      lines = ["Rare (recall): #{rare_str}" | lines]
    end

    # Summary
    summary_parts = []
    if Map.has_key?(metrics, :overall_button_accuracy) do
      summary_parts = ["btn_acc=#{Float.round(metrics.overall_button_accuracy * 100, 1)}%" | summary_parts]
    end
    if Map.has_key?(metrics, :overall_stick_mse) do
      summary_parts = ["stick_mse=#{Float.round(metrics.overall_stick_mse, 4)}" | summary_parts]
    end
    if length(summary_parts) > 0 do
      lines = ["Summary: #{Enum.join(summary_parts, " ")}" | lines]
    end

    Enum.reverse(lines) |> Enum.join("\n")
  end

  @doc """
  Compute action distribution statistics from a dataset.

  Useful for understanding class imbalance.
  """
  @spec action_distribution([map()]) :: map()
  def action_distribution(frames) when is_list(frames) do
    # Count button presses
    button_counts = Enum.reduce(frames, %{}, fn frame, acc ->
      buttons = frame.controller || %{}
      @button_names
      |> Enum.reduce(acc, fn btn, a ->
        pressed = Map.get(buttons, btn, false)
        if pressed do
          Map.update(a, btn, 1, &(&1 + 1))
        else
          a
        end
      end)
    end)

    total = length(frames)

    %{
      total_frames: total,
      button_press_rates: Enum.map(button_counts, fn {k, v} -> {k, v / total} end) |> Map.new(),
      button_counts: button_counts
    }
  end

  defp mean_accuracy(button_acc) when map_size(button_acc) == 0, do: 0.0
  defp mean_accuracy(button_acc) do
    values = Map.values(button_acc)
    Enum.sum(values) / length(values)
  end

  defp mean_mse(stick_mse) when map_size(stick_mse) == 0, do: 0.0
  defp mean_mse(stick_mse) do
    values = Map.values(stick_mse)
    Enum.sum(values) / length(values)
  end
end
