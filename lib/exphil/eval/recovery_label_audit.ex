defmodule ExPhil.Eval.RecoveryLabelAudit do
  @moduledoc """
  Compares projected expert labels with inputs actually issued later in a
  closed-loop teacher run. This measures label fidelity, not policy causality.
  Only contiguous, fully observed horizons are scored. Analog tolerance is 0.02.
  """

  alias ExPhil.Agents.MultishineExpert

  def sample(expert, player, previous, issued, frame, shifts \\ [0, 3, 4, 5]) do
    %{
      frame: frame,
      state: [trunc(player.action), trunc(player.action_frame), player.on_ground],
      on_loop: MultishineExpert.on_loop?(expert, player),
      issued: command(issued),
      predictions:
        Map.new(shifts, fn shift ->
          prediction =
            case MultishineExpert.label_ahead(expert, player, shift, previous) do
              {:ok, controller} -> command(controller)
              :skip -> nil
            end

          {shift, prediction}
        end)
    }
  end

  def report(samples) do
    by_frame = Map.new(samples, &{&1.frame, &1})
    if map_size(by_frame) != length(samples), do: raise(ArgumentError, "duplicate trace frames")

    comparisons =
      for sample <- samples,
          {shift, predicted} <- sample.predictions,
          predicted != nil,
          Enum.all?(0..shift, &Map.has_key?(by_frame, sample.frame + &1)) do
        future = Map.fetch!(by_frame, sample.frame + shift)

        %{
          frame: sample.frame,
          state: sample.state,
          on_loop: sample.on_loop,
          shift: shift,
          predicted: predicted,
          actual: future.issued,
          future_state: future.state,
          future_on_loop: future.on_loop,
          matches: matches?(predicted, future.issued)
        }
      end

    summary =
      comparisons
      |> Enum.group_by(&{&1.on_loop, &1.shift})
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn {{on_loop, shift}, rows} ->
        %{
          on_loop: on_loop,
          shift: shift,
          count: length(rows),
          mismatches: Enum.count(rows, &(not &1.matches))
        }
      end)

    %{summary: summary, comparisons: comparisons, samples: length(samples)}
  end

  defp command(controller) do
    %{
      buttons:
        Enum.map(
          [
            :button_a,
            :button_b,
            :button_x,
            :button_y,
            :button_z,
            :button_l,
            :button_r,
            :button_d_up
          ],
          &Map.fetch!(controller, &1)
        ),
      analog: [
        controller.main_stick.x,
        controller.main_stick.y,
        controller.c_stick.x,
        controller.c_stick.y,
        controller.l_shoulder,
        controller.r_shoulder
      ]
    }
  end

  defp matches?(left, right) do
    left.buttons == right.buttons and
      Enum.all?(Enum.zip(left.analog, right.analog), fn {predicted, actual} ->
        abs(predicted - actual) <= 0.02
      end)
  end
end
