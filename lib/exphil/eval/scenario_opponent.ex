defmodule ExPhil.Eval.ScenarioOpponent do
  @moduledoc "Response-only opponent input selection. Neutral releases controls; it cannot cancel an attack already in progress."

  def input("neutral", _recorded, neutral), do: neutral
  def input("replay", nil, neutral), do: neutral
  def input("replay", {_subject, opponent}, _neutral), do: opponent

  def input(mode, _recorded, _neutral),
    do: raise(ArgumentError, "invalid response opponent: #{inspect(mode)}")

  def recorded_neutral?(controller) do
    buttons = [
      :button_a,
      :button_b,
      :button_x,
      :button_y,
      :button_z,
      :button_l,
      :button_r,
      :button_start,
      :button_d_up,
      :button_d_down,
      :button_d_left,
      :button_d_right
    ]

    Enum.all?(buttons, &(Map.fetch!(controller, &1) == false)) and
      Enum.all?(
        [:main_stick_x, :main_stick_y, :c_stick_x, :c_stick_y],
        &(abs(Map.fetch!(controller, &1) - 0.5) <= 0.00001)
      ) and
      controller.l_trigger <= 0.00001 and controller.r_trigger <= 0.00001
  end
end
