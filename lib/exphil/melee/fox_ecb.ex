defmodule ExPhil.Melee.FoxEcb do
  @moduledoc """
  Fox's environment collision box (ECB) frame by frame through the airborne
  recovery states, recorded from the real game code in melee-sim-light
  (`reports/triage/fox_ecb_recorder.c`, 2026-09-23) into
  `priv/checkmate/fox_ecb.csv`. The ledge-snap box is Fox's constant
  {x 11, y 13, height 9}.

  `lookup(kind, t)` returns `{bottom_y, right_x, top_y}` (offsets from the
  fighter's position; left is the mirror of right) for frame `t` of a phase,
  holding the last recorded frame after the series ends. Kinds: `:fall`,
  `:jump`, `:charge`, `:travel_up | :travel_side | :travel_diag`,
  `:end_up | :end_side | :end_diag`, `:helpless`, `:side_b`.
  """

  @path Path.expand("../../../priv/checkmate/fox_ecb.csv", __DIR__)
  @external_resource @path

  @snap {11.0, 13.0, 9.0}

  rows =
    @path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> tl()
    |> Enum.map(fn line ->
      [sc, t, motion | rest] = String.split(line, ",")
      [_af, _x, _y, _facing, _top_x, top_y, _bx, bottom_y, right_x | _] = rest
      %{scenario: sc, t: String.to_integer(t), motion: String.to_integer(motion),
        bottom_y: String.to_float(bottom_y), right_x: String.to_float(right_x), top_y: String.to_float(top_y)}
    end)

  series = fn scenario, pred ->
    rows
    |> Enum.filter(&(&1.scenario == scenario and pred.(&1)))
    |> Enum.sort_by(& &1.t)
    |> Enum.map(&{&1.bottom_y, &1.right_x, &1.top_y})
    |> List.to_tuple()
  end

  @table %{
    fall: series.("fall", &(&1.motion == 29)),
    jump: series.("double_jump", &(&1.t >= 1)),
    charge: series.("firefox_up", &(&1.motion in [354, 355])),
    travel_up: series.("firefox_up", &(&1.motion in [356, 357])),
    travel_side: series.("firefox_side", &(&1.motion in [356, 357])),
    travel_diag: series.("firefox_diag", &(&1.motion in [356, 357])),
    end_up: series.("firefox_up", &(&1.motion == 358)),
    end_side: series.("firefox_side", &(&1.motion == 358)),
    end_diag: series.("firefox_diag", &(&1.motion == 358)),
    helpless: series.("firefox_up", &(&1.motion == 35)),
    side_b: series.("illusion", &(&1.motion in [350, 351, 352]))
  }

  @doc "Ledge-snap box `{x, y, height}`."
  def snap, do: @snap

  @doc "`{bottom_y, right_x, top_y}` for frame `t` of phase `kind`."
  def lookup(kind, t) do
    s = Map.fetch!(@table, kind)
    elem(s, max(0, min(t, tuple_size(s) - 1)))
  end

  @doc "The Fire Fox travel bucket for a stick angle in degrees."
  def travel_bucket(angle) do
    r = angle * :math.pi() / 180.0
    c = abs(:math.cos(r))
    s = abs(:math.sin(r))
    cond do
      s >= 0.9 -> :up
      c >= 0.9 -> :side
      true -> :diag
    end
  end
end
