defmodule ExPhil.StageCollision do
  @moduledoc """
  Queryable API over the extracted stage collision data
  (`priv/stage_collision/*.json`, produced by
  `scripts/extract_stage_collision.exs` — task #1; every legal stage is
  edge-validated against `Melee.Stages.edge_ground_position`).

  Consumers: `ExPhil.Situations` (`:walljump_zone` /
  `:walltech_available`), `FoxRecoveryExpert` (real ledge aim targets),
  the rewind viewer (via `Inspect.export_session`), and eventually the
  coach's scenario director.

  Accepts stage atoms or EXTERNAL (Slippi) ids — ids are converted via
  `Melee.Enums.Stage.from_external/1` (GOTCHA #96). Data is loaded once
  per stage and cached in `:persistent_term`.
  """

  @classes ~w(ground ceiling wall_left wall_right)

  @doc """
  Full collision data for a stage: `%{vertices: [[x,y],...], lines:
  [%{"v1" =>, "v2" =>, "class" =>, "ledge_grab" =>, ...}]}` or nil when
  no extracted data exists.
  """
  @spec data(atom() | number() | nil) :: map() | nil
  def data(stage) do
    case normalize(stage) do
      nil ->
        nil

      atom ->
        case :persistent_term.get({__MODULE__, atom}, :miss) do
          :miss ->
            loaded = load(atom)
            :persistent_term.put({__MODULE__, atom}, loaded)
            loaded

          hit ->
            hit
        end
    end
  end

  @doc """
  Segments of one class (`"ground"`, `"ceiling"`, `"wall_left"`,
  `"wall_right"`) as `[{x1, y1, x2, y2}]`. Empty list when no data.
  """
  @spec segments(atom() | number() | nil, String.t()) :: [{float(), float(), float(), float()}]
  def segments(stage, class) when class in @classes do
    case data(stage) do
      nil ->
        []

      %{vertices: varr, lines: lines} ->
        for l <- lines, l["class"] == class do
          [x1, y1] = elem(varr, l["v1"])
          [x2, y2] = elem(varr, l["v2"])
          {x1, y1, x2, y2}
        end
    end
  end

  @doc """
  Minimum distance from (x, y) to any wall segment (both sides), or nil
  when the stage has no extracted data. The labels' primitive.
  """
  @spec wall_distance(atom() | number() | nil, number(), number()) :: float() | nil
  def wall_distance(stage, x, y) do
    walls = segments(stage, "wall_left") ++ segments(stage, "wall_right")

    case walls do
      [] -> nil
      _ -> walls |> Enum.map(&point_segment_distance(x, y, &1)) |> Enum.min()
    end
  end

  @doc """
  Ledge-grab points: the outermost endpoint of each `ledge_grab` ground
  line, `[{x, y}]` (one per side on simple stages). Empty when no data.
  """
  @spec ledge_positions(atom() | number() | nil) :: [{float(), float()}]
  def ledge_positions(stage) do
    case data(stage) do
      nil ->
        []

      %{vertices: varr, lines: lines} ->
        for l <- lines, l["ledge_grab"], l["class"] == "ground" do
          [x1, y1] = elem(varr, l["v1"])
          [x2, y2] = elem(varr, l["v2"])
          if abs(x1) >= abs(x2), do: {x1, y1}, else: {x2, y2}
        end
    end
  end

  @doc """
  The BASE (static stage shell) line-group id for a decoded lines list:
  the group holding the most `ledge_grab` lines — every stage's static
  base has grabbable edges; moving objects (Randall, FoD platforms, PS
  transformation structures) never do. Tie-break: most lines. Found the
  hard way on PS, where the largest group (30 lines) is the fire-tree
  structure and the real base is 21 lines (4 ledges, edge 87.75).
  """
  @spec base_group([map()]) :: integer()
  def base_group(lines) do
    lines
    |> Enum.group_by(&Map.get(&1, "group", 0))
    |> Enum.max_by(fn {_g, ls} ->
      {Enum.count(ls, & &1["ledge_grab"]), length(ls)}
    end)
    |> elem(0)
  end

  @doc "Distance from point to segment {x1,y1,x2,y2}."
  @spec point_segment_distance(number(), number(), {number(), number(), number(), number()}) ::
          float()
  def point_segment_distance(px, py, {x1, y1, x2, y2}) do
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy

    t =
      if len_sq == 0.0,
        do: 0.0,
        else: max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))

    cx = x1 + t * dx
    cy = y1 + t * dy
    :math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
  end

  # ==========================================================================

  defp normalize(nil), do: nil
  defp normalize(stage) when is_atom(stage), do: stage

  defp normalize(stage) when is_number(stage) do
    case Melee.Enums.Stage.from_external(trunc(stage)) do
      :no_stage -> nil
      atom -> atom
    end
  end

  defp load(atom) do
    path = Path.join([:code.priv_dir(:exphil), "stage_collision", "#{atom}.json"])

    with {:ok, raw} <- File.read(path),
         {:ok, %{"vertices" => vs, "lines" => lines}} <- Jason.decode(raw) do
      %{vertices: List.to_tuple(vs), lines: Enum.reject(lines, &(&1["class"] == "dynamic"))}
    else
      _ -> nil
    end
  end
end
