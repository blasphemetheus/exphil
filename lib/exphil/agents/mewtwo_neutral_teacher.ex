defmodule ExPhil.Agents.MewtwoNeutralTeacher do
  @moduledoc """
  Experimental, stateful Mewtwo/Fox FD neutral teacher. It commits to a Tech
  routine until completion, but discards that routine on damage, capture or
  respawn. This is a live teacher, not a stateless replay relabeler. Approval
  must come from the separate live scenario battery before collecting labels.
  """
  alias Melee.{Enums, FrameData, Tech}
  defstruct tech: nil, choice: :wait

  def new, do: %__MODULE__{}

  def step(teacher, me, fox) do
    cond do
      interrupted?(me) ->
        {%__MODULE__{choice: :interrupted}, [:release_all]}

      teacher.tech != nil ->
        {status, tech, commands} = Tech.step(teacher.tech, me)
        {%{teacher | tech: if(status == :done, do: nil, else: tech)}, commands}

      not me.on_ground and match?({:short_hop, _}, teacher.choice) ->
        {:short_hop, direction} = teacher.choice

        x =
          if abs(me.position.x) < 65, do: if(direction == :right, do: 0.65, else: 0.35), else: 0.5

        {teacher, [:release_all, {:tilt, :main, x, 0.5}]}

      true ->
        execute(teacher, choose(me, fox), me)
    end
  end

  @doc "Decision only; public so negative contexts can be tested independently."
  def choose(me, fox) do
    dx = fox.position.x - me.position.x
    gap = abs(dx)
    toward = if dx >= 0, do: :right, else: :left
    away = if dx >= 0, do: :left, else: :right
    facing = me.facing == dx >= 0
    grounded_fox = fox.on_ground and abs(fox.position.y - me.position.y) < 4

    cond do
      interrupted?(me) ->
        :wait

      not me.on_ground or me.action not in [14, 15, 16, 17, 18, 20, 21, 22, 39, 40, 41] ->
        :wait

      interrupted?(fox) ->
        :wait

      abs(me.position.x) > 65 ->
        {:walk, if(me.position.x > 0, do: :left, else: :right)}

      dangerous?(fox) and fox.action in [44, 45, 46] and gap >= 24 and gap <= 40 and facing ->
        {:aerial, :nair, toward}

      dangerous?(fox) and gap < 45 ->
        if gap >= 24 and abs(me.position.x + sign(away) * 40) < 70,
          do: {:wavedash, away},
          else: :shield

      fox.action in [44, 45, 46] and gap > 17 and gap < 24 ->
        {:walk, away}

      not facing ->
        {:walk, toward}

      grounded_fox and fox.action in 178..182 and gap <= 45 ->
        if gap <= 12, do: :grab, else: {:run, toward}

      not grounded_fox ->
        if gap <= 10 and fox.position.y >= 3.0 and fox.position.y <= 12.0,
          do: {:aerial, :fair, toward},
          else: {:walk, toward}

      gap > 55 and abs(me.position.x + sign(toward) * 40) < 65 ->
        {:wavedash, toward}

      gap > 21 ->
        {:walk, toward}

      me.action in [20, 21, 22] ->
        :brake

      gap > 11 ->
        :down_tilt

      gap > 7 ->
        {:aerial, :fair, toward}

      true ->
        {:aerial, :nair, toward}
    end
  end

  defp interrupted?(p),
    do:
      p.action < 14 or p.action in 75..91 or
        p.action in 183..205 or p.action in 223..232 or p.action in 239..243

  # No generic 'safe after frame 12' assumption. Unknown attack states are
  # treated conservatively; this first teacher only targets Fox.
  defp dangerous?(fox) do
    action = Enums.Action.from_id(fox.action)

    fox.action in 44..69 and
      FrameData.attack_state(:fox, action, fox.action_frame) != :cooldown
  end

  defp execute(t, {:aerial, aerial, direction} = choice, me),
    do: start(t, choice, Tech.new(:shffl, :mewtwo, aerial: aerial, drift: direction), me)

  defp execute(t, {routine, direction} = choice, me) when routine in [:wavedash, :short_hop],
    do: start(t, choice, Tech.new(routine, :mewtwo, direction: direction), me)

  defp execute(t, choice, _me) do
    commands =
      case choice do
        {:walk, dir} -> [{:tilt, :main, if(dir == :right, do: 0.65, else: 0.35), 0.5}]
        {:run, dir} -> [{:tilt, :main, if(dir == :right, do: 1.0, else: 0.0), 0.5}]
        :grab -> [{:press, :z}]
        :shield -> [{:press, :r}]
        :down_tilt -> [{:tilt, :main, 0.5, 0.2}, {:press, :a}]
        _ -> []
      end

    {%{t | choice: choice}, [:release_all | commands]}
  end

  defp start(t, choice, tech, me) do
    {_, tech, commands} = Tech.step(tech, me)
    {%{t | choice: choice, tech: tech}, [:release_all | commands]}
  end

  defp sign(:right), do: 1
  defp sign(:left), do: -1
end
