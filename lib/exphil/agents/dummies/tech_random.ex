defmodule ExPhil.Agents.Dummies.TechRandom do
  @moduledoc """
  Reactive port-2 dummy: randomizes tech and getup options.

  The first Elixir-driven dummy (bridge `dummy_mode: "external"`, inputs via
  `MeleePort.send_controller(input, port: opponent_port)` from the
  AsyncRunner frame loop). Unlike the python-side scripted dummies (open-loop
  timing patterns), this one READS the game state — the training signal it
  creates is *reaction*: the policy can't learn one canned follow-up because
  the dummy techs in place / left / right / misses at random, and mixes its
  getups.

  Per knockdown it rolls a plan once:
    - 70% tech (uniformly in place / roll left / roll right) — R is pressed
      for 2 frames as the dummy falls near the ground in hit-fall states
      (the tech window is the ~20 frames before touchdown)
    - 30% miss -> lands in a downed state, then after a random 5-35 frame
      delay picks a getup: stand / getup attack / roll left / roll right

  Stateless between knockdowns: any non-knockdown action resets the plan.

  ## Usage (pure step function)

      state = TechRandom.new()
      {input, state} = TechRandom.step(opponent_player, state)
      MeleePort.send_controller(bridge, Map.put(input, :port, 2))
  """

  # Airborne hit states where a tech input arms: damage-fly family + tumble
  @damage_air 84..91
  @tumble 38

  # Downed (missed-tech) states, face-up and face-down families
  @downed 183..197
  # The "waiting on the ground, actionable" states where a getup input works
  @down_wait [184, 192]

  # Height below which the tech press fires (falling in a hit state —
  # touchdown lands within the ~20-frame tech window)
  @tech_height 12.0

  defstruct plan: nil, delay: 0, press_frames: 0

  @type t :: %__MODULE__{}

  @spec new() :: t()
  def new, do: %__MODULE__{}

  @doc """
  Compute this frame's input for the dummy. Returns `{input_map, new_state}`;
  the input is always a complete controller map (neutral when idle).
  """
  @spec step(map() | nil, t()) :: {map(), t()}
  def step(nil, _state), do: {neutral(), new()}

  def step(player, %__MODULE__{} = s) do
    action = trunc(player.action || 0)

    cond do
      action in @damage_air or action == @tumble ->
        s = ensure_tech_plan(s)
        falling? = (player.speed_y_self || 0.0) + (player.speed_y_attack || 0.0) < 0.0
        near_ground? = (player.y || 0.0) < @tech_height

        case s.plan do
          {:tech, dir} when falling? and near_ground? and s.press_frames > 0 ->
            {tech_input(dir), %{s | press_frames: s.press_frames - 1}}

          _ ->
            {neutral(), s}
        end

      action in @downed ->
        s = ensure_getup_plan(s)

        cond do
          s.delay > 0 ->
            {neutral(), %{s | delay: s.delay - 1}}

          action in @down_wait and s.press_frames > 0 ->
            {:getup, choice} = s.plan
            {getup_input(choice), %{s | press_frames: s.press_frames - 1}}

          true ->
            {neutral(), s}
        end

      true ->
        # Out of the knockdown lifecycle: fresh plan next time
        {neutral(), new()}
    end
  end

  # -- Plans --------------------------------------------------------------------

  defp ensure_tech_plan(%{plan: {:tech, _}} = s), do: s
  defp ensure_tech_plan(%{plan: :miss} = s), do: s

  defp ensure_tech_plan(s) do
    plan =
      if :rand.uniform() < 0.7 do
        {:tech, Enum.random([:in_place, :left, :right])}
      else
        :miss
      end

    %{s | plan: plan, press_frames: 2}
  end

  defp ensure_getup_plan(%{plan: {:getup, _}} = s), do: s

  defp ensure_getup_plan(s) do
    choice = Enum.random([:stand, :attack, :roll_left, :roll_right])
    %{s | plan: {:getup, choice}, delay: 5 + :rand.uniform(30), press_frames: 2}
  end

  # -- Inputs --------------------------------------------------------------------

  defp tech_input(dir) do
    stick =
      case dir do
        :in_place -> %{x: 0.5, y: 0.5}
        :left -> %{x: 0.0, y: 0.5}
        :right -> %{x: 1.0, y: 0.5}
      end

    %{neutral() | main_stick: stick, buttons: %{neutral().buttons | r: true}}
  end

  defp getup_input(:stand), do: %{neutral() | main_stick: %{x: 0.5, y: 1.0}}
  defp getup_input(:roll_left), do: %{neutral() | main_stick: %{x: 0.0, y: 0.5}}
  defp getup_input(:roll_right), do: %{neutral() | main_stick: %{x: 1.0, y: 0.5}}
  defp getup_input(:attack), do: %{neutral() | buttons: %{neutral().buttons | a: true}}

  @doc "Neutral controller input in MeleePort.send_controller format."
  @spec neutral() :: map()
  def neutral do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{
        a: false,
        b: false,
        x: false,
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      }
    }
  end
end
