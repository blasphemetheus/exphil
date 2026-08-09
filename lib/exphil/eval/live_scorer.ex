defmodule ExPhil.Eval.LiveScorer do
  @moduledoc """
  Live behavior scoring for a play session — the analyze_behavior.exs
  metrics computed DURING the game instead of from the replay afterwards.

  Wraps `Melee.GameEvents` (libmelee_ex's semantic event stream: stock
  losses classified SD-vs-KO, shield breaks, game boundaries) and adds
  the per-frame accumulators events can't carry: action diversity,
  damage dealt, shield time, offstage time, input-change rate.

  Fold every stepped game state through `step/3`; `report/1` returns the
  same row shape `scripts/analyze_behavior.exs` prints, so live-scored
  and replay-scored numbers are directly comparable.

      scorer = LiveScorer.new(1)
      scorer = LiveScorer.step(scorer, game_state, controller_sent)
      LiveScorer.report(scorer)
      #=> %{seconds: ..., stocks_lost: ..., sd_deaths: ..., ...}

  `game_state` is the BRIDGE game state (`ExPhil.Bridge.GameState`, what
  `MeleePort.step/3` returns) — internally adapted to the minimal
  `Melee.GameState` shape `Melee.GameEvents` folds over.
  """

  alias ExPhil.Bridge

  # Same sets as Melee.GameEvents / Interp.ReplayStats.
  @shield_states MapSet.new([178, 179, 180])

  defstruct [
    :bot_port,
    :opp_port,
    :tracker,
    frames: 0,
    actions_seen: MapSet.new(),
    shield_frames: 0,
    offstage_frames: 0,
    damage_dealt: 0.0,
    prev_opp_percent: nil,
    input_changes: 0,
    prev_controller: nil,
    stock_events: [],
    shield_breaks: 0,
    game_started: false,
    game_ended: false
  ]

  @type t :: %__MODULE__{}

  @doc "Fresh scorer for the bot on `bot_port` (opponent inferred as the other 2P port)."
  @spec new(1..4) :: t()
  def new(bot_port) do
    %__MODULE__{
      bot_port: bot_port,
      opp_port: if(bot_port == 1, do: 2, else: 1),
      tracker: Melee.GameEvents.new()
    }
  end

  @doc """
  Fold one frame. `controller_sent` is whatever the caller pressed this
  frame (any comparable term — used only for input-change counting) or
  `nil` to skip that metric.
  """
  @spec step(t(), Bridge.GameState.t(), term()) :: t()
  def step(%__MODULE__{} = scorer, %Bridge.GameState{} = gs, controller_sent \\ nil) do
    {events, tracker} = Melee.GameEvents.step(scorer.tracker, adapt(gs))
    scorer = %{scorer | tracker: tracker}
    scorer = Enum.reduce(events, scorer, &apply_event/2)

    if Bridge.GameState.in_game?(gs) do
      bot = Bridge.GameState.get_player(gs, scorer.bot_port)
      opp = Bridge.GameState.get_player(gs, scorer.opp_port)

      scorer
      |> count_frame(bot)
      |> count_damage(opp)
      |> count_inputs(controller_sent)
    else
      scorer
    end
  end

  @doc "The behavior row: same columns as scripts/analyze_behavior.exs."
  @spec report(t()) :: map()
  def report(%__MODULE__{} = s) do
    minutes = s.frames / 3600

    sd = Enum.count(s.stock_events, &(&1.kind == :sd))
    ko = Enum.count(s.stock_events, &(&1.kind == :ko and &1.port == s.bot_port))
    bot_losses = Enum.count(s.stock_events, &(&1.port == s.bot_port))
    taken = Enum.count(s.stock_events, &(&1.port == s.opp_port))

    %{
      seconds: Float.round(s.frames / 60, 1),
      stocks_lost: bot_losses,
      sd_deaths: Enum.count(s.stock_events, &(&1.port == s.bot_port and &1.kind == :sd)),
      ko_deaths: ko,
      stocks_taken: taken,
      damage_dealt: Float.round(s.damage_dealt, 1),
      shield_pct: pct(s.shield_frames, s.frames),
      shieldbreaks: s.shield_breaks,
      distinct_actions: MapSet.size(s.actions_seen),
      inputs_per_min: Float.round(s.input_changes / max(minutes, 1.0e-9), 0),
      offstage_pct: pct(s.offstage_frames, s.frames),
      # Bookkeeping the replay path can't give this cheaply:
      sd_deaths_any_port: sd,
      game_started: s.game_started,
      game_ended: s.game_ended
    }
  end

  ## Event handling

  defp apply_event({:game_start, _}, s), do: %{s | game_started: true}
  defp apply_event({:game_end, _}, s), do: %{s | game_ended: true}

  defp apply_event({:stock_lost, ev}, s), do: %{s | stock_events: s.stock_events ++ [ev]}

  defp apply_event({:shield_break, %{port: port}}, %{bot_port: port} = s),
    do: %{s | shield_breaks: s.shield_breaks + 1}

  defp apply_event(_other, s), do: s

  ## Per-frame accumulators

  defp count_frame(s, nil), do: %{s | frames: s.frames + 1}

  defp count_frame(s, bot) do
    action = as_int(bot.action)

    offstage? =
      is_number(bot.x) and is_number(bot.y) and (abs(bot.x) > 85.0 or bot.y < -10.0)

    %{
      s
      | frames: s.frames + 1,
        actions_seen: if(action, do: MapSet.put(s.actions_seen, action), else: s.actions_seen),
        shield_frames: s.shield_frames + bool_to_int(MapSet.member?(@shield_states, action)),
        offstage_frames: s.offstage_frames + bool_to_int(offstage?)
    }
  end

  defp count_damage(s, nil), do: %{s | prev_opp_percent: nil}

  defp count_damage(s, opp) do
    cur = opp.percent

    dealt =
      if is_number(cur) and is_number(s.prev_opp_percent) and cur > s.prev_opp_percent,
        do: cur - s.prev_opp_percent,
        else: 0.0

    %{s | damage_dealt: s.damage_dealt + dealt, prev_opp_percent: cur}
  end

  defp count_inputs(s, nil), do: s

  defp count_inputs(s, controller) do
    changed = s.prev_controller != nil and controller != s.prev_controller
    %{s | input_changes: s.input_changes + bool_to_int(changed), prev_controller: controller}
  end

  ## Bridge -> Melee adaptation (the minimal fields GameEvents reads)

  defp adapt(%Bridge.GameState{} = gs) do
    players =
      for {port, p} <- gs.players || %{}, p != nil, into: %{} do
        {port,
         %Melee.PlayerState{
           stock: p.stock,
           percent: p.percent,
           action: as_int(p.action),
           character: p.character
         }}
      end

    %Melee.GameState{menu_state: gs.menu_state, stage: gs.stage, players: players}
  end

  defp as_int(a) when is_integer(a), do: a
  defp as_int(a) when is_number(a), do: trunc(a)
  defp as_int(_), do: nil

  defp bool_to_int(true), do: 1
  defp bool_to_int(false), do: 0

  defp pct(_n, 0), do: 0.0
  defp pct(n, total), do: Float.round(n / total * 100, 1)
end
