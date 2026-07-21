defmodule ExPhil.Interp.ReplayStats do
  @moduledoc """
  The standard replay analyses (task #27) — every question we kept
  re-implementing in scratchpad scripts, as one canonical module.

  All functions accept either a replay path (parsed on demand) or the
  pre-extracted lists, so they compose cheaply inside larger analyses.

  Action-state sets mirror `ExPhil.Interp.GroundTruth` / the tech-chase
  scoreboard (`trace_tech_chase.exs`) so numbers stay comparable.
  """

  alias ExPhil.Data.Peppi

  @shield_states [178, 179, 180]
  # ShieldBreakFly/Fall/Down/Stand + FuraFura dizzy — the states a HARD
  # shield break transitions into. Confirmed 2026-07-17 (WS1): 89 of 92
  # shield runs >= 200f across r10-r12 probes ended in 205; hard shield
  # breaks at ~215f from full, so the old 450f "light-shield horizon"
  # threshold could never fire.
  @shield_break_states MapSet.new(205..211)
  @knockdown_entries [183, 191, 199, 200, 201]
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @tech_states [199, 200, 201]

  @doc """
  Parse a replay into per-port frame data used by everything below.

  Returns `%{p1: %{actions, players, controllers}, p2: %{...}, n: frames}`.
  """
  def load(path) do
    {:ok, replay} = Peppi.parse(Path.expand(path))

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    port = fn p ->
      players = Enum.map(frames, fn f -> f.game_state.players[p] end)

      %{
        actions: Enum.map(players, fn pl -> trunc((pl && pl.action) || 0) end),
        players: players
      }
    end

    p1 = port.(1) |> Map.put(:controllers, Enum.map(frames, & &1.controller))

    %{p1: p1, p2: port.(2), n: length(frames)}
  end

  @doc "Frequencies of action states, most common first."
  def action_histogram(actions, top \\ 10) do
    actions |> Enum.frequencies() |> Enum.sort_by(fn {_, c} -> -c end) |> Enum.take(top)
  end

  @doc "Lengths of consecutive runs where the action is in `state_set`."
  def run_lengths(actions, state_set) do
    set = MapSet.new(state_set)

    actions
    |> Enum.chunk_by(&MapSet.member?(set, &1))
    |> Enum.filter(&MapSet.member?(set, hd(&1)))
    |> Enum.map(&length/1)
  end

  @doc """
  Shield behavior summary: fraction of frames shielding, run count,
  run-length percentiles, and OBSERVED shield breaks (`breaks`: shield
  runs whose successor state is in the break family 205..211 — hard
  shield breaks at ~215f from full, which is why run-length p95 pins at
  215 in shield-locked policies).
  """
  def shield_stats(actions) do
    n = length(actions)
    runs = run_lengths(actions, @shield_states)
    frames = Enum.sum(runs)

    %{
      shield_frac: safe_div(frames, n),
      runs: length(runs),
      p50: percentile(runs, 0.5),
      p95: percentile(runs, 0.95),
      max: (if runs == [], do: 0, else: Enum.max(runs)),
      breaks: count_shield_breaks(actions)
    }
  end

  @doc "Shield runs whose successor action is a shield-break state (205..211)."
  def count_shield_breaks(actions) do
    set = MapSet.new(@shield_states)

    actions
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.count(fn [a, b] ->
      MapSet.member?(set, a) and MapSet.member?(@shield_break_states, b)
    end)
  end

  @doc """
  Knockdown episodes for a port's action list: entries into the lifecycle
  from outside, each tagged with its entry class
  (0 = missed tech, 1 = tech in place, 2 = roll F, 3 = roll B).
  Returns `%{episodes, by_class, tech_rate}`.
  """
  def knockdown_episodes(actions) do
    entries =
      actions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [a, b] ->
        b in @knockdown_entries and not MapSet.member?(@lifecycle, a)
      end)
      |> Enum.map(fn [_, b] -> entry_class(b) end)

    n = length(entries)
    teched = Enum.count(entries, &(&1 > 0))

    %{
      episodes: n,
      by_class: Enum.frequencies(entries),
      tech_rate: safe_div(teched, n)
    }
  end

  @doc "Hit events: transitions of a port INTO a hitstun state."
  def hit_events(actions) do
    actions
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.count(fn [a, b] ->
      MapSet.member?(@hitstun, b) and not MapSet.member?(@hitstun, a)
    end)
  end

  @doc """
  Trigger-press scan over a controller list: onsets and durations of
  digital L/R (or analog past `analog_threshold`).
  Returns list of `{onset_index, duration}`.
  """
  def trigger_presses(controllers, analog_threshold \\ 0.3) do
    flags =
      Enum.map(controllers, fn c ->
        Map.get(c, :button_l) || Map.get(c, :button_r) ||
          (Map.get(c, :l_shoulder) || 0) > analog_threshold ||
          (Map.get(c, :r_shoulder) || 0) > analog_threshold
      end)

    flags
    |> Enum.with_index()
    |> Enum.chunk_by(fn {f, _} -> f end)
    |> Enum.filter(fn [{f, _} | _] -> f end)
    |> Enum.map(fn chunk -> {elem(hd(chunk), 1), length(chunk)} end)
  end

  # Approach detection (gate 10). Attack set is approximate: ground
  # attacks + aerials (44..69) and grabs — enough for "did the approach
  # arrive with a threat attached", not a per-move taxonomy.
  @attack_states MapSet.new(Enum.to_list(44..69) ++ [212, 214])
  @approach_start_dist 40.0
  @approach_engage_dist 20.0
  @approach_max_closure_frames 90
  @approach_attack_window 15
  @approach_min_share 0.6

  @doc """
  Approach detection (2026-07-20, from the first human-vs-bot demo: the
  policy rarely initiates in neutral — drills taught the button
  sequence, not the go-in decision).

  An approach = an engagement (horizontal distance <= #{@approach_engage_dist})
  where, within the previous #{@approach_max_closure_frames} frames, distance
  was >= #{@approach_start_dist} AND P1 covered >= #{trunc(@approach_min_share * 100)}%
  of the closed distance (so P2 walking in, or P1 being comboed inward,
  doesn't count), with P1 not in hitstun at the closure start. An
  approach is ARMED when P1 enters an attack/grab state within
  #{@approach_attack_window} frames of engagement.

  Takes the `load/1` shape maps for both ports. Returns
  `%{approaches, armed, per_min, armed_per_min}` (per_min normalized by
  frames/3600).
  """
  def approach_stats(%{actions: p1_actions, players: p1_players} = p1, %{players: _} = p2) do
    n = length(p1_players)
    approaches = detect_approaches(p1, p2)

    armed =
      Enum.count(approaches, fn %{onset: i} ->
        p1_actions
        |> Enum.slice(i, @approach_attack_window)
        |> Enum.any?(&MapSet.member?(@attack_states, &1))
      end)

    %{
      approaches: length(approaches),
      armed: armed,
      per_min: safe_div(length(approaches) * 3600, n),
      armed_per_min: safe_div(armed * 3600, n)
    }
  end

  # Conversion detection (r15 pool sampling). An approach CONVERTS when the
  # opponent enters hitstun within @conversion_window frames of engagement —
  # the pool sampler upweights the whole decision arc (closure start through
  # the conversion window), not just the payoff.
  @conversion_window 45

  @doc """
  Conversions (2026-07-20, r15): approaches (as in `approach_stats/2`)
  where P2 enters hitstun within #{@conversion_window} frames of the
  engagement onset.

  Takes the `load/1` shape maps for both ports. Returns
  `%{approaches, conversions, spans}` where each span is
  `{closure_start, onset + #{@conversion_window}}` (clamped to the replay),
  i.e. the frame range conversion-weighted pool sampling upweights.

  Shares `detect_approaches/2` with `approach_stats/2` (consolidated
  2026-07-20 after the post-run test gate passed).
  """
  def conversion_stats(%{actions: _} = p1, %{actions: p2_actions} = p2) do
    n = length(p2_actions)
    approaches = detect_approaches(p1, p2)

    converting =
      Enum.filter(approaches, fn %{onset: i} ->
        # Entry-only: P2 must TRANSITION into hitstun inside the window, so
        # approaching an already-comboed opponent doesn't count as a fresh
        # conversion. Slice starts one frame early to catch entry at onset.
        p2_actions
        |> Enum.slice(max(i - 1, 0), @conversion_window + 1)
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.any?(fn [a, b] ->
          MapSet.member?(@hitstun, b) and not MapSet.member?(@hitstun, a)
        end)
      end)

    spans =
      Enum.map(converting, fn %{onset: i, closure_start: j0} ->
        {j0, min(i + @conversion_window, n - 1)}
      end)

    %{approaches: length(approaches), conversions: length(converting), spans: spans}
  end

  # Same detection as approach_stats/2 but returns per-approach detail
  # (engagement onset + closure start) instead of aggregate counts.
  defp detect_approaches(%{actions: p1_actions, players: p1_players}, %{players: p2_players}) do
    dist =
      Enum.zip(p1_players, p2_players)
      |> Enum.map(fn
        {%{x: x1}, %{x: x2}} when is_number(x1) and is_number(x2) -> abs(x1 - x2)
        _ -> nil
      end)

    engaged? = fn
      nil -> false
      d -> d <= @approach_engage_dist
    end

    onsets =
      dist
      |> Enum.with_index()
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [{a, _}, {b, _}] -> not engaged?.(a) and engaged?.(b) end)
      |> Enum.map(fn [_, {_, i}] -> i end)

    p1x = Enum.map(p1_players, fn pl -> pl && pl.x end)

    Enum.flat_map(onsets, fn i ->
      lo = max(i - @approach_max_closure_frames, 0)

      case lo..i |> Enum.map(fn j -> {Enum.at(dist, j), j} end) |> Enum.reject(&is_nil(elem(&1, 0))) do
        [] ->
          []

        window ->
          {d0, j0} = Enum.max_by(window, &elem(&1, 0))
          closed = d0 - (Enum.at(dist, i) || 0.0)

          p1_moved = abs((Enum.at(p1x, i) || 0.0) - (Enum.at(p1x, j0) || 0.0))
          free? = not MapSet.member?(@hitstun, Enum.at(p1_actions, j0) || 0)

          if d0 >= @approach_start_dist and closed > 0 and free? and
               p1_moved / max(closed, 1.0e-6) >= @approach_min_share do
            [%{onset: i, closure_start: j0}]
          else
            []
          end
      end
    end)
  end

  # A fresh OPENING (vs a combo continuation): a hitstun entry with no
  # hitstun frame in the preceding cooldown window. 10 hits in one juggle
  # = 1 opening; hit_events/1 would count all 10.
  @opening_cooldown 60

  @doc """
  Openings created against a port (r15, 2026-07-20 — the punish expert's
  whole thesis is "create openings"; this makes regression on it visible).

  Takes the opponent's action list. An opening = a transition into
  hitstun with no hitstun frame in the previous #{@opening_cooldown}
  frames; entries inside the window are combo continuations. Returns
  `%{openings, hit_entries, per_min, entries_per_min}` (per_min
  normalized by frames/3600).
  """
  def opening_stats(opp_actions, opts \\ []) do
    cooldown = Keyword.get(opts, :cooldown, @opening_cooldown)
    n = length(opp_actions)

    {openings, entries, _last_hit} =
      opp_actions
      |> Enum.with_index()
      |> Enum.reduce({0, 0, -cooldown - 1}, fn {a, i}, {op, en, last_hit} ->
        cond do
          not MapSet.member?(@hitstun, a) -> {op, en, last_hit}
          # Previous frame was hitstun: mid-run, not an entry
          last_hit == i - 1 -> {op, en, i}
          # Entry after a clean gap: a fresh opening
          i - last_hit > cooldown -> {op + 1, en + 1, i}
          # Entry within the window: same opening (combo re-hit)
          true -> {op, en + 1, i}
        end
      end)

    %{
      openings: openings,
      hit_entries: entries,
      per_min: safe_div(openings * 3600, n),
      entries_per_min: safe_div(entries * 3600, n)
    }
  end

  @doc "Fraction of frames a port is airborne."
  def airborne_frac(players) do
    n = length(players)
    air = Enum.count(players, fn pl -> pl && pl.on_ground == false end)
    safe_div(air, n)
  end

  @doc "Percentile that returns nil (not 0) on empty — for no-evidence gates."
  def percentile_or_nil([], _), do: nil
  def percentile_or_nil(list, p), do: percentile(list, p)

  @doc """
  The standard one-line game profile (ports fixed: p1 = policy, p2 = dummy).
  """
  def summarize(path) do
    d = load(path)
    shield = shield_stats(d.p1.actions)
    kd = knockdown_episodes(d.p2.actions)

    %{
      path: Path.basename(path),
      frames: d.n,
      shield: shield,
      approach: approach_stats(d.p1, d.p2),
      airborne_frac: airborne_frac(d.p1.players),
      opp_knockdowns: kd.episodes,
      opp_tech_rate: kd.tech_rate,
      opp_tech_classes: kd.by_class,
      opp_hit_events: hit_events(d.p2.actions),
      openings: opening_stats(d.p2.actions),
      tech_frames: Enum.count(d.p2.actions, &(&1 in @tech_states))
    }
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp entry_class(a) when a in [183, 191], do: 0
  defp entry_class(199), do: 1
  defp entry_class(200), do: 2
  defp entry_class(201), do: 3

  defp safe_div(_, 0), do: 0.0
  defp safe_div(a, b), do: Float.round(a / b, 4)

  defp percentile([], _), do: 0

  defp percentile(list, p) do
    sorted = Enum.sort(list)
    Enum.at(sorted, min(trunc(p * length(sorted)), length(sorted) - 1))
  end
end
