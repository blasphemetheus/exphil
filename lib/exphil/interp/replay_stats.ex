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
  Shield behavior summary: fraction of frames shielding, run count, and
  run-length percentiles (shield break ~ 450f of full hold).
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
      breaks_risked: Enum.count(runs, &(&1 >= 450))
    }
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

  @doc "Fraction of frames a port is airborne."
  def airborne_frac(players) do
    n = length(players)
    air = Enum.count(players, fn pl -> pl && pl.on_ground == false end)
    safe_div(air, n)
  end

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
      airborne_frac: airborne_frac(d.p1.players),
      opp_knockdowns: kd.episodes,
      opp_tech_rate: kd.tech_rate,
      opp_tech_classes: kd.by_class,
      opp_hit_events: hit_events(d.p2.actions),
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
