defmodule ExPhil.Interp.ReportCard do
  @moduledoc """
  Tier-1 report card (task #27): mechanics-health gates for a policy replay.

  Composes the pathology detectors proven on 2026-07-15 (jump discipline,
  shield discipline, idle deadlock, press durations) into one PASS/FAIL
  readout per replay. Gates calibrated from the user's fixtures with
  headroom; see docs/planning/INTERP_ROADMAP.md pathology #4 entries.

  Port 1 is assumed to be the policy. The gate MATH lives here so it is
  unit-testable and so loops can consume a numeric score programmatically
  (round selection, save-best); `scripts/report_card.exs` is the CLI printer.
  """

  alias ExPhil.Interp.ReplayStats

  @jumpsquat 24
  @rising MapSet.new([25, 26])
  @dj_states MapSet.new([27, 28])
  @airborne_states MapSet.new(25..34)
  @shield_states MapSet.new(178..182)
  # Post-shield options (approximate ID sets; good enough for trend gates)
  @oos_jump MapSet.new(24..28)
  @oos_escape MapSet.new(233..235)
  @oos_grab MapSet.new([212, 213])

  @type gate :: %{name: String.t(), value: term(), pass: boolean(), target: String.t()}

  @doc "Evaluate all gates for a replay path. Returns `%{gates, passed, total}`."
  @spec evaluate_path(Path.t()) :: %{gates: [gate()], passed: non_neg_integer(), total: pos_integer()}
  def evaluate_path(path), do: path |> ReplayStats.load() |> evaluate()

  @doc """
  Evaluate all gates on pre-loaded replay data (`ReplayStats.load/1` shape:
  needs `p1.actions`, `p1.controllers`).
  """
  @spec evaluate(map()) :: %{gates: [gate()], passed: non_neg_integer(), total: pos_integer()}
  def evaluate(%{p1: %{actions: actions, controllers: controllers}}) do
    shield = ReplayStats.shield_stats(actions)

    jump_rate = jump_rate(actions)
    {dj_pct, dj_p50} = dj_stats(actions)
    oos_idle_pct = oos_idle_pct(actions)
    max_idle = actions |> ReplayStats.run_lengths([14]) |> Enum.max(fn -> 0 end)
    xy_p50 = xy_press_p50(controllers)

    gates = [
      gate("jumps/100 grounded", jump_rate, jump_rate <= 2.5, "<=2.5; user 1.0-1.5"),
      gate("DJ per air stint %", dj_pct, dj_pct <= 35.0, "<=35; user 13-21"),
      gate("liftoff->DJ p50 (f)", dj_p50, dj_p50 == nil or dj_p50 >= 8, ">=8; user 12-16"),
      gate("shield occupancy %", shield.shield_frac * 100, shield.shield_frac <= 0.04, "<=4"),
      gate("shield run p95 (f)", shield.p95 || 0, (shield.p95 || 0) <= 180, "<=180"),
      gate("OOS idle %", oos_idle_pct, oos_idle_pct <= 50.0, "<=50"),
      gate("max idle streak (f)", max_idle, max_idle <= 300, "<=300 (5s; deadlock detector)"),
      gate("X/Y press p50 (f)", xy_p50, xy_p50 == nil or xy_p50 <= 6, "<=6 (SH-capable)"),
      # Gate 9 (added 2026-07-17 after WS1 confirmed the constant-215
      # mystery = hard shield BREAKS: 89 across r10-r12 probes). Scores
      # from here on are x/9 — not directly comparable to r1-r12's x/8.
      gate("shield breaks", shield.breaks, shield.breaks == 0, "==0 (broke 89x in r10-r12)")
    ]

    %{gates: gates, passed: Enum.count(gates, & &1.pass), total: length(gates)}
  end

  @doc "Convenience: passed-gate count for a replay path (loop-facing score)."
  @spec score(Path.t()) :: non_neg_integer()
  def score(path), do: evaluate_path(path).passed

  # -- jump discipline ---------------------------------------------------

  @doc "Jumpsquat entries per 100 grounded frames."
  def jump_rate(actions) do
    grounded = Enum.count(actions, &(&1 not in 24..34))

    squats =
      actions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.count(fn [a, b] -> a != @jumpsquat and b == @jumpsquat end)

    squats * 100 / max(grounded, 1)
  end

  @doc """
  Double-jump behavior per air stint: `{pct_of_stints_with_DJ,
  p50_frames_from_liftoff_to_DJ}` (p50 nil when no stint ever DJs).
  """
  def dj_stats(actions) do
    stints =
      actions
      |> Enum.chunk_by(&MapSet.member?(@airborne_states, &1))
      |> Enum.filter(fn chunk -> MapSet.member?(@rising, hd(chunk)) end)

    dj_idx = Enum.map(stints, fn c -> Enum.find_index(c, &MapSet.member?(@dj_states, &1)) end)
    dj_pct = Enum.count(dj_idx, & &1) * 100 / max(length(stints), 1)
    dj_p50 = pct(Enum.filter(dj_idx, & &1), 0.5)
    {dj_pct, dj_p50}
  end

  # -- shield discipline ---------------------------------------------------

  @doc "Percentage of shield exits with no jump/escape/grab follow-up within 20f."
  def oos_idle_pct(actions) do
    oos =
      actions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.filter(fn {[a, b], _} ->
        MapSet.member?(@shield_states, a) and not MapSet.member?(@shield_states, b)
      end)
      |> Enum.map(fn {_, i} ->
        window = Enum.slice(actions, i + 1, 20)

        cond do
          Enum.any?(window, &MapSet.member?(@oos_jump, &1)) -> :jump
          Enum.any?(window, &MapSet.member?(@oos_escape, &1)) -> :escape
          Enum.any?(window, &MapSet.member?(@oos_grab, &1)) -> :grab
          true -> :idle
        end
      end)
      |> Enum.frequencies()

    oos_total = oos |> Map.values() |> Enum.sum()
    Map.get(oos, :idle, 0) * 100 / max(oos_total, 1)
  end

  # -- press durations -----------------------------------------------------

  @doc "p50 of X/Y button hold durations in frames (nil if never pressed)."
  def xy_press_p50(controllers) do
    xy = press_runs(controllers, :button_x) ++ press_runs(controllers, :button_y)
    pct(xy, 0.5)
  end

  defp press_runs(controllers, key) do
    controllers
    |> Enum.map(fn c -> if c && Map.get(c, key), do: 1, else: 0 end)
    |> Enum.chunk_by(& &1)
    |> Enum.filter(&(hd(&1) == 1))
    |> Enum.map(&length/1)
  end

  # -- shared --------------------------------------------------------------

  @doc "Percentile with the report card's index convention (nil on empty)."
  def pct(xs, p) do
    case Enum.sort(xs) do
      [] -> nil
      s -> Enum.at(s, min(length(s) - 1, floor(p * length(s))))
    end
  end

  defp gate(name, value, pass, target), do: %{name: name, value: value, pass: pass, target: target}
end
