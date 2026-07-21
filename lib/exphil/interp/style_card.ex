defmodule ExPhil.Interp.StyleCard do
  @moduledoc """
  Humanlike-play style card (task #28, 2026-07-20) — SEPARATE from the
  mechanics-health `ReportCard` so the x/10 health score stays comparable
  round-over-round while style gates evolve per character.

  Two kinds of gate live here:
  - **Anti-superhuman ceilings** (reaction-time floor, anti-metronome,
    APM band): humanlike play has an upper bound, not just a floor. BC
    inherits human timing; the day an RL fine-tune reacts in 2 frames,
    these fail loudly.
  - **Situational-competence floors** (DI presence, recovery attempt
    rate, SD cap): things every human does that nets quietly drop.

  ALL thresholds are PROVISIONAL until calibrated against the human
  corpus (task #30 — real games of the 5-char set scored on these same
  metrics). Character packs override thresholds per char (task #29/#33:
  option-quality priors are char-specific — raw entropy is NOT the
  model; Mewtwo fair-OOS-heavy is correct play, not a mixup failure).

  Port 1 is the policy, as everywhere else in interp.
  """

  alias ExPhil.Interp.ReplayStats

  @attack_states MapSet.new(Enum.to_list(44..69) ++ [212, 214])
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @dj_states MapSet.new([27, 28])
  # Opponent-vulnerable events we expect a punish reaction to: missed-tech
  # bounces + special-fall landing lag. Approximate on purpose — trend gate.
  @vulnerable_entries MapSet.new([183, 191, 43])
  @special_floor 341

  # Offstage geometry (FD-calibrated, same as dj_recovery_check.exs)
  @edge_x 88.0
  @offstage_y -6.0
  @gap_merge 10
  @min_stretch 15

  @reaction_window 60
  @min_presses_for_cv 10
  @di_deflection 0.15
  @min_hitstun_stint 5

  @default_pack %{
    reaction_p50_min: 13,
    metronome_cv_min: 0.2,
    apm_band: {120, 700},
    di_pct_min: 60.0,
    recovery_rate_min: 0.8,
    sd_max: 1
  }

  # Char packs override thresholds (and later add pack-specific gates:
  # spacing bands, L-cancel with #31 lag tables, option priors from #29).
  @packs %{
    mewtwo: %{},
    zelda: %{},
    game_and_watch: %{},
    fox: %{},
    ice_climbers: %{}
  }

  @type gate :: %{name: String.t(), value: term(), pass: boolean(), target: String.t()}

  @doc "Evaluate the style card for a replay path. `opts[:char]` picks a pack."
  def evaluate_path(path, opts \\ []), do: path |> ReplayStats.load() |> evaluate(opts)

  @doc "Evaluate on pre-loaded `ReplayStats.load/1` data."
  def evaluate(%{p1: p1, p2: p2, n: n}, opts \\ []) do
    pack = pack(opts[:char])

    r_p50 = reaction_p50(p1.actions, p2.actions)
    cv = press_interval_cv(p1.controllers)
    apm = apm(p1.controllers, n)
    di = di_presence_pct(p1.actions, p1.controllers)
    rec = recovery_stats(p1)
    {lo, hi} = pack.apm_band

    gates = [
      gate(
        "reaction p50 (f)",
        r_p50,
        r_p50 == nil or r_p50 >= pack.reaction_p50_min,
        ">=#{pack.reaction_p50_min} (anti-superhuman; PROVISIONAL)"
      ),
      gate(
        "press interval CV",
        cv,
        cv == nil or cv >= pack.metronome_cv_min,
        ">=#{pack.metronome_cv_min} (anti-metronome; PROVISIONAL)"
      ),
      gate("inputs/min", apm, apm >= lo and apm <= hi, "#{lo}..#{hi} (PROVISIONAL)"),
      gate(
        "DI presence %",
        di,
        di == nil or di >= pack.di_pct_min,
        ">=#{pack.di_pct_min} of hitstun stints (PROVISIONAL)"
      ),
      gate(
        "recovery attempt rate",
        rec.rate,
        rec.rate == nil or rec.rate >= pack.recovery_rate_min,
        ">=#{pack.recovery_rate_min} of offstage situations (PROVISIONAL)"
      ),
      gate("SDs", rec.sds, rec.sds <= pack.sd_max, "<=#{pack.sd_max} (PROVISIONAL)")
    ]

    %{gates: gates, passed: Enum.count(gates, & &1.pass), total: length(gates)}
  end

  @doc "Threshold pack for a character (unknown/nil -> defaults)."
  def pack(char), do: Map.merge(@default_pack, Map.get(@packs, char, %{}))

  # -- anti-superhuman -----------------------------------------------------

  @doc """
  p50 frames from an opponent-vulnerable event (missed-tech bounce,
  special-fall landing) to P1's next attack initiation within
  #{@reaction_window}f. nil when no reacted-to events (no evidence, pass).
  """
  def reaction_p50(p1_actions, p2_actions) do
    events =
      p2_actions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index(1)
      |> Enum.filter(fn {[a, b], _} ->
        MapSet.member?(@vulnerable_entries, b) and not MapSet.member?(@vulnerable_entries, a)
      end)
      |> Enum.map(&elem(&1, 1))

    attack_entries =
      p1_actions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index(1)
      |> Enum.filter(fn {[a, b], _} ->
        MapSet.member?(@attack_states, b) and not MapSet.member?(@attack_states, a)
      end)
      |> Enum.map(&elem(&1, 1))

    deltas =
      Enum.flat_map(events, fn e ->
        case Enum.find(attack_entries, fn a -> a > e and a - e <= @reaction_window end) do
          nil -> []
          a -> [a - e]
        end
      end)

    ReplayStats.percentile_or_nil(deltas, 0.5)
  end

  @doc """
  Coefficient of variation (std/mean) of inter-onset intervals for the
  most-pressed button among A/B/X/Y. nil below #{@min_presses_for_cv}
  presses (no evidence). A metronome scores ~0; humans jitter.
  """
  def press_interval_cv(controllers) do
    onsets_by_button =
      for key <- [:button_a, :button_b, :button_x, :button_y] do
        controllers
        |> Enum.map(&((&1 && Map.get(&1, key)) || false))
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.with_index(1)
        |> Enum.filter(fn {[a, b], _} -> not a and b end)
        |> Enum.map(&elem(&1, 1))
      end

    onsets = Enum.max_by(onsets_by_button, &length/1)

    if length(onsets) < @min_presses_for_cv do
      nil
    else
      intervals =
        onsets |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end)

      mean = Enum.sum(intervals) / length(intervals)

      std =
        :math.sqrt(
          Enum.sum(Enum.map(intervals, fn i -> (i - mean) * (i - mean) end)) /
            length(intervals)
        )

      Float.round(std / max(mean, 1.0e-6), 4)
    end
  end

  @doc "Button onsets (all digital buttons) per minute of game time."
  def apm(controllers, n) do
    buttons = [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]

    onsets =
      Enum.reduce(buttons, 0, fn key, acc ->
        acc +
          (controllers
           |> Enum.map(&((&1 && Map.get(&1, key)) || false))
           # Leading false so a press already down on frame 0 counts as an
           # onset (pairwise scan otherwise drops it — off-by-one APM)
           |> then(&[false | &1])
           |> Enum.chunk_every(2, 1, :discard)
           |> Enum.count(fn [a, b] -> not a and b end))
      end)

    Float.round(onsets * 3600 / max(n, 1), 1)
  end

  # -- situational competence ----------------------------------------------

  @doc """
  Percentage of hitstun stints (>= #{@min_hitstun_stint}f) where the main
  stick deflects >= #{@di_deflection} from center at least once — i.e.
  the policy DIs instead of riding hits with a neutral stick. nil when
  the policy was never in hitstun.
  """
  def di_presence_pct(actions, controllers) do
    stints =
      actions
      |> Enum.with_index()
      |> Enum.chunk_by(fn {a, _} -> MapSet.member?(@hitstun, a) end)
      |> Enum.filter(fn [{a, _} | _] = chunk ->
        MapSet.member?(@hitstun, a) and length(chunk) >= @min_hitstun_stint
      end)

    case stints do
      [] ->
        nil

      _ ->
        with_di =
          Enum.count(stints, fn chunk ->
            Enum.any?(chunk, fn {_, i} ->
              case Enum.at(controllers, i) do
                nil ->
                  false

                c ->
                  ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
                  x = (Map.get(ms, :x) || 0.5) - 0.5
                  y = (Map.get(ms, :y) || 0.5) - 0.5
                  :math.sqrt(x * x + y * y) >= @di_deflection
              end
            end)
          end)

        Float.round(with_di * 100 / length(stints), 1)
    end
  end

  @doc """
  Offstage recovery behavior: `%{situations, attempts, rate, sds}`.
  A situation = an offstage airborne stretch (same geometry as
  dj_recovery_check.exs); an attempt = a DJ or any special-move state
  during the stretch. `sds` approximates self-destructs as stock losses
  with no hitstun in the preceding 60f. Char-specific recovery
  taxonomies (task #32) will replace the attempt heuristic.
  """
  def recovery_stats(%{actions: actions, players: players}) do
    stretches =
      players
      |> Enum.with_index()
      |> Enum.filter(fn {pl, _} ->
        pl != nil and pl.on_ground == false and
          (abs(pl.x || 0.0) > @edge_x or (pl.y || 0.0) < @offstage_y)
      end)
      |> Enum.map(&elem(&1, 1))
      |> Enum.reduce([], fn i, acc ->
        case acc do
          [{s, e} | rest] when i - e <= @gap_merge -> [{s, i} | rest]
          _ -> [{i, i} | acc]
        end
      end)
      |> Enum.filter(fn {s, e} -> e - s + 1 >= @min_stretch end)
      |> Enum.reverse()

    attempts =
      Enum.count(stretches, fn {s, e} ->
        actions
        |> Enum.slice(s, e - s + 1)
        |> Enum.any?(fn a -> MapSet.member?(@dj_states, a) or a >= @special_floor end)
      end)

    n_sit = length(stretches)

    %{
      situations: n_sit,
      attempts: attempts,
      rate: if(n_sit == 0, do: nil, else: Float.round(attempts / n_sit, 3)),
      sds: count_sds(actions, players)
    }
  end

  defp count_sds(actions, players) do
    stocks = Enum.map(players, fn pl -> pl && trunc(pl.stock || 0) end)

    stocks
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.count(fn {[a, b], i} ->
      is_integer(a) and is_integer(b) and b < a and
        not Enum.any?(
          Enum.slice(actions, max(i - 60, 0), min(i, 60) + 1),
          &MapSet.member?(@hitstun, &1)
        )
    end)
  end

  defp gate(name, value, pass, target), do: %{name: name, value: value, pass: pass, target: target}
end
