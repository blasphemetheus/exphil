defmodule ExPhil.Eval.TechStats do
  @moduledoc """
  Tech, getting-comboed and defensive-movement statistics of ONE player over
  a game (2026-10-14, the s906 ep2 live look: tech rate 0.26 vs 0.45 and
  wavedashes 2.7 vs 4.6 /min were the two gaps no instrument explained).

  Same contract as `ExPhil.Eval.PlayStats`: `from_game/2` returns mergeable
  raw counts, `summarize/1` turns them into rates, so expert replays and
  bot games are read by the same code.

  Three sections:

    * **tech** — every landing from a knockdown is a tech SITUATION: a
      tech (199..201 in place / forward / backward, 202..204 wall/ceiling)
      or a miss (183 / 191). Techs and missed-tech getups are classed
      TOWARD / AWAY from the opponent (forward rolls go the way the
      player faces, backward rolls the other way). A miss is followed by
      what the player did from the ground (hit while down, neutral getup,
      getup attack, roll toward / away) and what it COST: damage taken
      within `@punish_window` frames, and whether any hit landed at all.
      The same detector on the opponent gives tech-chase offense.

    * **comboed** — `punish_quality.exs`'s opening detector run on the
      player's OWN percent: openings taken per minute, hits and damage per
      opening taken, share that ended in a death. Conversion (the other
      direction) stays in punish_quality; the two are separate questions.

    * **defense** — shields, dash→shield (shield entered within 5 f of
      dashing/running), spot dodges, rolls, and airdodge→land split into
      WAVEDASH (from a jumpsquat, ≤ 4 airborne frames) and WAVELAND (any
      other short airdodge into a landing).
  """

  @punish_window 90
  @combo_gap 45
  @pre_quiet 30
  @kill_window 90

  @techs [199, 200, 201]
  @wall_techs [202, 203, 204]
  @misses [183, 191]
  @lying [184, 192]
  @down_hit [185, 193]
  @getup_neutral [186, 194]
  @getup_attack [187, 195]
  @getup_roll_f [188, 196]
  @getup_roll_b [189, 197]
  @dash_run [20, 21]

  @type frame :: %{own: map(), opp: map()}

  @doc "Raw counts for one game (list of `%{own: Player, opp: Player}` in frame order)."
  @spec from_game([frame()]) :: map()
  def from_game([]), do: empty()

  def from_game(frames) do
    own = frames |> Enum.map(& &1.own) |> List.to_tuple()
    opp = frames |> Enum.map(& &1.opp) |> List.to_tuple()
    n = tuple_size(own)

    %{"frames" => n}
    |> Map.merge(tech_counts(own, opp, n, "tech"))
    |> Map.merge(tech_counts(opp, own, n, "opp_tech"))
    |> Map.merge(tech_chase(own, opp, n))
    |> Map.merge(comboed_counts(own, n))
    |> Map.merge(defense_counts(own, n))
  end

  def empty, do: %{}

  @spec merge(map(), map()) :: map()
  def merge(a, b), do: Map.merge(a, b, fn _k, x, y -> x + y end)

  @doc "Per-minute rates and ratios."
  @spec summarize(map()) :: map()
  def summarize(c) do
    minutes = max(Map.get(c, "frames", 0), 1) / 3600
    g = fn k -> Map.get(c, k, 0) end
    per_min = fn k -> Float.round(g.(k) / minutes, 3) end
    ratio = fn n, d -> if d == 0, do: nil, else: Float.round(n / d, 3) end
    sits = g.("tech_techs") + g.("tech_misses")
    opp_sits = g.("opp_tech_techs") + g.("opp_tech_misses")

    %{
      "minutes" => Float.round(minutes, 1),
      # tech
      "tech_situations_per_min" => Float.round(sits / minutes, 3),
      "tech_rate" => ratio.(g.("tech_techs"), sits),
      "wall_techs_per_min" => per_min.("tech_wall"),
      "tech_in_place_share" => ratio.(g.("tech_in_place"), g.("tech_techs")),
      "tech_toward_share" => ratio.(g.("tech_toward"), g.("tech_techs")),
      "tech_away_share" => ratio.(g.("tech_away"), g.("tech_techs")),
      "tech_punished_share" => ratio.(g.("tech_punished"), g.("tech_techs")),
      "tech_cost_per" => ratio.(g.("tech_cost"), g.("tech_techs")),
      "miss_punished_share" => ratio.(g.("tech_miss_punished"), g.("tech_misses")),
      "miss_cost_per" => ratio.(g.("tech_miss_cost"), g.("tech_misses")),
      "miss_hit_on_ground_share" => ratio.(g.("tech_miss_hit_down"), g.("tech_misses")),
      "miss_getup_neutral_share" => ratio.(g.("tech_miss_neutral"), g.("tech_misses")),
      "miss_getup_attack_share" => ratio.(g.("tech_miss_attack"), g.("tech_misses")),
      "miss_getup_toward_share" => ratio.(g.("tech_miss_toward"), g.("tech_misses")),
      "miss_getup_away_share" => ratio.(g.("tech_miss_away"), g.("tech_misses")),
      "miss_lying_frames_mean" => ratio.(g.("tech_miss_lying"), g.("tech_misses")),
      # tech-chase (offense on the opponent's misses)
      "opp_tech_situations_per_min" => Float.round(opp_sits / minutes, 3),
      "opp_tech_rate" => ratio.(g.("opp_tech_techs"), opp_sits),
      "chase_punished_share" => ratio.(g.("chase_punished"), g.("opp_tech_misses")),
      "chase_cost_per" => ratio.(g.("chase_cost"), g.("opp_tech_misses")),
      # comboed
      "openings_taken_per_min" => per_min.("cb_openings"),
      "hits_per_opening_taken" => ratio.(g.("cb_hits"), g.("cb_openings")),
      "dmg_per_opening_taken" => ratio.(g.("cb_dmg"), g.("cb_openings")),
      "opening_taken_death_share" => ratio.(g.("cb_kills"), g.("cb_openings")),
      "comboed_3plus_share" => ratio.(g.("cb_3plus"), g.("cb_openings")),
      # defense
      "shields_per_min" => per_min.("def_shields"),
      "dash_shields_per_min" => per_min.("def_dash_shields"),
      "shield_time_share" => ratio.(g.("def_shield_frames"), g.("frames")),
      "spotdodges_per_min" => per_min.("def_spotdodges"),
      "rolls_per_min" => per_min.("def_rolls"),
      "wavedashes_per_min" => per_min.("def_wavedashes"),
      "wavelands_per_min" => per_min.("def_wavelands")
    }
  end

  # ---- tech ----------------------------------------------------------------------

  # `me`/`them` are tuples of Player maps; `prefix` names the counts.
  defp tech_counts(me, them, n, prefix) do
    act = fn i -> trunc(elem(me, i).action || 0) end

    init = %{techs: 0, misses: 0, wall: 0, in_place: 0, toward: 0, away: 0,
             punished: 0, cost: 0.0, miss_punished: 0, miss_cost: 0.0, miss_hit_down: 0,
             miss_neutral: 0, miss_attack: 0, miss_toward: 0, miss_away: 0, miss_lying: 0}

    r =
      Enum.reduce(1..(n - 1)//1, init, fn i, a ->
        p = act.(i - 1)
        q = act.(i)

        cond do
          q in @techs and p not in @techs ->
            dir = roll_dir(elem(me, i), elem(them, i), q == 200, q == 201)
            {pun, cost} = damage_after(me, i, n)

            %{a | techs: a.techs + 1,
                  in_place: a.in_place + b2i(dir == :in_place),
                  toward: a.toward + b2i(dir == :toward),
                  away: a.away + b2i(dir == :away),
                  punished: a.punished + b2i(pun), cost: a.cost + cost}

          q in @wall_techs and p not in @wall_techs ->
            %{a | wall: a.wall + 1}

          q in @misses and p not in @misses ->
            {pun, cost} = damage_after(me, i, n)
            {getup, lying} = getup_after(me, them, i, n)

            %{a | misses: a.misses + 1,
                  miss_punished: a.miss_punished + b2i(pun), miss_cost: a.miss_cost + cost,
                  miss_hit_down: a.miss_hit_down + b2i(getup == :hit_down),
                  miss_neutral: a.miss_neutral + b2i(getup == :neutral),
                  miss_attack: a.miss_attack + b2i(getup == :attack),
                  miss_toward: a.miss_toward + b2i(getup == :toward),
                  miss_away: a.miss_away + b2i(getup == :away),
                  miss_lying: a.miss_lying + lying}

          true ->
            a
        end
      end)

    Map.new(r, fn {k, v} -> {"#{prefix}_#{k}", v} end)
  end

  # forward rolls travel the way the player faces; backward the other way
  defp roll_dir(_me, _them, false, false), do: :in_place

  defp roll_dir(me, them, forward?, _backward?) do
    facing = if (me.facing || 1) >= 0, do: 1, else: -1
    to_opp = if (them.x || 0.0) >= (me.x || 0.0), do: 1, else: -1
    travel = if forward?, do: facing, else: -facing
    if travel == to_opp, do: :toward, else: :away
  end

  # {hit within the window?, damage taken within the window} — stops at a stock loss
  defp damage_after(me, i, n) do
    hi = min(i + @punish_window, n - 1)

    Enum.reduce_while((i + 1)..hi//1, {false, 0.0}, fn j, {hit, dmg} ->
      a = elem(me, j - 1)
      b = elem(me, j)

      if (b.stock || 0) < (a.stock || 0) do
        {:halt, {hit, dmg}}
      else
        d = max(0.0, ((b.percent || 0) - (a.percent || 0)) * 1.0)
        {:cont, {hit or d > 0, dmg + d}}
      end
    end)
  end

  # first thing the player did after a missed tech, and how long they lay there
  defp getup_after(me, them, i, n) do
    hi = min(i + 180, n - 1)

    Enum.reduce_while((i + 1)..hi//1, {:none, 0}, fn j, {_, lying} ->
      a = trunc(elem(me, j).action || 0)

      cond do
        a in @lying -> {:cont, {:none, lying + 1}}
        a in @misses -> {:cont, {:none, lying}}
        a in @down_hit -> {:halt, {:hit_down, lying}}
        a in @getup_neutral -> {:halt, {:neutral, lying}}
        a in @getup_attack -> {:halt, {:attack, lying}}
        a in @getup_roll_f -> {:halt, {roll_dir(elem(me, j), elem(them, j), true, false), lying}}
        a in @getup_roll_b -> {:halt, {roll_dir(elem(me, j), elem(them, j), false, true), lying}}
        true -> {:halt, {:none, lying}}
      end
    end)
  end

  # tech-chase: the opponent missed a tech — did WE hit them within the window?
  defp tech_chase(_own, opp, n) do
    act = fn i -> trunc(elem(opp, i).action || 0) end

    Enum.reduce(1..(n - 1)//1, %{"chase_punished" => 0, "chase_cost" => 0.0}, fn i, a ->
      if act.(i) in @misses and act.(i - 1) not in @misses do
        {pun, cost} = damage_after(opp, i, n)
        %{a | "chase_punished" => a["chase_punished"] + b2i(pun), "chase_cost" => a["chase_cost"] + cost}
      else
        a
      end
    end)
  end

  # ---- comboed (punish_quality's opening detector, on our own percent) --------

  defp comboed_counts(me, n) do
    init = %{openings: 0, hits: 0, dmg: 0.0, kills: 0, three_plus: 0, cur: nil, quiet: 9999}

    close = fn a, kill? ->
      c = a.cur
      %{a | openings: a.openings + 1, hits: a.hits + c.hits, dmg: a.dmg + c.dmg,
            kills: a.kills + b2i(kill?), three_plus: a.three_plus + b2i(c.hits >= 3), cur: nil}
    end

    r =
      Enum.reduce(1..(n - 1)//1, init, fn i, a ->
        p = elem(me, i - 1)
        q = elem(me, i)
        d = ((q.percent || 0) - (p.percent || 0)) * 1.0
        rose = d > 0.01 and (q.stock || 0) == (p.stock || 0)
        died = (q.stock || 0) < (p.stock || 0)

        cond do
          rose and a.cur == nil and a.quiet >= @pre_quiet ->
            %{a | cur: %{last: i, dmg: d, hits: 1}, quiet: 0}

          rose and a.cur != nil and i - a.cur.last <= @combo_gap ->
            %{a | cur: %{a.cur | last: i, dmg: a.cur.dmg + d, hits: a.cur.hits + 1}, quiet: 0}

          rose ->
            a = if a.cur, do: close.(a, false), else: a
            cur = if a.quiet >= @pre_quiet, do: %{last: i, dmg: d, hits: 1}, else: nil
            %{a | cur: cur, quiet: 0}

          died and a.cur != nil and i - a.cur.last <= @kill_window ->
            %{close.(a, true) | quiet: a.quiet + 1}

          # an opening stays open for the KILL window, not the combo gap: a
          # KO blow flies 40-80 f, and closing at the gap (punish_quality's
          # original ordering) never saw the death
          a.cur != nil and i - a.cur.last > @kill_window ->
            %{close.(a, false) | quiet: a.quiet + 1}

          true ->
            %{a | quiet: a.quiet + 1}
        end
      end)

    r = if r.cur, do: close.(r, false), else: r

    %{"cb_openings" => r.openings, "cb_hits" => r.hits, "cb_dmg" => r.dmg,
      "cb_kills" => r.kills, "cb_3plus" => r.three_plus}
  end

  # ---- defensive movement --------------------------------------------------------

  defp defense_counts(me, n) do
    act = fn i -> trunc(elem(me, i).action || 0) end
    acts = for i <- 0..(n - 1)//1, do: act.(i)
    runs = acts |> Enum.chunk_by(& &1) |> Enum.map(&{hd(&1), length(&1)})

    # airdodge run of <= 4 frames into a landing; wavedash if a jumpsquat sat
    # within the 3 actions before the airdodge, waveland otherwise
    {wd, wl} =
      runs
      |> Enum.with_index()
      |> Enum.reduce({0, 0}, fn {{a, len}, k}, {wd, wl} ->
        next = Enum.at(runs, k + 1)

        if a == 236 and len <= 4 and match?({43, _}, next) do
          before = runs |> Enum.slice(max(k - 3, 0), min(k, 3)) |> Enum.map(&elem(&1, 0))
          if 24 in before, do: {wd + 1, wl}, else: {wd, wl + 1}
        else
          {wd, wl}
        end
      end)

    r =
      Enum.reduce(1..(n - 1)//1, %{shields: 0, dash_shields: 0, shield_frames: 0, spot: 0, rolls: 0}, fn i, a ->
        q = act.(i)
        p = act.(i - 1)
        recent_dash? = Enum.any?(max(i - 5, 0)..(i - 1)//1, fn j -> act.(j) in @dash_run end)

        %{a | shields: a.shields + b2i(q == 178 and p != 178),
              dash_shields: a.dash_shields + b2i(q == 178 and p != 178 and recent_dash?),
              shield_frames: a.shield_frames + b2i(q in [179, 181]),
              spot: a.spot + b2i(q == 235 and p != 235),
              rolls: a.rolls + b2i(q in [233, 234] and p not in [233, 234])}
      end)

    %{"def_shields" => r.shields, "def_dash_shields" => r.dash_shields, "def_shield_frames" => r.shield_frames,
      "def_spotdodges" => r.spot, "def_rolls" => r.rolls, "def_wavedashes" => wd, "def_wavelands" => wl}
  end

  defp b2i(true), do: 1
  defp b2i(_), do: 0
end
