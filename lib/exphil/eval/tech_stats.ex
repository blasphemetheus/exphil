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

    * **out of shield** (Bradley, 10-14: "you can jump cancel, jump then
      aerial, wavedash out of shield") — every shield episode (a run of
      178..182) is classed by how it ended. A jumpsquat (24) within 3 f of
      leaving shield is a JUMP CANCEL, then by what follows it within 12 f:
      upsmash (63), grab (212), shine (360/361 — Fox's shine OOS is
      jumpsquat → shine), wavedash (airdodge ≤ 4 f → landing), aerial
      (65..69 / 366 within 20 f), or a plain jump. Without a jump cancel:
      shield-grab (212 straight from shield), roll, spot dodge, hit out of
      shield (hitstun), or a shield DROP into anything else. PRESSURED =
      the episode contained shield stun (181); CONNECTED = the subject
      dealt damage within 30 f of leaving shield.
  """

  @shield_states [178, 179, 180, 181, 182]
  @oos_options ~w(jc_upsmash jc_grab jc_shine jc_wavedash jc_aerial jc_jump shield_grab roll spotdodge hit drop)

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
    |> Map.merge(oos_counts(own, opp, n))
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
      "wavelands_per_min" => per_min.("def_wavelands"),
      # out of shield
      "oos_exits_per_min" => per_min.("oos_exits"),
      "oos_pressured_share" => ratio.(g.("oos_pressured"), g.("oos_exits")),
      "oos_jc_share" => ratio.(Enum.sum(for o <- @oos_options, String.starts_with?(o, "jc_"), do: g.("oos_#{o}")), g.("oos_exits")),
      "oos_connected_share" => ratio.(g.("oos_connected"), g.("oos_exits")),
      "oos_pressured_connected_share" => ratio.(g.("oos_pressured_connected"), g.("oos_pressured"))
    }
    |> Map.merge(Map.new(@oos_options, fn o -> {"oos_#{o}_share", ratio.(g.("oos_#{o}"), g.("oos_exits"))} end))
    |> Map.merge(Map.new(@oos_options, fn o -> {"oos_pressured_#{o}_share", ratio.(g.("oos_pressured_#{o}"), g.("oos_pressured"))} end))
  end

  @doc "Out-of-shield option names, in report order."
  def oos_options, do: @oos_options

  # ---- out of shield ---------------------------------------------------------------

  defp oos_counts(me, them, n) do
    act = fn i -> trunc(elem(me, i).action || 0) end
    shield? = fn i -> act.(i) in @shield_states end

    init =
      %{"oos_exits" => 0, "oos_pressured" => 0, "oos_connected" => 0, "oos_pressured_connected" => 0}
      |> Map.merge(Map.new(@oos_options, &{"oos_#{&1}", 0}))
      |> Map.merge(Map.new(@oos_options, &{"oos_pressured_#{&1}", 0}))

    {counts, _in_shield, _stunned} =
      Enum.reduce(1..(n - 1)//1, {init, shield?.(0), false}, fn i, {c, in_shield, stunned} ->
        cond do
          shield?.(i) ->
            {c, true, stunned or act.(i) == 181}

          in_shield ->
            # left shield on frame i
            option = oos_option(me, i, n)
            connected = dealt_within(them, i, n, 30)

            c =
              c
              |> bump("oos_exits")
              |> bump("oos_#{option}")
              |> bump("oos_connected", connected)
              |> bump("oos_pressured", stunned)
              |> bump("oos_pressured_#{option}", stunned)
              |> bump("oos_pressured_connected", stunned and connected)

            {c, false, false}

          true ->
            {c, false, false}
        end
      end)

    counts
  end

  defp bump(c, k, cond? \\ true), do: if(cond?, do: Map.update!(c, k, &(&1 + 1)), else: c)

  # what the player did from frame i (first non-shield frame) on
  defp oos_option(me, i, n) do
    act = fn j -> trunc(elem(me, j).action || 0) end
    a0 = act.(i)
    window = fn from, len -> for j <- from..min(from + len, n - 1)//1, do: act.(j) end
    jc_at = Enum.find(i..min(i + 3, n - 1)//1, fn j -> act.(j) == 24 end)

    cond do
      a0 in 75..91 -> "hit"
      a0 in [233, 234] -> "roll"
      a0 == 235 -> "spotdodge"
      a0 in [212, 214] -> "shield_grab"

      jc_at != nil ->
        after_js = window.(jc_at + 1, 12)

        cond do
          63 in after_js -> "jc_upsmash"
          212 in after_js -> "jc_grab"
          Enum.any?(after_js, &(&1 in [360, 361])) -> "jc_shine"
          wavedash_after?(me, jc_at, n) -> "jc_wavedash"
          Enum.any?(window.(jc_at + 1, 20), &(&1 in 65..69 or &1 == 366)) -> "jc_aerial"
          true -> "jc_jump"
        end

      true -> "drop"
    end
  end

  # jumpsquat at j, then an airdodge run of <= 4 frames into a landing
  defp wavedash_after?(me, j, n) do
    act = fn k -> trunc(elem(me, k).action || 0) end
    start = Enum.find((j + 1)..min(j + 6, n - 1)//1, fn k -> act.(k) == 236 end)

    start != nil and
      Enum.any?(1..4, fn len ->
        k = start + len
        k < n and act.(k) == 43 and Enum.all?(start..(k - 1), &(act.(&1) == 236))
      end)
  end

  # did the opponent's percent rise within `win` frames of i?
  defp dealt_within(them, i, n, win) do
    Enum.any?((i + 1)..min(i + win, n - 1)//1, fn j ->
      (elem(them, j).percent || 0) > (elem(them, j - 1).percent || 0) + 0.01
    end)
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
