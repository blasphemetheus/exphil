defmodule ExPhil.Eval.TechStatsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.TechStats

  # one frame per action id; own at x=0 facing +x, opponent at x=+40
  defp frames(actions, opts \\ []) do
    percents = Keyword.get(opts, :percents, List.duplicate(0.0, length(actions)))
    stocks = Keyword.get(opts, :stocks, List.duplicate(4, length(actions)))
    facing = Keyword.get(opts, :facing, 1)

    Enum.zip([actions, percents, stocks])
    |> Enum.map(fn {a, pct, st} ->
      %{own: %{action: a, x: 0.0, y: 0.0, percent: pct, stock: st, facing: facing, on_ground: true},
        opp: %{action: 14, x: 40.0, y: 0.0, percent: 0.0, stock: 4, facing: -1, on_ground: true}}
    end)
  end

  defp pad(list, n), do: list ++ List.duplicate(14, n)

  test "tech in place, forward (toward) and backward (away) techs are classed by direction" do
    acts = pad([75, 75, 199, 199, 14, 14, 75, 200, 200, 14, 75, 201, 201, 14], 100)
    s = TechStats.summarize(TechStats.from_game(frames(acts)))
    assert s["tech_rate"] == 1.0
    assert s["tech_in_place_share"] == Float.round(1 / 3, 3)
    assert s["tech_toward_share"] == Float.round(1 / 3, 3)
    assert s["tech_away_share"] == Float.round(1 / 3, 3)
    assert s["tech_punished_share"] == 0.0
  end

  test "a forward roll AWAY from the opponent when facing away from them" do
    acts = pad([75, 200, 200, 14], 100)
    s = TechStats.summarize(TechStats.from_game(frames(acts, facing: -1)))
    assert s["tech_away_share"] == 1.0
  end

  test "missed tech: lying frames, getup option, and what it cost" do
    # miss, lie 10 f, hit on the ground 20 f later for 12%
    acts = pad([75, 183] ++ List.duplicate(184, 10) ++ [185, 185, 76, 76, 14], 100)
    n = length(acts)
    pcts = List.duplicate(0.0, 12) ++ List.duplicate(12.0, n - 12)
    s = TechStats.summarize(TechStats.from_game(frames(acts, percents: pcts)))
    assert s["tech_rate"] == 0.0
    assert s["miss_hit_on_ground_share"] == 1.0
    assert s["miss_lying_frames_mean"] == 10.0
    assert s["miss_punished_share"] == 1.0
    assert s["miss_cost_per"] == 12.0
  end

  test "missed tech followed by a roll toward the opponent, unpunished" do
    acts = pad([75, 191, 192, 192, 196, 196, 14], 100)
    s = TechStats.summarize(TechStats.from_game(frames(acts)))
    assert s["miss_getup_toward_share"] == 1.0
    assert s["miss_punished_share"] == 0.0
    assert s["miss_cost_per"] == 0.0
  end

  test "comboed: three hits inside the gap are one opening; a death inside the kill window closes it" do
    n = 400
    pcts =
      for i <- 0..(n - 1) do
        cond do
          i < 100 -> 0.0
          i < 120 -> 10.0
          i < 140 -> 20.0
          true -> 30.0
        end
      end
    stocks = for i <- 0..(n - 1), do: if(i < 200, do: 4, else: 3)
    s = TechStats.summarize(TechStats.from_game(frames(List.duplicate(14, n), percents: pcts, stocks: stocks)))
    assert s["openings_taken_per_min"] == Float.round(1 / (n / 3600), 3)
    assert s["hits_per_opening_taken"] == 3.0
    assert s["dmg_per_opening_taken"] == 30.0
    assert s["comboed_3plus_share"] == 1.0
    assert s["opening_taken_death_share"] == 1.0
  end

  test "two hits 60 f apart are two openings" do
    n = 400
    pcts = for i <- 0..(n - 1), do: cond do: (i < 100 -> 0.0; i < 160 -> 10.0; true -> 20.0)
    s = TechStats.summarize(TechStats.from_game(frames(List.duplicate(14, n), percents: pcts)))
    assert s["hits_per_opening_taken"] == 1.0
    assert s["openings_taken_per_min"] == Float.round(2 / (n / 3600), 3)
  end

  test "wavedash (from jumpsquat) vs waveland, dash-to-shield, spot dodge, roll" do
    acts =
      pad(
        [20, 20, 24, 24, 236, 236, 43, 43, 14] ++          # wavedash
        [25, 25, 30, 30, 236, 236, 236, 43, 14] ++         # waveland (no jumpsquat)
        [20, 20, 21, 178, 179, 179, 180, 14] ++            # dash -> shield
        [14, 14, 14, 14, 14, 14, 178, 179, 180, 14] ++     # standing shield
        [235, 235, 14, 233, 233, 14, 234, 234, 14],        # spot, roll f, roll b
        100)
    n = length(acts)
    s = TechStats.summarize(TechStats.from_game(frames(acts)))
    per_min = fn k -> Float.round(k / (n / 3600), 3) end
    assert s["wavedashes_per_min"] == per_min.(1)
    assert s["wavelands_per_min"] == per_min.(1)
    assert s["shields_per_min"] == per_min.(2)
    assert s["dash_shields_per_min"] == per_min.(1)
    assert s["spotdodges_per_min"] == per_min.(1)
    assert s["rolls_per_min"] == per_min.(2)
    assert s["shield_time_share"] == Float.round(3 / n, 3)
  end

  test "tech-chase: the opponent's miss we hit within the window" do
    n = 200
    own = for _ <- 1..n, do: %{action: 14, x: 0.0, y: 0.0, percent: 0.0, stock: 4, facing: 1, on_ground: true}
    opp_acts = pad([75, 183, 184, 184, 186, 14], n - 6)
    opp_pct = for i <- 0..(n - 1), do: if(i < 30, do: 0.0, else: 15.0)
    opp = Enum.zip(opp_acts, opp_pct) |> Enum.map(fn {a, p} -> %{action: a, x: 40.0, y: 0.0, percent: p, stock: 4, facing: -1, on_ground: true} end)
    s = TechStats.summarize(TechStats.from_game(Enum.zip(own, opp) |> Enum.map(fn {a, b} -> %{own: a, opp: b} end)))
    assert s["opp_tech_rate"] == 0.0
    assert s["chase_punished_share"] == 1.0
    assert s["chase_cost_per"] == 15.0
  end

  test "out of shield: jump-cancel options, shield grab, roll, drop; pressured subset; connected" do
    sh = [178, 179, 179]
    acts =
      pad(
        sh ++ [24, 24, 63, 63, 63, 14] ++                        # JC upsmash
        sh ++ [24, 24, 212, 213, 14] ++                           # JC grab
        sh ++ [24, 360, 361, 361, 14] ++                          # JC shine (Fox)
        sh ++ [24, 24, 236, 236, 236, 43, 43, 14] ++              # JC wavedash
        sh ++ [24, 24, 25, 25, 25, 67, 67, 70, 14] ++             # jump -> bair
        sh ++ [24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 14] ++ # plain jump (no aerial in 20 f)
        sh ++ [212, 213, 14] ++                                   # shield grab
        [178, 181, 181, 179, 233, 233, 14] ++                     # pressured, rolled
        sh ++ [180, 14, 14] ++                                    # drop
        [178, 181, 76, 76, 14],                                   # pressured, hit out of shield
        100)
    n = length(acts)
    # the JC upsmash connects: opponent percent rises 5 f after the exit (frame 3)
    opp_pct = for i <- 0..(n - 1), do: if(i >= 8, do: 10.0, else: 0.0)
    fr = frames(acts) |> Enum.with_index() |> Enum.map(fn {f, i} -> put_in(f, [:opp, :percent], Enum.at(opp_pct, i)) end)
    s = TechStats.summarize(TechStats.from_game(fr))
    assert s["oos_exits_per_min"] == Float.round(10 / (n / 3600), 3)
    for o <- ~w(jc_upsmash jc_grab jc_shine jc_wavedash jc_aerial jc_jump shield_grab roll drop hit) do
      assert s["oos_#{o}_share"] == 0.1, "#{o}"
    end
    assert s["oos_jc_share"] == 0.6
    assert s["oos_pressured_share"] == 0.2
    assert s["oos_pressured_roll_share"] == 0.5
    assert s["oos_pressured_hit_share"] == 0.5
    assert s["oos_connected_share"] == 0.1
    assert s["oos_pressured_connected_share"] == 0.0
  end

  test "merge adds counts and an empty game is harmless" do
    a = TechStats.from_game(frames(pad([75, 199, 14], 50)))
    b = TechStats.from_game(frames(pad([75, 183, 184, 186, 14], 50)))
    s = TechStats.summarize(TechStats.merge(a, b))
    assert s["tech_rate"] == 0.5
    assert TechStats.from_game([]) == %{}
  end
end
