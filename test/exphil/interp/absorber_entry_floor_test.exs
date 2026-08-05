defmodule ExPhil.Interp.AbsorberEntryFloorTest do
  @moduledoc """
  GOTCHA #79 floor test for the absorber-entry detector (W1, 2026-08-04).

  The fixture pair is the YS stochastic contrast itself: same checkpoint
  (ms_g4_d2mix), same stage, same stand dummy —
  `ys_multishine_absorbed_2026-08-04.slp` collapsed (39.9/min, chain 1,
  51.9% squat, one 405-frame spell) and `ys_multishine_good_2026-08-04.slp`
  played (285.6/min, chain 236, 2.5% squat, 42 platform frames total).
  Both were watched during the 08-04 stage sweep.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Interp.AbsorberEntry
  alias ExPhil.Test.MetricFloor

  @absorbed "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"
  @good "test/fixtures/replays/ys_multishine_good_2026-08-04.slp"

  # The entry's real frame shape (write it down FIRST — #79): mid-cycle
  # shine-jump rises ~29f, grounded landing at y≈23.5 (YS side platform),
  # then Squat/SquatWait dominates — as ONE long spell (r2: 405f) or MANY
  # short ones (r3: dozens <120f), so the detector keys on post-landing
  # basin OCCUPANCY over a horizon, never on spell length.
  @shape "airborne→grounded at y>15, then ≥50% Squat/SquatWait occupancy over the next 120f"

  test "detects entries in the absorbed replay (the floor)" do
    frames = MetricFloor.frames!(@absorbed)

    MetricFloor.assert_floor(length(AbsorberEntry.entries(frames)), 1,
      metric: "AbsorberEntry.entries count",
      subject: "the absorbed YS replay (watched: 51.9% squat, chain 1)",
      shape: @shape
    )
  end

  test "discriminates absorbed from good" do
    absorbed = length(AbsorberEntry.entries(MetricFloor.frames!(@absorbed)))
    good = length(AbsorberEntry.entries(MetricFloor.frames!(@good)))

    MetricFloor.assert_discriminates(absorbed, good,
      metric: "AbsorberEntry.entries count",
      min_ratio: 2.0,
      absolute: true
    )
  end

  test "the good run's platform touch does not count as an entry" do
    # r1 landed on a platform briefly (42 plat frames) and left — a
    # detector that fires on mere platform CONTACT would anchor snippets
    # on healthy escapes and dilute the curation signal.
    frames = MetricFloor.frames!(@good)
    landings = AbsorberEntry.platform_landings(frames)
    entries = AbsorberEntry.entries(frames)

    assert entries == [],
           "good run produced #{length(entries)} entries from #{length(landings)} landings — " <>
             "the occupancy gate should reject healthy platform touches"
  end
end
