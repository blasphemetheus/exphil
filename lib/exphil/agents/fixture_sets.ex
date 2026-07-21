defmodule ExPhil.Agents.FixtureSets do
  @moduledoc """
  SINGLE SOURCE for expert fixture lists.

  The mewtwo_combo list was previously hardcoded in BOTH
  scripts/dagger_drill.exs and scripts/teacher_coverage.exs — a silent
  drift risk called out in the 2026-07-17 r13-prep recon: adding a fixture
  to the trainer but not the coverage audit makes the audit lie about the
  teacher it claims to measure. Add new fixtures HERE only.
  """

  @mewtwo_combo [
    "test/fixtures/replays/mewtwo_fair_chains.slp",
    "test/fixtures/replays/mewtwo_shfair_only.slp",
    "test/fixtures/replays/mewtwo_approach_fair.slp",
    "test/fixtures/replays/mewtwo_turnaround_fair.slp",
    "test/fixtures/replays/mewtwo_oos_chains.slp",
    "test/fixtures/replays/mewtwo_ground_neutral.slp",
    # Recorded 2026-07-18 (task #22): behind-response chains (18 behind
    # episodes both facings + 13 organic tech-chase entries; targets the
    # opponent_behind scenario regression) and FH-no-DJ air control
    # (226 full-hop stints, 3.1% DJ at p50=31f; targets instant-DJ timing)
    "test/fixtures/replays/mewtwo_behind_response.slp",
    "test/fixtures/replays/mewtwo_fh_air_control.slp"
  ]

  @doc "Fixture paths for the mewtwo_combo composite expert."
  @spec mewtwo_combo() :: [Path.t()]
  def mewtwo_combo, do: @mewtwo_combo

  @doc "Comma-joined form for CLI --fixtures style consumers."
  @spec mewtwo_combo_csv() :: String.t()
  def mewtwo_combo_csv, do: Enum.join(@mewtwo_combo, ",")

  # Recorded 2026-07-20 on mainline beta.19 (post-demo neutral push):
  # NOT in @mewtwo_combo yet — dtilt/uptilt are outside the fair expert's
  # vocabulary (feeding them to its table without an expert extension
  # would relabel them as fair frames). Wire into the combo list when the
  # dtilt/uptilt expert branch exists; ground_neutral_v2 can join sooner.
  @mewtwo_neutral_extras [
    # ~2min dense: max-range tail dtilts + uptilts, both facings
    "test/fixtures/replays/mewtwo_dtilt_uptilt_dense.slp",
    # ~3min varied: approach/hold/retreat texture, dtilt at spacing,
    # uptilt anti-air, wavedash-back dtilt
    "test/fixtures/replays/mewtwo_ground_neutral_v2.slp"
  ]

  # ~3min offstage recovery menu: DJ->teleport ledge/stage, low/high,
  # both sides, some deliberately bad (low+far) starts. For the recovery
  # taxonomy (#32), StyleCard recovery gates, and the alpha=1.0 DJ check.
  @mewtwo_recovery ["test/fixtures/replays/mewtwo_recovery_mix.slp"]

  # G&W starter pack (task #23 transfer test), Bradley's option list:
  # neutral_dense = SH fair/dair, grab, dtilt, wavedash dtilt, run-up
  # shield -> dtilt/grab, up-B OOS; movement_ledge = movement + ledge
  # regrab patterns.
  @gnw_starter [
    "test/fixtures/replays/gnw_neutral_dense.slp",
    "test/fixtures/replays/gnw_movement_ledge.slp"
  ]

  @doc "Mewtwo dtilt/uptilt + neutral-texture fixtures (pending expert branch)."
  @spec mewtwo_neutral_extras() :: [Path.t()]
  def mewtwo_neutral_extras, do: @mewtwo_neutral_extras

  @doc "Mewtwo offstage-recovery fixture."
  @spec mewtwo_recovery() :: [Path.t()]
  def mewtwo_recovery, do: @mewtwo_recovery

  @doc "Game & Watch starter fixtures (transfer test, task #23)."
  @spec gnw_starter() :: [Path.t()]
  def gnw_starter, do: @gnw_starter
end
