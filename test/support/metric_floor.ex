defmodule ExPhil.Test.MetricFloor do
  @moduledoc """
  Test template for behavioral metrics: **a metric must score a KNOWN-GOOD
  execution highly.**

  This exists because of GOTCHAS #79. `ShineChain` v2 was tuned to reject a
  sloppy fixture and succeeded — while also scoring every *correct*
  multishine as "max chain 1", because it required grounded reflectors and a
  real multishine is ~1/3 grounded by construction. The investigation then
  spent two days hunting a technique Melee does not have (#78).

  The failure mode is specific and worth naming: **a metric validated only
  against known-BAD data can be maximally wrong.** It correctly rejects the
  bad thing while also rejecting the good thing, and nothing in the test
  suite notices, because every assertion it has is of the form "the bad
  input scores low" — which a constant `0` also satisfies.

  So every metric in this repo should own two assertions:

    1. `assert_floor/3` — a known-good execution scores at or above a floor.
       This is the one that was missing, and it is the one that matters.
    2. `assert_discriminates/3` — good scores meaningfully above bad. Cheap
       insurance against the opposite failure (a metric that says "great"
       about everything, e.g. one that returns a constant).

  ## Usage

      alias ExPhil.Test.MetricFloor

      test "scores a real multishine highly" do
        frames = MetricFloor.frames!("test/fixtures/replays/fox_multishine_closed.slp")
        actions = Enum.map(frames, & &1.game_state.players[1].action)

        MetricFloor.assert_floor(
          Enum.max(ShineChain.chains(actions), fn -> 0 end),
          50,
          metric: "ShineChain.chains max",
          subject: "the canonical multishine fixture",
          shape: "~9-frame cycle: 2 grounded reflector frames + 4 airborne"
        )
      end

  The `:shape` option is deliberately required-by-convention rather than by
  the compiler: #79's actual instruction is *"first write down the
  technique's actual frame-by-frame shape"*, and writing it into the test is
  how the next person checks the metric against reality instead of against
  the metric's own assumptions.

  ## When you are ADDING a metric (chaingrab, edgeguard, wavedash, ...)

  Write the floor test FIRST, against a recording you have watched and
  believe is correct, and watch it FAIL against the not-yet-written metric.
  A floor test written after the metric tends to be tuned to whatever the
  metric already does, which is how #79 happened in the first place.
  """

  import ExUnit.Assertions

  alias ExPhil.Data.Peppi

  @doc """
  Parse a replay fixture into training frames (positive frames only, ports
  1/2 — the convention every other loader in the repo uses).

  Raises with a pointed message when the fixture is missing or unparseable,
  because a metric test that silently skips is worse than no test.
  """
  @spec frames!(String.t(), keyword()) :: [map()]
  def frames!(path, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)
    opponent_port = Keyword.get(opts, :opponent_port, if(player_port == 1, do: 2, else: 1))
    expanded = Path.expand(path)

    unless File.exists?(expanded) do
      flunk("""
      Metric floor test cannot run: fixture missing at #{expanded}

      A floor test needs a recording you have WATCHED and believe is a
      correct execution. Record one (scripts/record_multishine.exs and its
      descendants) rather than weakening the test.
      """)
    end

    case Peppi.parse(expanded) do
      {:ok, replay} ->
        replay
        |> Peppi.to_training_frames(player_port: player_port, opponent_port: opponent_port)
        |> Enum.reject(&(&1.game_state.frame < 0))

      {:error, reason} ->
        flunk("Metric floor test: fixture #{path} failed to parse (#{inspect(reason)})")
    end
  end

  @doc """
  Assert a known-good execution scores at or above `floor`.

  Options (all strings, all for the failure message — this is a
  documentation-carrying assertion on purpose):

    * `:metric` — what was measured, e.g. `"ShineChain.chains max"`
    * `:subject` — what it was measured on, e.g. `"the canonical fixture"`
    * `:shape` — the technique's actual frame-by-frame shape (see moduledoc)
  """
  @spec assert_floor(number(), number(), keyword()) :: true
  def assert_floor(score, floor, opts \\ []) do
    metric = Keyword.get(opts, :metric, "metric")
    subject = Keyword.get(opts, :subject, "a known-good execution")
    shape = Keyword.get(opts, :shape)

    if score < floor do
      flunk("""
      #{metric} scored #{inspect(score)} on #{subject} — below the floor of #{inspect(floor)}.

      This is the GOTCHAS #79 failure: the metric may be forbidding the
      technique's own mechanics. Before "fixing" the fixture or lowering the
      floor, check the metric against the technique's real frame shape.
      #{if shape, do: "\nDocumented shape: #{shape}", else: ""}

      A metric validated only against known-BAD data can be maximally wrong:
      it rejects the bad thing AND the good thing.
      """)
    end

    true
  end

  @doc """
  Assert the metric separates good from bad by at least `min_ratio`
  (default 2.0). Guards the opposite failure from `assert_floor/3`: a
  metric that scores everything highly (a constant, or one measuring
  something incidental).

  Pass `absolute: true` when the metric's scale makes a ratio meaningless
  (e.g. scores that can legitimately be 0) — then `min_ratio` is read as a
  minimum absolute difference instead.
  """
  @spec assert_discriminates(number(), number(), keyword()) :: true
  def assert_discriminates(good_score, bad_score, opts \\ []) do
    min_ratio = Keyword.get(opts, :min_ratio, 2.0)
    absolute? = Keyword.get(opts, :absolute, false)
    metric = Keyword.get(opts, :metric, "metric")

    separated? =
      if absolute? do
        good_score - bad_score >= min_ratio
      else
        bad_score == 0 or good_score / max(bad_score, 1.0e-9) >= min_ratio
      end

    unless separated? do
      flunk("""
      #{metric} does not separate good from bad:
        known-good: #{inspect(good_score)}
        known-bad:  #{inspect(bad_score)}
        required:   #{if absolute?, do: "difference >= #{min_ratio}", else: "ratio >= #{min_ratio}x"}

      A metric that scores everything alike measures something incidental to
      the technique (or is constant). Check what it actually keys on.
      """)
    end

    true
  end
end
