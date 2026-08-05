defmodule ExPhil.Interp.AbsorberEntry do
  @moduledoc """
  Detector for the platform-landing absorber entry (INTERP_ROADMAP_V2 W1,
  2026-08-04).

  The mechanism, established on the YS contrastive pair and confirmed at
  the logit level: a mid-cycle shine-jump lands the bot on a low platform
  (grounded at y ≈ 23.5 — off-distribution for FD-only training), the
  jump-cancel X head goes hard-silent in that context (fire 0.0000 over
  ~5,400 platform frames, max logit < 0), and the down+B motif degenerates
  into shine-hold/squat. Patching own-y alone to platform height on healthy
  ground windows reproduces the silence (X mean −0.99 → −6.0) — causal,
  not correlational.

  The DETECTABLE event is the door, not the spell: `entries/2` returns
  platform landings that were followed by basin occupancy — the anchor for
  curation snippets (cycle 4+) and offline absorber counting. Keying on
  entry rather than spell length matters because the absorbed texture
  varies: YS r2 fell into one 405-frame spell, r3 into dozens of sub-120f
  spells; both exceed 50% post-landing basin occupancy.

  Floor-tested per GOTCHA #79 against the fixture pair
  `ys_multishine_{good,absorbed}_2026-08-04.slp`
  (test/exphil/interp/absorber_entry_floor_test.exs).
  """

  @default_platform_y 15.0
  @default_horizon 120
  @default_occupancy 0.5

  @doc """
  Indices of platform landings: airborne→grounded transitions with
  `y > :platform_y` (default #{@default_platform_y}).
  """
  def platform_landings(frames, opts \\ []) do
    platform_y = Keyword.get(opts, :platform_y, @default_platform_y)

    frames
    |> Enum.map(fn f ->
      p = f.game_state.players[1]
      {p.on_ground, p.y}
    end)
    |> Enum.with_index()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [{{g0, _}, _}, {{g1, y1}, _}] ->
      not g0 and g1 and y1 > platform_y
    end)
    |> Enum.map(fn [_, {_, i}] -> i end)
  end

  @doc """
  Absorber entries: platform landings where Squat/SquatWait occupancy over
  the next `:horizon` frames (default #{@default_horizon}) reaches
  `:occupancy` (default #{@default_occupancy}). Returns
  `[%{at: index, occupancy: float}]`.
  """
  def entries(frames, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, @default_horizon)
    min_occ = Keyword.get(opts, :occupancy, @default_occupancy)
    squat = ExPhil.Constants.squat()
    squat_wait = ExPhil.Constants.squat_wait()

    basin? =
      frames
      |> Enum.map(&(&1.game_state.players[1].action in [squat, squat_wait]))
      |> List.to_tuple()

    n = tuple_size(basin?)

    for at <- platform_landings(frames, opts),
        at + 1 < n,
        span = min(horizon, n - at - 1),
        occ = Enum.count((at + 1)..(at + span), &elem(basin?, &1)) / span,
        occ >= min_occ do
      %{at: at, occupancy: occ}
    end
  end
end
