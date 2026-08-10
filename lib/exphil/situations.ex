defmodule ExPhil.Situations do
  @moduledoc """
  Per-frame situation labels — the shared-rules module of the labeling
  program (docs/planning/SITUATION_LABELS.md, winnowed 2026-08-10; coach
  roadmap #1).

  A situation label names the strategic context a frame lives in,
  derived mechanically from game state. One rule set, consumed two ways:

  - **Batch**: `label_states/3` maps parsed game states to bitmasks
    (u64 per frame — store beside MmapCorpus labels, filter cheaply).
  - **Live**: thread `fold/3` over incoming states like
    `ExPhil.Eval.LiveScorer` does; the context carries the little
    history the windowed labels need.

  Keeping both frontends on `fold/3` is the anti-drift guarantee: live
  and post-hoc labels can never disagree on the same frames.

  ## Bit registry

  `labels/0` is the ordered registry; a label's bit is its index.
  **APPEND-ONLY**: stored bitmasks outlive code, so new labels go at
  the end and dead labels keep their slot (rename to `:retired_N`).

  ## Umbrella invariant

  `:neutral`, `:advantage`, `:disadvantage` are mutually exclusive
  parents (a trade — both players in hitstun — gets none). Child labels
  (e.g. `:in_hitstun`) are computed unconditionally; only the parent
  bits enforce exclusivity.

  ## Conventions

  All labels are EGO-CENTRIC to `own_port` (`:juggle` = we juggle them;
  `:being_juggled` = the mirror). Thresholds live in the constants
  block below — tune there, never inline. Unknown stages fall back to
  FD geometry (same convention as the per-stage-ledge embedding work).
  """

  alias ExPhil.Bridge.GameState

  # ==========================================================================
  # Registry (APPEND-ONLY — bit = index)
  # ==========================================================================

  @labels [
    # -- stage geometry
    :onstage_center,
    :onstage_corner,
    :offstage,
    :above_stage,
    :on_platform,
    :below_ledge,
    :ledge_hang,
    :ledge_occupied_by_opp,
    :near_blastzone,
    :platform_underneath,
    # -- umbrella
    :neutral,
    :advantage,
    :disadvantage,
    # -- neutral children
    :approach,
    :retreat,
    # -- advantage children
    :juggle,
    :tech_chase,
    :ledge_trap,
    :shield_pressure_ours,
    :shield_break_confirm,
    :pummel_throw_decision,
    :edgeguard,
    :conversion_open,
    :combo_active,
    # -- disadvantage children
    :in_hitstun,
    :tumble,
    :being_juggled,
    :being_tech_chased,
    :being_edgeguarded,
    :recovery_low,
    :recovery_high,
    :resource_exhausted,
    :cornered,
    :shield_pressure_theirs,
    :shield_low,
    # -- ledge micro
    :edge_danger,
    :ledge_option_pending,
    :both_offstage,
    # -- game flow
    :respawn_invincible,
    :post_kill_neutral,
    :last_stock_ours,
    :last_stock_theirs,
    :percent_lead,
    :percent_deficit,
    :timeout_relevant,
    # -- execution windows
    :jc_window,
    :shine_cancellable,
    # -- hygiene
    :warmup_frames,
    # -- appended 2026-08-11 (registry is append-only): wall interaction
    # zones from extracted stage collision (ExPhil.StageCollision)
    :walljump_zone,
    :walltech_available
  ]

  @bit Map.new(Enum.with_index(@labels))

  # ==========================================================================
  # Thresholds & action sets
  # ==========================================================================

  # Geometry
  @center_frac 0.4
  @corner_margin 20.0
  @edge_danger_margin 20.0
  @above_stage_y 30.0
  @platform_y 5.0
  @below_ledge_y -15.0
  @offstage_y -10.0
  @blastzone_margin 40.0
  @fd_edge 85.5656967163

  # Interaction ranges / windows
  @juggle_dy 15.0
  @juggle_dx 40.0
  @chase_range 50.0
  @shield_pressure_range 30.0
  @ledge_trap_range 40.0
  @corner_opp_range 60.0
  @approach_window 10
  @approach_delta 5.0
  @combo_max_gap 20
  @conversion_window 90
  @post_kill_window 300
  @warmup 120
  @timeout_frame 27_000
  @shield_low 20.0
  @percent_gap 40.0
  # Wall-zone reach (units to the nearest wall segment): walljump needs
  # near-contact; walltech has the tech window's leniency
  @walljump_dist 8.0
  @walltech_dist 12.0

  # Action states (libmelee enum)
  @rebirth [12, 13]
  @first_actionable 14
  @dash_run [0x14, 0x15]
  @tumble 38
  @hitstun_actions MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @shield_states MapSet.new([178, 179, 180])
  @shield_break MapSet.new(205..211)
  # Knockdown lifecycle: bounce 183/191, lying 184/192, getups 186-189 /
  # 194-197, techs 199-201 (same sets as MewtwoTechChaseExpert)
  @knockdown MapSet.new(
               [183, 184, 191, 192] ++
                 Enum.to_list(186..189) ++ Enum.to_list(194..197) ++ Enum.to_list(199..201)
             )
  @grab_hold MapSet.new(212..218)
  @cliff [252, 253]
  @cliff_wait 253
  @jumpsquat 24
  @reflector_ground MapSet.new(360..363)

  # ==========================================================================
  # Public API
  # ==========================================================================

  @doc "Ordered label registry (bit = index)."
  @spec labels() :: [atom()]
  def labels, do: @labels

  @doc "Bit index of a label."
  @spec bit(atom()) :: non_neg_integer()
  def bit(label), do: Map.fetch!(@bit, label)

  @doc "Encode a label set as a u64 bitmask."
  @spec to_mask(Enumerable.t()) :: non_neg_integer()
  def to_mask(set), do: Enum.reduce(set, 0, fn l, acc -> Bitwise.bor(acc, Bitwise.bsl(1, bit(l))) end)

  @doc "Decode a bitmask back to a label MapSet."
  @spec from_mask(non_neg_integer()) :: MapSet.t()
  def from_mask(mask) do
    @labels
    |> Enum.with_index()
    |> Enum.filter(fn {_l, i} -> Bitwise.band(mask, Bitwise.bsl(1, i)) != 0 end)
    |> MapSet.new(fn {l, _i} -> l end)
  end

  @doc "Fresh fold context. Reset at every game boundary."
  @spec new_context() :: map()
  def new_context do
    %{
      dists: :queue.new(),
      n_dists: 0,
      combo_gap: nil,
      conv_until: nil,
      opp_stock: nil,
      opp_killed_at: nil,
      opp_hitstun_prev: false
    }
  end

  @doc """
  Label one state. Returns `{ctx, MapSet}` — thread the ctx across
  consecutive frames of ONE game (windowed labels read it); pass
  `new_context/0` at game start.
  """
  @spec fold(map(), GameState.t(), integer()) :: {map(), MapSet.t()}
  def fold(ctx, %GameState{} = gs, own_port) do
    opp_port = if own_port == 1, do: 2, else: 1
    own = GameState.get_player(gs, own_port)
    opp = GameState.get_player(gs, opp_port)

    if own == nil or opp == nil do
      {ctx, MapSet.new()}
    else
      geo = geometry(gs.stage)
      frame = gs.frame || 0
      dist = distance(own, opp)

      own_disadv = disadvantaged?(own)
      opp_disadv = disadvantaged?(opp)
      opp_hitstun = hitstun?(opp)

      # ---- context updates (before labels that read them) ----
      {dists, n} = push_window(ctx.dists, ctx.n_dists, dist, @approach_window)

      combo_gap =
        cond do
          opp_hitstun and not hitstun?(own) -> 0
          ctx.combo_gap != nil and ctx.combo_gap < @combo_max_gap -> ctx.combo_gap + 1
          true -> nil
        end

      was_neutral = not ctx.opp_hitstun_prev and ctx.combo_gap == nil

      conv_until =
        cond do
          opp_hitstun and not ctx.opp_hitstun_prev and was_neutral -> frame + @conversion_window
          ctx.conv_until != nil and frame <= ctx.conv_until -> ctx.conv_until
          true -> nil
        end

      opp_killed_at =
        cond do
          ctx.opp_stock != nil and (opp.stock || 0) < ctx.opp_stock -> frame
          true -> ctx.opp_killed_at
        end

      ctx = %{
        ctx
        | dists: dists,
          n_dists: n,
          combo_gap: combo_gap,
          conv_until: conv_until,
          opp_stock: opp.stock,
          opp_killed_at: opp_killed_at,
          opp_hitstun_prev: opp_hitstun
      }

      # ---- geometry ----
      own_off = offstage?(own, geo)
      opp_off = offstage?(opp, geo)

      set =
        []
        |> put(:onstage_center, own.on_ground and abs(x(own)) < @center_frac * geo.edge)
        |> put(:onstage_corner, own.on_ground and abs(x(own)) > geo.edge - @corner_margin)
        |> put(:offstage, own_off)
        |> put(:above_stage, not own.on_ground and abs(x(own)) <= geo.edge and y(own) > @above_stage_y)
        |> put(:on_platform, own.on_ground and y(own) > @platform_y)
        |> put(:below_ledge, own_off and y(own) < @below_ledge_y)
        |> put(:ledge_hang, act(own) in @cliff)
        |> put(:ledge_occupied_by_opp, act(opp) in @cliff)
        |> put(:near_blastzone, near_blastzone?(own, geo))
        |> put(:platform_underneath, not own.on_ground and platform_underneath?(own, geo))

      # ---- umbrella (mutually exclusive) ----
      advantage = opp_disadv and not own_disadv
      disadvantage = own_disadv and not opp_disadv
      neutral = not own_disadv and not opp_disadv and act(own) >= @first_actionable and act(opp) >= @first_actionable

      set =
        set
        |> put(:neutral, neutral)
        |> put(:advantage, advantage)
        |> put(:disadvantage, disadvantage)

      # ---- neutral children ----
      {d_then, have_window} = peek_window(ctx.dists, ctx.n_dists, @approach_window)

      set =
        set
        |> put(:approach, neutral and have_window and dist < d_then - @approach_delta)
        |> put(:retreat, neutral and have_window and dist > d_then + @approach_delta)

      # ---- advantage children ----
      set =
        set
        |> put(:juggle, not opp.on_ground and juggled?(opp, own))
        |> put(:tech_chase, MapSet.member?(@knockdown, act(opp)) and own.on_ground and dist < @chase_range)
        |> put(
          :ledge_trap,
          act(opp) in @cliff and own.on_ground and
            abs(x(own)) > geo.edge - @ledge_trap_range and sign(x(own)) == sign(x(opp))
        )
        |> put(:shield_pressure_ours, MapSet.member?(@shield_states, act(opp)) and dist < @shield_pressure_range)
        |> put(:shield_break_confirm, MapSet.member?(@shield_break, act(opp)))
        |> put(:pummel_throw_decision, MapSet.member?(@grab_hold, act(own)))
        |> put(:edgeguard, opp_off and not own_off and act(opp) not in @cliff)
        |> put(:conversion_open, ctx.conv_until != nil)
        |> put(:combo_active, ctx.combo_gap != nil)

      # ---- disadvantage children ----
      set =
        set
        |> put(:in_hitstun, hitstun?(own))
        |> put(:tumble, act(own) == @tumble)
        |> put(:being_juggled, not own.on_ground and juggled?(own, opp))
        |> put(:being_tech_chased, MapSet.member?(@knockdown, act(own)) and opp.on_ground and dist < @chase_range)
        |> put(:being_edgeguarded, own_off and not opp_off and act(own) not in @cliff)
        |> put(:recovery_low, own_off and y(own) < 0.0)
        |> put(:recovery_high, own_off and y(own) >= 0.0)
        |> put(:resource_exhausted, own_off and (own.jumps_left || 0) == 0)
        |> put(
          :cornered,
          own.on_ground and abs(x(own)) > geo.edge - @corner_margin - @platform_y and
            sign(x(opp)) == sign(x(own)) and abs(x(opp)) < abs(x(own)) and dist < @corner_opp_range
        )
        |> put(:shield_pressure_theirs, MapSet.member?(@shield_states, act(own)) and dist < @shield_pressure_range)
        |> put(:shield_low, (own.shield_strength || 60.0) < @shield_low)

      # ---- ledge micro ----
      set =
        set
        |> put(:edge_danger, edge_danger?(own, geo))
        |> put(:ledge_option_pending, act(own) == @cliff_wait)
        |> put(:both_offstage, own_off and opp_off)

      # ---- game flow ----
      set =
        set
        |> put(:respawn_invincible, act(own) in @rebirth or act(opp) in @rebirth)
        |> put(:post_kill_neutral, ctx.opp_killed_at != nil and frame - ctx.opp_killed_at < @post_kill_window)
        |> put(:last_stock_ours, (own.stock || 0) == 1)
        |> put(:last_stock_theirs, (opp.stock || 0) == 1)
        |> put(:percent_lead, own.stock == opp.stock and pct(opp) - pct(own) > @percent_gap)
        |> put(:percent_deficit, own.stock == opp.stock and pct(own) - pct(opp) > @percent_gap)
        |> put(:timeout_relevant, frame > @timeout_frame)

      # ---- execution / hygiene ----
      set =
        set
        |> put(:jc_window, act(own) == @jumpsquat)
        |> put(:shine_cancellable, MapSet.member?(@reflector_ground, act(own)))
        |> put(:warmup_frames, frame < @warmup)

      # ---- wall zones (extracted collision; false when no data) ----
      # Guarded: wall_distance only runs when the state could use a wall
      wall_dist =
        if not own.on_ground or hitstun?(own),
          do: ExPhil.StageCollision.wall_distance(gs.stage, x(own), y(own))

      set =
        set
        |> put(:walljump_zone, wall_dist != nil and not own.on_ground and wall_dist < @walljump_dist)
        |> put(:walltech_available, wall_dist != nil and hitstun?(own) and wall_dist < @walltech_dist)

      {ctx, MapSet.new(set)}
    end
  end

  @doc """
  Batch frontend: label a list of consecutive game states (one game).
  Returns bitmasks in order. `opts`: `:as` — `:mask` (default) or `:set`.
  """
  @spec label_states([GameState.t()], integer(), keyword()) :: [non_neg_integer() | MapSet.t()]
  def label_states(game_states, own_port, opts \\ []) do
    as = Keyword.get(opts, :as, :mask)

    {_ctx, out} =
      Enum.reduce(game_states, {new_context(), []}, fn gs, {ctx, acc} ->
        {ctx, set} = fold(ctx, gs, own_port)
        {ctx, [if(as == :set, do: set, else: to_mask(set)) | acc]}
      end)

    Enum.reverse(out)
  end

  # ==========================================================================
  # Rules helpers
  # ==========================================================================

  defp put(list, label, true), do: [label | list]
  defp put(list, _label, _), do: list

  defp act(p), do: trunc(p.action || 0)
  defp x(p), do: p.x || 0.0
  defp y(p), do: p.y || 0.0
  defp pct(p), do: p.percent || 0.0
  defp sign(v), do: if(v >= 0, do: 1, else: -1)

  defp distance(a, b) do
    dx = x(a) - x(b)
    dy = y(a) - y(b)
    :math.sqrt(dx * dx + dy * dy)
  end

  defp hitstun?(p) do
    (p.hitstun_frames_left || 0) > 0 or MapSet.member?(@hitstun_actions, act(p))
  end

  defp disadvantaged?(p) do
    hitstun?(p) or act(p) == @tumble or MapSet.member?(@knockdown, act(p))
  end

  defp offstage?(p, geo), do: abs(x(p)) > geo.edge or (not p.on_ground and y(p) < @offstage_y)

  defp juggled?(victim, over) do
    (hitstun?(victim) or act(victim) == @tumble) and
      y(victim) > y(over) + @juggle_dy and abs(x(victim) - x(over)) < @juggle_dx
  end

  defp edge_danger?(p, geo) do
    toward_edge =
      case p.speed_ground_x_self do
        s when is_number(s) and s != 0.0 -> s * x(p) > 0.0
        _ -> (p.facing || 1) * x(p) > 0.0
      end

    p.on_ground and act(p) in @dash_run and toward_edge and
      abs(x(p)) > geo.edge - @edge_danger_margin
  end

  defp near_blastzone?(p, geo) do
    case geo.blast do
      nil ->
        false

      {left, right, top, bottom} ->
        x(p) < left + @blastzone_margin or x(p) > right - @blastzone_margin or
          y(p) > top - @blastzone_margin or y(p) < bottom + @blastzone_margin
    end
  end

  defp platform_underneath?(p, geo) do
    Enum.any?(geo.platforms, fn {h, l, r} ->
      is_number(h) and is_number(l) and is_number(r) and
        x(p) >= l and x(p) <= r and h < y(p)
    end)
  end

  # Rolling distance window (:queue of the last `max`+1 samples): peek
  # returns the sample ~max frames ago for the approach/retreat delta
  defp push_window(q, n, value, max) do
    q = :queue.in(value, q)

    if n >= max + 1 do
      {_, q} = :queue.out(q)
      {q, n}
    else
      {q, n + 1}
    end
  end

  defp peek_window(q, n, max) do
    case :queue.peek(q) do
      {:value, oldest} -> {oldest, n >= max + 1}
      :empty -> {0.0, false}
    end
  end

  # Per-stage geometry, cached per stage id (labels run per-frame — do
  # not hit Melee.Stages 60x/sec)
  @doc """
  Stage geometry used by the rules (edge x, platforms, blastzones), by
  EXTERNAL stage id. Public for consumers that render or export the
  same geometry the labels were computed from (Inspect.export_session).
  """
  @spec geometry(number() | nil) :: %{edge: float(), blast: tuple() | nil, platforms: list()}
  def geometry(stage) do
    id = trunc(stage || 0)

    case :persistent_term.get({__MODULE__, :geo, id}, nil) do
      nil ->
        geo = build_geometry(id)
        :persistent_term.put({__MODULE__, :geo, id}, geo)
        geo

      geo ->
        geo
    end
  end

  defp build_geometry(id) do
    # game_state.stage carries EXTERNAL (Slippi) ids; Melee.Stages keys
    # on stage ATOMS / internal ids. Convert first — external YS (8)
    # collides with internal FoD (8), so passing the raw id silently
    # uses the wrong stage's geometry (found 2026-08-11).
    stage = Melee.Enums.Stage.from_external(id)

    %{
      edge: Melee.Stages.edge_ground_position(stage) || @fd_edge,
      blast: Melee.Stages.blastzones(stage),
      platforms:
        [
          Melee.Stages.top_platform_position(stage),
          Melee.Stages.side_platform_position(true, stage),
          Melee.Stages.side_platform_position(false, stage)
        ]
        |> Enum.reject(&match?({nil, _, _}, &1))
    }
  end
end
