defmodule ExPhil.Options do
  @moduledoc """
  Named-option recognition: controller/action-state streams -> discrete,
  human-readable option EVENTS (coach roadmap #4a).

  Where `ExPhil.Situations` answers "what context is this frame in",
  this answers "what did the player just DO" — the vocabulary the
  per-situation corpus statistics aggregate over ("in edge_danger,
  strong players choose: dash_back 40%, wavedash_back 25%, ...") and
  the language the coach speaks.

  Recognition is a fold over consecutive frames keyed on action-STATE
  TRANSITIONS (an option fires once, on the frame its state is entered
  — not on every frame it persists). Attack categories reuse
  `ExPhil.Eval.NeutralScan.category/1`; multishine chains remain
  `ExPhil.Eval.ShineChain`'s business (integrate downstream, don't
  duplicate).

  ## v1 vocabulary

  Movement: `:dash`, `:dashdance` (>=3 flips), `:wavedash` (jumpsquat ->
  airdodge -> grounded within the wavedash window), `:waveland`
  (airdodge -> grounded, no recent jumpsquat), `:double_jump`,
  `:airdodge`, `:spotdodge`, `:roll_forward`, `:roll_backward`,
  `:shield_on`.

  Attacks (from NeutralScan taxonomy, fired on entry): `:jab`,
  `:dash_attack`, `:tilt`, `:smash`, `:aerial`, `:special`
  (shine lands here for spacies; meta carries the raw action id).

  Grab game: `:grab`, `:throw` (meta: direction).

  Defense/tech: `:tech_in_place`, `:tech_roll`, `:missed_tech`,
  `:getup_attack`, `:getup_stand`.

  Ledge (from CliffWait, the LedgeExpert option table): `:ledge_getup`,
  `:ledge_attack`, `:ledge_roll`, `:ledge_jump`.

  Deliberately deferred: `:lcancel` (needs landing-lag measurement),
  `:ledgedash`, `:shorthop`/`:fullhop` split (needs jump-press frames or
  apex height), out-of-shield attribution.
  """

  alias ExPhil.Eval.NeutralScan

  @dashing 0x14
  @jumpsquat 24
  @double_jump 25
  @tumble_bounce [183, 191]
  @tech_in_place 199
  @tech_roll [200, 201]
  @getup_stand [186, 194]
  @getup_attack [187, 195]
  @roll_f 233
  @roll_b 234
  @spotdodge 235
  @airdodge 236
  @shield_on 178
  @grab [212, 214]
  @throws %{219 => :forward, 220 => :back, 221 => :up, 222 => :down}
  @cliff_wait 253
  @ledge_options %{
    254 => :ledge_getup,
    255 => :ledge_getup,
    256 => :ledge_attack,
    257 => :ledge_attack,
    258 => :ledge_roll,
    259 => :ledge_roll,
    260 => :ledge_jump,
    261 => :ledge_jump,
    262 => :ledge_jump,
    263 => :ledge_jump
  }

  @wavedash_window 10
  @dashdance_window 24
  @dashdance_min_flips 3

  @type event :: %{index: non_neg_integer(), option: atom(), meta: map()}

  @doc """
  Recognize option events for `port`'s player over a list of game
  states (one game, consecutive frames). Returns events ordered by
  index. Each event fires once, at the frame its option is initiated.
  """
  @spec events([map()], integer()) :: [event()]
  def events(game_states, port) do
    players = Enum.map(game_states, &(&1.players[port]))

    {events, _} =
      players
      |> Enum.with_index()
      |> Enum.reduce({[], new_ctx()}, fn {p, i}, {acc, ctx} ->
        # Virtual "nothing" prev on the first frame so entries at index 0
        # still fire (a game's opening state is an entry, not a hold)
        prev = ctx.prev || %{action: -1, on_ground: nil, facing: nil}

        if p == nil do
          {acc, %{ctx | prev: p}}
        else
          a = act(p)
          pa = act(prev)
          entered = a != pa

          ctx = track(ctx, p, prev, i)

          new =
            []
            |> maybe(entered and a == @dashing, :dash, %{direction: dir(p)}, i)
            |> maybe(dashdance?(ctx, i), :dashdance, %{flips: @dashdance_min_flips}, i)
            |> maybe(entered and a == @double_jump, :double_jump, %{}, i)
            |> maybe(entered and a == @airdodge and not wavedash_candidate?(ctx, i), :airdodge, %{}, i)
            |> maybe(entered and a == @spotdodge, :spotdodge, %{}, i)
            |> maybe(entered and a == @roll_f, :roll_forward, %{}, i)
            |> maybe(entered and a == @roll_b, :roll_backward, %{}, i)
            |> maybe(entered and a == @shield_on, :shield_on, %{}, i)
            |> maybe(entered and a in @grab, :grab, %{}, i)
            |> maybe(entered and Map.has_key?(@throws, a), :throw, %{direction: @throws[a]}, i)
            |> maybe(entered and a == @tech_in_place, :tech_in_place, %{}, i)
            |> maybe(entered and a in @tech_roll, :tech_roll, %{}, i)
            |> maybe(entered and a in @tumble_bounce, :missed_tech, %{}, i)
            |> maybe(entered and a in @getup_attack, :getup_attack, %{}, i)
            |> maybe(entered and a in @getup_stand, :getup_stand, %{}, i)
            |> maybe(wavedash_landed?(ctx, p, prev), wavedash_kind(ctx), %{}, i)
            |> attack_event(entered, a, pa, i)
            |> ledge_event(entered, a, pa, i)

          # dashdance fires once per burst: clear flip history after firing
          ctx = if Enum.any?(new, &(&1.option == :dashdance)), do: %{ctx | flips: []}, else: ctx
          ctx = if Enum.any?(new, &(&1.option in [:wavedash, :waveland])), do: %{ctx | airdodge_at: nil}, else: ctx

          {new ++ acc, %{ctx | prev: p}}
        end
      end)

    events |> Enum.reverse() |> Enum.sort_by(& &1.index)
  end

  @doc "Frequency map of options in an event list."
  @spec frequencies([event()]) :: map()
  def frequencies(events), do: events |> Enum.map(& &1.option) |> Enum.frequencies()

  # ==========================================================================
  # Fold context
  # ==========================================================================

  defp new_ctx do
    %{prev: nil, jumpsquat_at: nil, airdodge_at: nil, flips: [], last_dir: nil}
  end

  defp track(ctx, p, prev, i) do
    a = act(p)
    pa = act(prev)

    ctx = if a == @jumpsquat and pa != @jumpsquat, do: %{ctx | jumpsquat_at: i}, else: ctx
    ctx = if a == @airdodge and pa != @airdodge, do: %{ctx | airdodge_at: i}, else: ctx

    # Dash-direction flips (for dashdance): record direction each time a
    # DASH is entered with the opposite direction of the previous dash
    if a == @dashing and pa != @dashing do
      d = dir(p)

      flips =
        if ctx.last_dir != nil and d != ctx.last_dir,
          do: [{i, d} | ctx.flips],
          else: ctx.flips

      %{ctx | flips: Enum.filter(flips, fn {j, _} -> i - j <= @dashdance_window end), last_dir: d}
    else
      ctx
    end
  end

  defp dashdance?(ctx, i) do
    length(Enum.filter(ctx.flips, fn {j, _} -> i - j <= @dashdance_window end)) >=
      @dashdance_min_flips
  end

  defp wavedash_candidate?(ctx, i) do
    ctx.jumpsquat_at != nil and i - ctx.jumpsquat_at <= @wavedash_window
  end

  # Airdodge -> grounded = the wavedash/waveland landing frame
  defp wavedash_landed?(ctx, p, prev) do
    ctx.airdodge_at != nil and p.on_ground == true and prev.on_ground != true and
      act(prev) == @airdodge
  end

  defp wavedash_kind(ctx) do
    if ctx.jumpsquat_at != nil and ctx.airdodge_at != nil and
         ctx.airdodge_at - ctx.jumpsquat_at <= @wavedash_window,
       do: :wavedash,
       else: :waveland
  end

  defp attack_event(events, entered, a, pa, i) do
    cat = NeutralScan.category(a)

    # :grab is owned by the dedicated rule above (which also catches
    # dash-grab 214); firing it here too would double-count
    if entered and cat != nil and cat != :grab and NeutralScan.category(pa) != cat do
      [%{index: i, option: cat, meta: %{action: a}} | events]
    else
      events
    end
  end

  defp ledge_event(events, entered, a, pa, i) do
    case @ledge_options[a] do
      nil ->
        events

      option when pa == @cliff_wait ->
        if entered, do: [%{index: i, option: option, meta: %{}} | events], else: events

      _ ->
        events
    end
  end

  defp maybe(events, true, option, meta, i), do: [%{index: i, option: option, meta: meta} | events]
  defp maybe(events, _cond, _option, _meta, _i), do: events

  defp act(p), do: trunc(p.action || 0)
  defp dir(p), do: if((p.facing || 1) >= 0, do: :right, else: :left)
end
