defmodule ExPhil.Eval.ShineChain do
  @moduledoc """
  Multishine chain metrics for the Fox execution track (GOALS.md Track B).

  `scripts/trace_multishine.exs` already counts EVENTS — jump-cancels,
  shines-out-of-jumpsquat, empty hops. What it cannot answer is the actual
  goal: *how long does the loop hold before it drops?* "It multishines
  sometimes" is ambiguous between two very different failures:

    * **entry failure** — rarely starts the loop at all (a decision/timing
      problem at the start), and
    * **sustain failure** — starts fine but drops after one or two
      (an execution problem inside the loop).

  Chain length separates them, so the fix can be aimed correctly.

  ## The multishine cycle (corrected 2026-07-24)

  A real multishine is NOT fully grounded. Per SmashWiki (and every TAS),
  multishining is "repeatedly shining, jump-cancelling it, and shining again
  **the frame they leave the ground**":

      shine (grounded reflector 360..363)
        -> jump-cancel (jumpsquat 24, 3 frames — only up-smash/up-B/grab can
           cancel jumpsquat; down-B during it is EATEN)
          -> takeoff (aerial jump 25, 1-2 frames)
            -> AERIAL shine on the first airborne frame (365..368) — halts
               Fox's rise, so he lands within a few frames (canonically ~5
               airborne total)
              -> reflector persists into the grounded state
                 # cycle repeats via JC of the landed grounded reflector

  So each cycle contains a SHORT airborne stretch WITH an aerial shine in it.
  What distinguishes the sloppy shine-jump-airshine loop (the 2026-07-23
  fixture) is not that it shines in the air — it is that its air stretch is a
  FULL JUMP (tens of frames).

  A **chain** is a run of grounded-reflector segments linked by jumpsquat
  and/or an airborne stretch of at most `:max_air_gap` frames (default
  #{8}) that contains an aerial reflector. Its **length** is the number of
  grounded reflector segments — i.e. completed cycles.

  ## Metric history (why this is v3)

  v1 conflated grounded and aerial reflectors and reported the 75%-aerial
  fixture as "clean, max 73". v2 over-corrected: it required
  `grounded_fraction ~ 1.0` and broke the chain on ANY aerial reflector —
  but a zero-airtime multishine is not a real Melee technique (the 2026-07-24
  probes "proving the bridge can't shine-cancel jumpsquat" were testing a
  mechanic that does not exist). v3 encodes the real cycle: short airtime
  with an aerial shine CONTINUES the chain; long airtime or shineless
  airtime breaks it.

  Action IDs mirror `ExPhil.Agents.MultishineExpert`.
  """

  # Fox action states (Melee internal), same as MultishineExpert.
  @jumpsquat 24
  @aerial_jump 25
  @reflector_ground 360..363
  @reflector_air 365..368

  # Max airborne frames (takeoff + aerial reflector) between two grounded
  # reflector segments for the chain to continue. A real multishine cycle is
  # ~5 airborne frames; a short hop is ~30+, a full hop far more — 8 cleanly
  # separates the technique from the sloppy full-jump loop.
  @default_max_air_gap 8

  @doc """
  Family of one action id: `:ground_reflect`, `:air_reflect`, `:jumpsquat`,
  `:aerial_jump`, or `:other`.
  """
  def family(a) when a in @reflector_ground, do: :ground_reflect
  def family(a) when a in @reflector_air, do: :air_reflect
  def family(@jumpsquat), do: :jumpsquat
  def family(@aerial_jump), do: :aerial_jump
  def family(_), do: :other

  @doc """
  Chain lengths (grounded shines per unbroken multishine run) for an
  action-id list, order preserved as encountered.

  Options: `:max_air_gap` (default #{@default_max_air_gap}) — max airborne
  frames per cycle before the chain breaks.
  """
  def chains(actions, opts \\ []) when is_list(actions) do
    actions |> chains_detailed(opts) |> Enum.map(& &1.length)
  end

  @doc """
  Chains with the reason each one ended — the diagnostic that separates a
  MISSED SHINE from simply stopping.

  Returns `[%{length:, ended_by:}]` where `ended_by` is:

    * `:air_shine` — the airborne stretch had an aerial shine but exceeded
      `:max_air_gap`: the sloppy full-jump shine-on-the-way-down loop.
    * `:empty_hop` — left the ground from jumpsquat and never shined; the
      canonical dropped-multishine.
    * `:aerial_jump` — airborne with no shine, not straight from jumpsquat.
    * `:other_action` — left the loop some other way (landed and waited,
      got hit, did something else).
    * `:end_of_input` — the replay ended mid-chain (not a real failure).

  A run of `:empty_hop` endings at a CONSISTENT length points at a phase /
  timing bug or an expert-table coverage cliff at that depth; a spread of
  lengths points at drift or sampling noise.
  """
  def chains_detailed(actions, opts \\ []) when is_list(actions) do
    max_air_gap = Keyword.get(opts, :max_air_gap, @default_max_air_gap)

    actions
    |> Enum.map(&family/1)
    |> Enum.chunk_by(& &1)
    |> Enum.map(&{hd(&1), length(&1)})
    |> collapse_chains(max_air_gap)
  end

  # Walk {family, frame_count} segments, accumulating multishine runs.
  # Grounded reflectors extend the chain; jumpsquat keeps it alive (the JC);
  # an airborne stretch is TENTATIVE until Fox returns to the ground — short
  # + containing an aerial shine continues the chain, anything else breaks it.
  defp collapse_chains(segments, max_air_gap) do
    init = %{acc: [], cur: 0, air: 0, air_shine?: false, from_js?: false, prev: nil}

    st =
      Enum.reduce(segments, init, fn {fam, n}, st ->
        case fam do
          :ground_reflect ->
            st = land(st, max_air_gap)
            %{st | cur: st.cur + 1, prev: fam}

          :jumpsquat ->
            st = land(st, max_air_gap)
            %{st | prev: fam}

          :air_reflect ->
            %{
              st
              | air: st.air + n,
                air_shine?: true,
                from_js?: st.from_js? or st.prev == :jumpsquat,
                prev: fam
            }

          :aerial_jump ->
            %{
              st
              | air: st.air + n,
                from_js?: st.from_js? or st.prev == :jumpsquat,
                prev: fam
            }

          _other ->
            reason = if st.air > 0, do: gap_reason(st, max_air_gap), else: :other_action

            %{
              st
              | acc: close(st.acc, st.cur, reason),
                cur: 0,
                air: 0,
                air_shine?: false,
                from_js?: false,
                prev: fam
            }
        end
      end)

    close(st.acc, st.cur, :end_of_input) |> Enum.reverse()
  end

  # Back on the ground with a pending airborne stretch: does it EXTEND the
  # chain (short, with an aerial shine — a real multishine cycle) or BREAK it
  # (full jump / shineless hop)?
  defp land(%{air: 0} = st, _max_air_gap), do: st

  defp land(st, max_air_gap) do
    if st.air_shine? and st.air <= max_air_gap do
      %{st | air: 0, air_shine?: false, from_js?: false}
    else
      %{
        st
        | acc: close(st.acc, st.cur, gap_reason(st, max_air_gap)),
          cur: 0,
          air: 0,
          air_shine?: false,
          from_js?: false
      }
    end
  end

  defp gap_reason(st, max_air_gap) do
    cond do
      # Air part was a valid cycle; the break is whatever came after it.
      st.air_shine? and st.air <= max_air_gap -> :other_action
      st.air_shine? -> :air_shine
      st.from_js? -> :empty_hop
      true -> :aerial_jump
    end
  end

  defp close(acc, 0, _reason), do: acc
  defp close(acc, n, reason), do: [%{length: n, ended_by: reason} | acc]

  @doc """
  Summary for the Track B gates.

  Returns `%{chains:, shines:, mean_length:, max_length:, entries:,
  sustained:, empty_hops:}` where:

    * `entries` — chains that got at least one shine out (B1 signal)
    * `sustained` — chains of length >= `:sustain_min` (default 5, the
      B2 target)
    * `empty_hops` — left the ground from jumpsquat and never shined: the
      canonical dropped-loop failure

  `grounded_fraction` is a DIAGNOSTIC, not a gate: a real multishine sits
  around 0.4-0.6 (each cycle has both a grounded and an aerial reflector
  segment). ~1.0 means lone ground shines with no cycling; ~0.25 with long
  airtime is the sloppy full-jump loop. Gate on `max_length`/`sustained`.

  `mean_length`/`max_length` are nil when there are no chains, so a
  no-evidence run reads as nil rather than a misleading 0.0.
  """
  def summary(actions, opts \\ []) when is_list(actions) do
    sustain_min = Keyword.get(opts, :sustain_min, 5)
    detailed = chains_detailed(actions, opts)
    cs = Enum.map(detailed, & &1.length)
    n = length(cs)

    fams = Enum.map(actions, &family/1)
    ground = Enum.count(fams, &(&1 == :ground_reflect))
    air = Enum.count(fams, &(&1 == :air_reflect))

    %{
      # Multishine chains (grounded shines linked by JC + short aerial shine)
      chains: n,
      shines: Enum.sum(cs),
      mean_length: if(n > 0, do: Float.round(Enum.sum(cs) / n, 2), else: nil),
      max_length: if(n > 0, do: Enum.max(cs), else: nil),
      entries: Enum.count(cs, &(&1 >= 1)),
      sustained: Enum.count(cs, &(&1 >= sustain_min)),
      length_histogram: Enum.frequencies(cs),
      ended_by: Enum.frequencies_by(detailed, & &1.ended_by),
      ground_shine_frames: ground,
      air_shine_frames: air,
      grounded_fraction:
        if(ground + air > 0, do: Float.round(ground / (ground + air), 3), else: nil),
      empty_hops: empty_hops(actions, opts)
    }
  end

  @doc """
  Left the ground from jumpsquat and never shined — jumped without the
  multishine's first-airborne-frame shine.

  A good multishine cycle also passes through jumpsquat -> takeoff, but its
  takeoff is 1-2 frames and immediately followed by the aerial reflector;
  that is NOT an empty hop. Counted as one when the takeoff exceeds
  `:max_air_gap` frames or is followed by anything but an aerial reflector.
  """
  def empty_hops(actions, opts \\ []) when is_list(actions) do
    max_air_gap = Keyword.get(opts, :max_air_gap, @default_max_air_gap)

    actions
    |> Enum.map(&family/1)
    |> Enum.chunk_by(& &1)
    |> Enum.map(&{hd(&1), length(&1)})
    |> Enum.chunk_every(3, 1)
    |> Enum.count(fn
      [{:jumpsquat, _}, {:aerial_jump, n} | rest] ->
        n > max_air_gap or not match?([{:air_reflect, _} | _], rest)

      _ ->
        false
    end)
  end

  @doc """
  Summary straight from a parsed replay path (port defaults to 1).
  Returns `{:ok, summary}` or `{:error, reason}`.
  """
  def summary_for_replay(path, opts \\ []) do
    port = Keyword.get(opts, :port, 1)

    case ExPhil.Data.Peppi.parse(Path.expand(path)) do
      {:ok, replay} ->
        actions =
          replay
          |> ExPhil.Data.Peppi.to_training_frames(
            player_port: port,
            opponent_port: if(port == 1, do: 2, else: 1)
          )
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(fn f -> trunc((f.game_state.players[port] && f.game_state.players[port].action) || 0) end)

        {:ok, summary(actions, opts)}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
