defmodule ExPhil.Eval.Opening do
  @moduledoc """
  General opening detector (Bradley 2026-09-21: "fair opening is not really
  Fox — grab, up smash, nair, back air, up air, shine are all openings; make
  something that looks for an opening in general").

  Slim frames (`ExPhil.Eval.ScenarioScan.player_summary/1` shape), attacker
  P1, defender P2. An **opening** is any hit edge on the defender — a
  transition into hitstun, a grab-capture state, or a thrown state — with
  the attacker's action on that frame recorded as the **opener** and
  classified by move family. Optionally require `:neutral_lookback` frames
  of no interaction before it (default 0: a drill start is already
  interaction-free by construction; the replay scanners use 60).

  From the opening, the chain walks forward like `AerialChain`: further hit
  edges count while the defender has not had an actionable frame; the chain
  ends at the first actionable gap, a stock change, the window, or the end
  of the frames. An opening is **converted** when it produced a second hit
  before the gap (a true follow-up by action-id inference), or when a grab
  led to a throw. Damage is the defender's percent delta over the chain.

  Actionability is inferred from action ids only (INVARIANTS: label
  uncertain rather than certify); every record carries `actionability:
  :inferred`.
  """

  @hitstun MapSet.new(Enum.to_list(75..91))
  @captured MapSet.new(Enum.to_list(223..232))
  @thrown MapSet.new(Enum.to_list(239..242))
  @hit_states MapSet.union(@hitstun, MapSet.union(@captured, @thrown))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @shield MapSet.new([178, 179, 180, 181, 182])
  @landing MapSet.new([42, 43, 70, 71, 72, 73, 74])
  @non_neutral MapSet.union(@hit_states, MapSet.union(@lifecycle, @shield))

  # Attacker action -> opener family (GALE01 ids; specials are character-specific
  # and land in :special by range).
  @families [
    {:jab, MapSet.new([44, 45, 46, 47, 48, 49])},
    {:dash_attack, MapSet.new([50])},
    {:tilt, MapSet.new([51, 52, 53, 54, 55, 56, 57])},
    {:smash, MapSet.new([58, 59, 60, 61, 62, 63, 64])},
    {:aerial, MapSet.new([65, 66, 67, 68, 69])},
    {:grab, MapSet.new([212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222])},
    {:getup_attack, MapSet.new([185, 190, 193, 198])},
    {:ledge_attack, MapSet.new([258, 259, 260, 261, 262, 263, 264, 265])}
  ]

  @aerial_names %{65 => :nair, 66 => :fair, 67 => :bair, 68 => :uair, 69 => :dair}
  @smash_names %{58 => :fsmash, 59 => :fsmash, 60 => :fsmash, 61 => :fsmash, 62 => :fsmash, 63 => :usmash, 64 => :dsmash}
  @tilt_names %{51 => :ftilt, 52 => :ftilt, 53 => :ftilt, 54 => :ftilt, 55 => :ftilt, 56 => :utilt, 57 => :dtilt}

  @default_window 240

  @type opening :: %{
          frame: integer(),
          opener_action: integer(),
          opener: atom(),
          family: atom(),
          hits: non_neg_integer(),
          damage: float(),
          chain_end: integer(),
          end_reason: :actionable_gap | :window | :stock | :end_of_frames,
          converted?: boolean(),
          actionability: :inferred
        }

  @doc "Every opening with its chain accounting. Options: `:neutral_lookback` (0), `:window` (240)."
  @spec openings([map()], keyword()) :: [opening()]
  def openings(frames, opts \\ []) do
    lookback = Keyword.get(opts, :neutral_lookback, 0)
    window = Keyword.get(opts, :window, @default_window)
    arr = List.to_tuple(frames)
    n = tuple_size(arr)

    if n < 2 do
      []
    else
      {os, _} =
        Enum.reduce(1..(n - 1)//1, {[], -1}, fn i, {acc, busy_until} ->
          prev = elem(arr, i - 1)
          cur = elem(arr, i)

          if i > busy_until and hit?(prev, cur) and not MapSet.member?(@hit_states, prev.p2.action) and neutral_before?(arr, i, lookback) do
            o = walk(arr, i, min(i + window, n - 1))
            {[o | acc], o.end_index}
          else
            {acc, busy_until}
          end
        end)

      os |> Enum.reverse() |> Enum.map(&Map.delete(&1, :end_index))
    end
  end

  @doc "Counts: openings, converted, by family/opener, mean hits and damage per opening."
  def summary(frames, opts \\ []) do
    os = openings(frames, opts)
    n = length(os)

    %{
      openings: n,
      converted: Enum.count(os, & &1.converted?),
      conversion_rate: if(n == 0, do: 0.0, else: Enum.count(os, & &1.converted?) / n),
      by_family: Enum.frequencies_by(os, & &1.family),
      by_opener: Enum.frequencies_by(os, & &1.opener),
      mean_hits: if(n == 0, do: 0.0, else: Enum.sum(Enum.map(os, & &1.hits)) / n),
      mean_damage: if(n == 0, do: 0.0, else: Enum.sum(Enum.map(os, & &1.damage)) / n),
      actionability: :inferred
    }
  end

  @doc "Move family + name for an attacker action id."
  def classify(action) do
    family = Enum.find_value(@families, :special_or_other, fn {fam, set} -> if MapSet.member?(set, action), do: fam end)

    family =
      if family == :special_or_other and action >= 341 and action <= 400, do: :special, else: family

    name =
      case family do
        :aerial -> Map.get(@aerial_names, action, :aerial)
        :smash -> Map.get(@smash_names, action, :smash)
        :tilt -> Map.get(@tilt_names, action, :tilt)
        :grab -> :grab
        :special -> :special
        other -> other
      end

    {family, name}
  end

  defp walk(arr, start, last) do
    first = elem(arr, start)
    stock0 = first.p2.stock
    pct0 = elem(arr, max(start - 1, 0)).p2.percent
    {family, name} = classify(first.p1.action)
    grabbed? = MapSet.member?(@captured, first.p2.action)

    {hits, thrown?, i, reason} =
      Enum.reduce_while((start + 1)..last//1, {1, false, start, :window}, fn i, {hits, thrown?, _, _} ->
        f = elem(arr, i)
        prev = elem(arr, i - 1)

        cond do
          f.p2.stock != stock0 -> {:halt, {hits, thrown?, i, :stock}}
          hit?(prev, f) -> {:cont, {hits + 1, thrown?, i, :window}}
          MapSet.member?(@thrown, f.p2.action) and not MapSet.member?(@thrown, prev.p2.action) -> {:cont, {hits, true, i, :window}}
          actionable?(f.p2.action) -> {:halt, {hits, thrown?, i, :actionable_gap}}
          true -> {:cont, {hits, thrown?, i, :window}}
        end
      end)

    reason = if reason == :window and last == tuple_size(arr) - 1, do: :end_of_frames, else: reason
    endf = elem(arr, i)

    %{
      frame: first.frame,
      opener_action: first.p1.action,
      opener: name,
      family: family,
      hits: hits,
      damage: max(endf.p2.percent - pct0, 0.0) * 1.0,
      chain_end: endf.frame,
      end_reason: reason,
      converted?: hits >= 2 or (grabbed? and thrown?),
      actionability: :inferred,
      end_index: i
    }
  end

  # A hit is visible in the slim shape as a percent increase (every damaging
  # hit, including a second hit that keeps the defender in the SAME hitstun
  # action id — an edge-only detector misses those) or as an edge into a
  # hit state (grabs deal no damage until the throw).
  defp hit?(prev, cur), do: cur.p2.percent > prev.p2.percent or hit_edge?(prev.p2.action, cur.p2.action)

  defp hit_edge?(prev, cur), do: MapSet.member?(@hit_states, cur) and not MapSet.member?(@hit_states, prev)

  defp actionable?(action),
    do: not (MapSet.member?(@hit_states, action) or MapSet.member?(@lifecycle, action) or MapSet.member?(@landing, action))

  defp neutral_before?(_arr, _i, 0), do: true

  defp neutral_before?(arr, i, lookback) do
    lo = i - lookback

    lo >= 0 and
      Enum.all?(lo..(i - 1)//1, fn j ->
        f = elem(arr, j)
        not MapSet.member?(@non_neutral, f.p1.action) and not MapSet.member?(@non_neutral, f.p2.action)
      end)
  end
end
