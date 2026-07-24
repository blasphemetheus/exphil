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

  ## The multishine cycle

  A multishine alternates reflector → jumpsquat → reflector → …:

      shine (reflector 360..368)
        -> jump-cancel (jumpsquat 24)
          -> shine (reflector)   # cycle repeats

  A **chain** is a maximal alternating run of reflector/jumpsquat segments;
  its **length** is the number of reflector segments in it (i.e. how many
  shines came out). Length 1 = a lone shine, no multishine. The classic
  drop is jumpsquat → aerial jump (25): jumped but never shined — the
  "empty hop" the trace script already counts.

  Action IDs mirror `ExPhil.Agents.MultishineExpert`.
  """

  # Fox action states (Melee internal), same as MultishineExpert.
  # CRITICAL distinction (added 2026-07-23 after a coarse version conflated
  # them and reported a 75%-aerial fixture as a "clean multishine, max 73"):
  # a real grounded multishine stays on the GROUND — grounded reflector
  # (360..363), jump-cancelled instantly so Fox never leaves the stage. The
  # sloppy shine-jump-AIR-shine loop uses the AERIAL reflector (365..368)
  # and is a different, worse behavior. Counting both as "a shine" hides
  # exactly the failure we care about.
  @jumpsquat 24
  @aerial_jump 25
  @reflector_ground 360..363
  @reflector_air 365..368

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
  Chain lengths (shines per unbroken alternating run) for an action-id list,
  longest-first order preserved as encountered.

  Runs of the same family collapse first (an action persists for several
  frames), so this is robust to frame counts.
  """
  def chains(actions) when is_list(actions) do
    actions |> chains_detailed() |> Enum.map(& &1.length)
  end

  @doc """
  Chains with the reason each one ended — the diagnostic that separates a
  MISSED SHINE from simply stopping.

  Returns `[%{length:, ended_by:}]` where `ended_by` is:

    * `:empty_hop` — jumpsquat → aerial jump. Jumped but never shined; the
      canonical dropped-multishine.
    * `:other_action` — left the loop some other way (landed, got hit,
      did something else).
    * `:end_of_input` — the replay ended mid-chain (not a real failure).

  A run of `:empty_hop` endings at a CONSISTENT length points at a phase /
  timing bug or an expert-table coverage cliff at that depth; a spread of
  lengths points at drift or sampling noise.
  """
  def chains_detailed(actions) when is_list(actions) do
    actions
    |> Enum.map(&family/1)
    |> Enum.chunk_by(& &1)
    |> Enum.map(&hd/1)
    |> collapse_chains()
  end

  # Walk the family-segment sequence, accumulating CLEAN grounded multishine
  # runs: consecutive GROUND reflectors whose only gaps are jumpsquat (the
  # jump-cancel). Length = grounded-shine count. The chain BREAKS on an
  # aerial reflector or an airborne jump — because that means Fox left the
  # ground, which is precisely NOT a clean multishine.
  defp collapse_chains(segments) do
    {chains, current, prev} =
      Enum.reduce(segments, {[], 0, nil}, fn seg, {acc, cur, prev} ->
        case seg do
          :ground_reflect ->
            {acc, cur + 1, seg}

          # jumpsquat keeps an open GROUNDED chain alive (it's the JC)
          :jumpsquat ->
            {acc, cur, seg}

          # left the ground — the chain is over. Distinguish HOW:
          #   air_reflect  -> shined in the air (the sloppy loop)
          #   aerial_jump  -> jumped without shining (empty hop)
          :air_reflect ->
            {close(acc, cur, :air_shine), 0, seg}

          :aerial_jump ->
            reason = if prev == :jumpsquat, do: :empty_hop, else: :aerial_jump
            {close(acc, cur, reason), 0, seg}

          _ ->
            {close(acc, cur, :other_action), 0, seg}
        end
      end)

    _ = prev
    close(chains, current, :end_of_input) |> Enum.reverse()
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
    * `empty_hops` — jumpsquat immediately followed by an aerial jump: the
      canonical dropped-loop failure

  `mean_length`/`max_length` are nil when there are no chains, so a
  no-evidence run reads as nil rather than a misleading 0.0.
  """
  def summary(actions, opts \\ []) when is_list(actions) do
    sustain_min = Keyword.get(opts, :sustain_min, 5)
    detailed = chains_detailed(actions)
    cs = Enum.map(detailed, & &1.length)
    n = length(cs)

    fams = Enum.map(actions, &family/1)
    ground = Enum.count(fams, &(&1 == :ground_reflect))
    air = Enum.count(fams, &(&1 == :air_reflect))

    %{
      # Clean grounded-multishine chains (the thing we actually want)
      chains: n,
      shines: Enum.sum(cs),
      mean_length: if(n > 0, do: Float.round(Enum.sum(cs) / n, 2), else: nil),
      max_length: if(n > 0, do: Enum.max(cs), else: nil),
      entries: Enum.count(cs, &(&1 >= 1)),
      sustained: Enum.count(cs, &(&1 >= sustain_min)),
      length_histogram: Enum.frequencies(cs),
      ended_by: Enum.frequencies_by(detailed, & &1.ended_by),
      # The headline number that catches "it looks like it's shining but
      # it's airborne": fraction of shine frames spent GROUNDED. A clean
      # multishine is ~1.0; the sloppy shine-jump-airshine loop is low.
      ground_shine_frames: ground,
      air_shine_frames: air,
      grounded_fraction:
        if(ground + air > 0, do: Float.round(ground / (ground + air), 3), else: nil),
      empty_hops: empty_hops(actions)
    }
  end

  @doc "Jumpsquat immediately followed by an aerial jump — jumped, never shined."
  def empty_hops(actions) when is_list(actions) do
    actions
    |> Enum.map(&family/1)
    |> Enum.chunk_by(& &1)
    |> Enum.map(&hd/1)
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.count(fn [a, b] -> a == :jumpsquat and b == :aerial_jump end)
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
