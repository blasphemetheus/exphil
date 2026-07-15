defmodule ExPhil.Interp.GroundTruth do
  @moduledoc """
  Per-frame ground-truth features from parsed replays (INTERP_ROADMAP
  Phase 0).

  Melee gives interpretability something LLMs never do: the complete true
  game state per frame. Every probe target here is exact, not annotated.
  Action-state IDs follow libmelee; the knockdown sets match
  `scripts/trace_tech_chase.exs` (the scoreboard's definitions) so probe
  results stay comparable with conversion scores.

  Tech-choice encoding (the entry state IS the choice):
    0 = missed tech (183 DownBoundU / 191 DownBoundD)
    1 = tech in place (199 Passive)
    2 = tech roll forward (200 PassiveStandF)
    3 = tech roll backward (201 PassiveStandB)
   -1 = not applicable
  """

  @knockdown_entries [183, 191, 199, 200, 201]
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @shield MapSet.new([178, 179, 180])

  # Distance buckets: 15 units each (matches MewtwoFairExpert's range keying),
  # capped at bucket 5 (>= 75 units = "far").
  @dist_bucket_size 15.0
  @dist_bucket_max 5

  # next_kd lookahead horizon (frames). 60 = one second; tech animations
  # resolve well inside this, so probing "next_kd_choice" at negative
  # offsets tests prediction-before-visibility (Phase 4).
  @next_kd_horizon 60

  @doc """
  Compute per-frame label maps for a list of training frames (as returned by
  `Peppi.to_training_frames/2`, already filtered to `frame >= 0`).

  Returns a list (one map per frame) — pass through `align_to_windows/3` to
  get window-aligned Nx tensors.
  """
  def frame_labels(frames, opts \\ []) do
    opponent_port = Keyword.get(opts, :opponent_port, 2)
    player_port = Keyword.get(opts, :player_port, 1)

    states =
      Enum.map(frames, fn f ->
        {f.game_state.players[player_port], f.game_state.players[opponent_port]}
      end)

    opp_actions = Enum.map(states, fn {_own, opp} -> action_of(opp) end)
    tech_choices = episode_tech_choices(opp_actions)
    next_kds = next_kd_choices(opp_actions)

    # Memory-dependent labels (v2): these CANNOT be copied from the current
    # frame — they exist only if the GRU integrates over its window.
    entries = episode_entry_flags(opp_actions)
    since_buckets = time_since_kd_buckets(entries)
    until_buckets = frames_until_kd_buckets(entries)
    opp_percents = Enum.map(states, fn {_own, opp} -> percent_of(opp) end)
    damaged_recent = damaged_within(opp_percents, 30)

    [states, tech_choices, next_kds, since_buckets, until_buckets, damaged_recent]
    |> Enum.zip()
    |> Enum.map(fn {{own, opp}, tech_choice, next_kd, since_b, until_b, dmg30} ->
      opp_action = action_of(opp)
      own_action = action_of(own)
      dist = distance(own, opp)

      %{
        own_action: own_action,
        opp_action: opp_action,
        opp_knockdown: bool01(MapSet.member?(@lifecycle, opp_action)),
        opp_hitstun: bool01(MapSet.member?(@hitstun, opp_action)),
        opp_shielding: bool01(MapSet.member?(@shield, opp_action)),
        own_shielding: bool01(MapSet.member?(@shield, own_action)),
        opp_offstage: bool01(offstage?(opp)),
        own_offstage: bool01(offstage?(own)),
        own_airborne: bool01(own != nil and own.on_ground == false),
        distance: dist,
        dist_bucket: min(trunc(dist / @dist_bucket_size), @dist_bucket_max),
        own_percent: percent_of(own),
        opp_percent: percent_of(opp),
        tech_choice: tech_choice,
        next_kd_choice: next_kd,
        time_since_kd_bucket: since_b,
        frames_until_kd_bucket: until_b,
        opp_damaged_recent: dmg30,
        own_facing: facing_of(own),
        # The fair-in-place pathology's ground truth (P3 case #3): is the
        # opponent BEHIND the bot (opposite side from facing)?
        opp_behind: opp_behind(own, opp)
      }
    end)
  end

  @doc """
  Align per-frame labels to window-end decision frames and tensorize.

  Row `i` of a stride-1, window-`w` activation matrix corresponds to frame
  `i + w - 1`; this slices labels accordingly and stacks each feature into
  an Nx tensor of length `n`.
  """
  def align_to_windows(per_frame_labels, window, n) do
    aligned = Enum.slice(per_frame_labels, window - 1, n)

    if length(aligned) != n do
      raise ArgumentError,
            "label/activation misalignment: #{length(aligned)} labels for #{n} rows " <>
              "(window #{window}, #{length(per_frame_labels)} frames)"
    end

    aligned
    |> hd()
    |> Map.keys()
    |> Map.new(fn key ->
      values = Enum.map(aligned, &Map.fetch!(&1, key))

      type =
        cond do
          key in [:distance, :own_percent, :opp_percent] ->
            :f32

          key in [
            :own_action,
            :opp_action,
            :tech_choice,
            :next_kd_choice,
            :time_since_kd_bucket,
            :frames_until_kd_bucket,
            :own_facing
          ] ->
            # s64 — several of these carry -1 as the mask value
            :s64

          true ->
            :u8
        end

      {key, Nx.tensor(values, type: type, backend: Nx.BinaryBackend)}
    end)
  end

  @doc """
  Names of the binary (u8 0/1) features — the natural first probe targets.
  """
  def binary_features do
    [
      :opp_knockdown,
      :opp_hitstun,
      :opp_shielding,
      :own_shielding,
      :opp_offstage,
      :own_offstage,
      :own_airborne
    ]
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp action_of(nil), do: 0
  defp action_of(player), do: trunc(player.action || 0)

  defp percent_of(nil), do: 0.0
  defp percent_of(player), do: (player.percent || 0) * 1.0

  defp offstage?(nil), do: false

  defp offstage?(player) do
    x = player.x || 0.0
    y = player.y || 0.0
    # FD edge = 85.57; drill games are all FD. Generalize per-stage later.
    abs(x) > 85.57 or y < -5.0
  end

  defp distance(nil, _), do: 0.0
  defp distance(_, nil), do: 0.0

  defp distance(own, opp) do
    dx = (own.x || 0.0) - (opp.x || 0.0)
    dy = (own.y || 0.0) - (opp.y || 0.0)
    :math.sqrt(dx * dx + dy * dy)
  end

  defp bool01(true), do: 1
  defp bool01(_), do: 0

  defp facing_of(nil), do: 1
  defp facing_of(player), do: if((player.facing || 1) >= 0, do: 1, else: -1)

  defp opp_behind(nil, _), do: 0
  defp opp_behind(_, nil), do: 0

  defp opp_behind(own, opp) do
    dx = (opp.x || 0.0) - (own.x || 0.0)
    side = if dx >= 0, do: 1, else: -1
    bool01(side != facing_of(own))
  end

  defp entry_class(a) when a in [183, 191], do: 0
  defp entry_class(199), do: 1
  defp entry_class(200), do: 2
  defp entry_class(201), do: 3
  defp entry_class(_), do: -1

  # Forward pass: for each frame, the entry class of the CURRENT knockdown
  # lifecycle episode (-1 outside episodes). An episode starts when the
  # action transitions into an entry state from outside the lifecycle.
  defp episode_tech_choices(opp_actions) do
    {choices, _} =
      Enum.map_reduce(opp_actions, {-1, false}, fn a, {current, was_in} ->
        in_lifecycle = MapSet.member?(@lifecycle, a)

        current =
          cond do
            in_lifecycle and not was_in and a in @knockdown_entries -> entry_class(a)
            in_lifecycle -> current
            true -> -1
          end

        {current, {current, in_lifecycle}}
      end)

    choices
  end

  # Backward pass: for each frame, the entry class of the NEXT episode start
  # within the lookahead horizon (-1 if none). Probing this at frames BEFORE
  # the knockdown tests prediction vs reaction (Phase 4).
  defp next_kd_choices(opp_actions) do
    entries = episode_entry_flags(opp_actions)

    entries
    |> Enum.reverse()
    |> Enum.map_reduce({-1, @next_kd_horizon + 1}, fn entry, {cls, dist} ->
      {cls, dist} = if entry >= 0, do: {entry, 0}, else: {cls, dist + 1}
      value = if dist <= @next_kd_horizon, do: cls, else: -1
      {value, {cls, dist}}
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  # v2 memory targets ---------------------------------------------------------

  # Frames since the last episode entry, bucketed. Class 4 (">180 / never")
  # doubles as a sanity ceiling: at window 60, distinguishing bucket 3
  # (61-180) from 4 is impossible for the trunk too — probes should fail it.
  #   0 = entry frame, 1 = 1-15, 2 = 16-60, 3 = 61-180, 4 = >180 or never
  defp time_since_kd_buckets(entries) do
    {buckets, _} =
      Enum.map_reduce(entries, 1_000_000, fn entry, dist ->
        dist = if entry >= 0, do: 0, else: dist + 1

        bucket =
          cond do
            dist == 0 -> 0
            dist <= 15 -> 1
            dist <= 60 -> 2
            dist <= 180 -> 3
            true -> 4
          end

        {bucket, dist}
      end)

    buckets
  end

  # Frames until the NEXT episode entry, bucketed; -1 beyond the horizon.
  # The per-bucket decodability curve is the lead-time analysis (P4) and the
  # instrument for the next_kd=1.000 mystery.
  #   0 = entry frame, 1 = 1-5, 2 = 6-15, 3 = 16-30, 4 = 31-60, -1 = farther
  defp frames_until_kd_buckets(entries) do
    entries
    |> Enum.reverse()
    |> Enum.map_reduce(1_000_000, fn entry, dist ->
      dist = if entry >= 0, do: 0, else: dist + 1

      bucket =
        cond do
          dist == 0 -> 0
          dist <= 5 -> 1
          dist <= 15 -> 2
          dist <= 30 -> 3
          dist <= 60 -> 4
          true -> -1
        end

      {bucket, dist}
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  # Did the opponent's percent increase within the last `horizon` frames?
  # Pure memory: the current frame shows the percent, not its history.
  defp damaged_within([], _horizon), do: []

  defp damaged_within(percents, horizon) do
    shifted = List.duplicate(hd(percents), horizon) ++ percents

    percents
    |> Enum.zip(shifted)
    |> Enum.map(fn {now, earlier} -> bool01(now > earlier) end)
  end

  # Per-frame: the entry class if this frame STARTS an episode, else -1.
  defp episode_entry_flags(opp_actions) do
    {flags, _} =
      Enum.map_reduce(opp_actions, false, fn a, was_in ->
        in_lifecycle = MapSet.member?(@lifecycle, a)

        flag =
          if in_lifecycle and not was_in and a in @knockdown_entries,
            do: entry_class(a),
            else: -1

        {flag, in_lifecycle}
      end)

    flags
  end
end
