defmodule ExPhil.Eval.MewtwoNeutralBenchmark do
  @moduledoc "Replay diagnostics for the first Fox/FD neutral drill; move entries are attempts, not wins."
  alias ExPhil.Eval.NeutralExchange

  def qualification(runs) do
    if Enum.all?(runs, & &1.valid) do
      %{
        opening_each_run:
          Enum.all?(runs, &((&1.metrics.neutral.outcomes[:subject_opens] || 0) > 0)),
        down_tilt: Enum.any?(runs, &(&1.metrics.down_tilt_attempts > 0)),
        grab: Enum.any?(runs, &(&1.metrics.grab_attempts > 0)),
        aerial: Enum.any?(runs, &(&1.metrics.aerial_attempts > 0)),
        short_hop: Enum.any?(runs, &(&1.metrics.short_hops > 0)),
        wavedash: Enum.any?(runs, &(&1.metrics.wavedashes > 0)),
        shield: Enum.all?(runs, &(&1.metrics.max_shield_action_streak <= 120)),
        no_standing_sd: Enum.all?(runs, &(&1.mode != "stand" or &1.metrics.deaths == 0))
      }
    else
      %{valid_sessions: false}
    end
  end

  def score(replay, last_frame) do
    frames = Enum.filter(replay.frames, &(&1.frame_number >= 0 and &1.frame_number <= last_frame))
    pairs = Enum.chunk_every(frames, 2, 1, :discard)

    onsets =
      pairs
      |> Enum.filter(fn [a, b] -> a.players[1].action != b.players[1].action end)
      |> Enum.frequencies_by(fn [_, b] -> trunc(b.players[1].action) end)

    waves =
      pairs
      |> Enum.with_index()
      |> Enum.count(fn {[a, b], index} ->
        before = a.players[1]
        p = b.players[1]
        c = p.controller
        later = Enum.at(frames, min(index + 11, length(frames) - 1)).players[1]

        not before.on_ground and p.action == 43 and before.action != 43 and
          before.action in [25, 26, 236] and
          (c.button_l or c.button_r) and c.main_stick_y < 0.4 and abs(c.main_stick_x - 0.5) > 0.2 and
          abs(later.x - before.x) > 5
      end)

    {_, _, hops} =
      Enum.reduce(frames, {nil, nil, []}, fn g, {previous, hop, hops} ->
        p = g.players[1]

        hop =
          if previous != nil and previous.action == 24 and not p.on_ground,
            do: %{floor: previous.y, peak: p.y, valid: true},
            else: hop

        hop =
          if hop,
            do: %{
              hop
              | peak: max(hop.peak, p.y),
                valid: hop.valid and not p.in_hitstun and p.action not in [27, 28, 43, 236]
            },
            else: nil

        if hop && p.on_ground do
          result = if hop.valid, do: [hop.peak - hop.floor | hops], else: hops
          {p, nil, result}
        else
          {p, hop, hops}
        end
      end)

    {_, shield_max} =
      Enum.reduce(frames, {0, 0}, fn g, {streak, maximum} ->
        next = if g.players[1].action in 178..182, do: streak + 1, else: 0
        {next, max(maximum, next)}
      end)

    rows = NeutralExchange.rows(replay, 1) |> Enum.filter(&(&1.frame <= last_frame))

    %{
      frames: length(frames),
      neutral: NeutralExchange.score(rows),
      down_tilt_attempts: onsets[57] || 0,
      grab_attempts: (onsets[212] || 0) + (onsets[214] || 0),
      aerial_attempts: Enum.sum(for a <- 65..69, do: onsets[a] || 0),
      aerial_types: Map.take(onsets, Enum.to_list(65..69)),
      wavedashes: waves,
      short_hops: Enum.count(hops, &(&1 > 1 and &1 < 20)),
      hop_apices: hops,
      max_shield_action_streak: shield_max,
      deaths: Enum.sum(for [a, b] <- pairs, do: max(0, a.players[1].stock - b.players[1].stock))
    }
  end
end
