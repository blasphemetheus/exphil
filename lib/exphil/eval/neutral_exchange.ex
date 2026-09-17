defmodule ExPhil.Eval.NeutralExchange do
  @moduledoc """
  Non-overlapping two-player neutral exchanges. Shielding is eligible neutral;
  shield contact is not an opening. A hit or capture ends an exchange, followed
  by a short trade-observation window. A fresh neutral lead-in is then required.

  Opening attribution to a move is deliberately not inferred here: a recent
  attack alone does not prove it caused the hit. Intended for no-item trials;
  percent changes in recordings with hazards need separate attribution review.
  """

  def rows(replay, subject_port) do
    [opponent] = Enum.reject(replay.metadata.players, &(&1.port == subject_port))
    stage = Melee.Enums.Stage.from_external(replay.metadata.stage)
    edge = Melee.Stages.edge_ground_position(stage)
    if edge == nil, do: raise(ArgumentError, "Unsupported stage geometry")

    for f <- replay.frames, f.frame_number >= 0 do
      player = fn port ->
        p = Map.fetch!(f.players, port)
        if p.in_hitstun == nil, do: raise(ArgumentError, "Replay lacks hitstun flags")

        %{
          action: trunc(p.action),
          percent: p.percent,
          stock: p.stock,
          hitstun: p.in_hitstun,
          offstage: abs(p.x) > edge or p.y < -5.0
        }
      end

      %{frame: f.frame_number, subject: player.(subject_port), opponent: player.(opponent.port)}
    end
  end

  def score(rows, opts \\ []) do
    lead = Keyword.get(opts, :lead, 30)
    max_frames = Keyword.get(opts, :max_frames, 600)
    trade_frames = Keyword.get(opts, :trade_frames, 5)

    unless lead > 0 and max_frames > 0 and trade_frames >= 0,
      do: raise(ArgumentError, "Invalid exchange windows")

    init = %{phase: :idle, streak: 0, previous: nil, events: []}

    state =
      Enum.reduce(rows, init, fn row, s ->
        boundary =
          cond do
            s.previous == nil ->
              nil

            row.frame != s.previous.frame + 1 ->
              :gap

            row.subject.stock != s.previous.subject.stock or
                row.opponent.stock != s.previous.opponent.stock ->
              :stock_change

            true ->
              nil
          end

        s = if boundary, do: close(s, row.frame, {:censored, boundary}), else: s
        s = if boundary, do: %{s | previous: nil, streak: 0}, else: s
        s = advance(s, row, lead, max_frames, trade_frames)
        %{s | previous: row}
      end)

    end_frame = if state.previous, do: state.previous.frame, else: nil
    state = close(state, end_frame, {:censored, :end_of_recording})
    events = Enum.reverse(state.events)

    %{
      exchanges: length(events),
      outcomes: Enum.frequencies_by(events, & &1.outcome),
      events: events
    }
  end

  defp advance(%{phase: :idle} = s, row, lead, _, _) do
    streak = if eligible?(row.subject) and eligible?(row.opponent), do: s.streak + 1, else: 0

    if streak >= lead,
      do: %{s | phase: {:active, row.frame}, streak: 0},
      else: %{s | streak: streak}
  end

  defp advance(%{phase: {:active, start}} = s, row, _, max_frames, trade_frames) do
    subject_hit = hit?(s.previous, row, :subject)
    opponent_hit = hit?(s.previous, row, :opponent)

    cond do
      subject_hit and opponent_hit ->
        close(s, row.frame, :trade)

      subject_hit or opponent_hit ->
        outcome = if opponent_hit, do: :subject_opens, else: :opponent_opens

        if trade_frames == 0,
          do: close(s, row.frame, outcome),
          else: %{s | phase: {:pending, start, row.frame, outcome, trade_frames}}

      row.subject.offstage or row.opponent.offstage ->
        close(s, row.frame, {:censored, :offstage})

      row.frame - start >= max_frames ->
        close(s, row.frame, :timeout)

      true ->
        s
    end
  end

  defp advance(%{phase: {:pending, _, first, outcome, window}} = s, row, _, _, _) do
    other = if outcome == :subject_opens, do: :subject, else: :opponent

    cond do
      hit?(s.previous, row, other) -> close(s, row.frame, :trade)
      row.frame - first >= window -> close(s, row.frame, outcome)
      true -> s
    end
  end

  defp hit?(nil, _, _), do: false

  defp hit?(previous, row, who) do
    a = Map.fetch!(previous, who)
    b = Map.fetch!(row, who)

    b.percent > a.percent or (b.hitstun and not a.hitstun) or
      (captured?(b.action) and not captured?(a.action))
  end

  defp eligible?(p) do
    p.stock > 0 and p.action >= 14 and not p.hitstun and not p.offstage and
      p.action not in 212..232 and p.action not in 239..243 and p.action not in 183..205
  end

  defp captured?(action), do: action in 223..228 or action in [231, 232]

  defp close(%{phase: :idle} = s, _, _), do: s

  defp close(s, finish, outcome) do
    {start, first} =
      case s.phase do
        {:active, start} -> {start, nil}
        {:pending, start, first, _, _} -> {start, first}
      end

    {outcome, reason} =
      case outcome do
        {:censored, reason} -> {:censored, reason}
        other -> {other, nil}
      end

    first =
      if first == nil and outcome in [:subject_opens, :opponent_opens, :trade],
        do: finish,
        else: first

    event = %{
      start: start,
      finish: finish,
      first_contact: first,
      outcome: outcome,
      censor_reason: reason
    }

    %{s | phase: :idle, streak: 0, events: [event | s.events]}
  end
end
