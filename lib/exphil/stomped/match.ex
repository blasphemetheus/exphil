defmodule ExPhil.Stomped.Match do
  @moduledoc """
  One match on the stomped.dev arena, as a pure state machine.

  The arena opens a WebSocket per match and sends JSON steps (`character`,
  `ready`, `game`, `over`) as text and the raw `.slp` event stream as binary,
  one message per frame. This module turns each incoming message into the
  messages to send back; it does no I/O, so `ExPhil.Stomped.Socket` is a thin
  wrapper and tests (or an offline replay driver) can call it directly.

  ## Timing

  The arena shows the game `inputDelayFrames` behind the engine. Having seen
  frame F, the first pad still open is `F + 1 + inputDelayFrames`. A bot whose
  round trip (network both ways plus compute) is over the 11 ms budget aims
  further ahead instead: `:lead_extra` adds whole frames to that target. In
  ExPhil's delay numbering the policy then plays at reaction delay
  `k = inputDelayFrames + lead_extra`.
  """

  require Logger

  alias ExPhil.Stomped.{Events, Pad}

  # `game.frame` before the engine has finished any frame.
  @no_frame_yet -124

  @frame_ms 1000 / 60
  # The engine's share of a frame: the arena's budget is one frame minus this.
  @engine_ms 5.67
  # Allowance for our own decision and for jitter in the round trip.
  @compute_ms 2
  @margin_ms 3
  # The arena accepts one ping a second.
  @ping_gap_ms 1_100
  @ping_every_frames 120
  @rtt_window 5

  defstruct bot: nil,
            bot_state: nil,
            lead_extra: 0,
            hold_ahead: 0,
            lead: 2,
            port: nil,
            match_id: nil,
            step: nil,
            pick: nil,
            readied?: false,
            sizes: %{},
            awaiting_replay?: false,
            last_target: nil,
            result: nil,
            frames: 0,
            compute_us_total: 0,
            compute_us_max: 0,
            input_delay: 1,
            game_on?: false,
            # round-trip measurement (the `ping` step)
            ping_at: nil,
            ping_step: nil,
            last_ping_at: nil,
            frames_since_ping: 0,
            rtt_recent: [],
            rtt_all: []

  @type reply :: {:text, iodata()} | {:binary, binary()}
  @type t :: %__MODULE__{}

  @doc """
  Start a match.

  Options:

    * `:bot` — `{module, opts}` implementing `ExPhil.Stomped.Bot` (required)
    * `:lead_extra` — extra frames to aim ahead, for round trips over the
      frame budget (default 0), or `:auto` to set the lead from the
      measured round trip (see `lead_for/2`)
    * `:hold_ahead` — also send each pad for this many following frames
      (default 0). The arena keeps the last pad received before a poll, so
      if the next decision is late the game repeats this input instead of
      reading neutral; a decision that is on time simply overwrites it.
    * `:match_id` — the id from the `x-stomped-match` header, for logs
  """
  @spec new(keyword()) :: t()
  def new(opts) do
    {bot, bot_opts} = Keyword.fetch!(opts, :bot)

    %__MODULE__{
      bot: bot,
      bot_state: bot.init(bot_opts),
      lead_extra: Keyword.get(opts, :lead_extra, 0),
      hold_ahead: Keyword.get(opts, :hold_ahead, 0),
      match_id: Keyword.get(opts, :match_id)
    }
  end

  @doc "Handle a text message (a JSON step). Returns the replies to send, in order."
  @spec handle_text(binary(), t()) :: {[reply()], t()}
  def handle_text(text, %__MODULE__{} = match) do
    case Jason.decode(text) do
      {:ok, %{"step" => step} = frame} ->
        if reason = frame["error"] do
          Logger.warning("[stomped #{match.match_id}] step #{step} rejected: #{reason}")
        end

        match |> sample_ping(step) |> absorb(frame) |> step(step, frame)

      other ->
        Logger.warning("[stomped #{match.match_id}] unreadable text frame: #{inspect(other)}")
        {[], match}
    end
  end

  @doc "Handle a binary message (`.slp` events). Returns the pads to send."
  @spec handle_binary(binary(), t()) :: {[reply()], t()}
  def handle_binary(message, %__MODULE__{} = match) do
    header? = Events.header?(message)

    cond do
      # After a reconnect the live frames resume before the `replay` answer
      # arrives; nothing is readable until the message that opens with 0x35.
      match.awaiting_replay? and not header? ->
        {[], match}

      map_size(match.sizes) == 0 and not header? ->
        {[], match}

      true ->
        started = System.monotonic_time(:microsecond)

        {events, sizes} =
          case Events.parse(message, match.sizes) do
            {:ok, events, sizes} ->
              {events, sizes}

            {:partial, events, sizes, reason} ->
              Logger.warning(
                "[stomped #{match.match_id}] event stream unreadable: #{inspect(reason)}"
              )

              {events, sizes}
          end

        match = %{match | sizes: sizes, awaiting_replay?: false}
        {frames, leftover} = Events.frames(events)
        {pad, match} = play_frames(frames, match)
        match = bot_events(leftover, match)

        replies =
          case pad do
            nil ->
              []

            {target, inputs} ->
              for f <- target..(target + match.hold_ahead)//1,
                  do: {:binary, Pad.encode(f, inputs)}
          end

        {ping, match} = game_ping(match, frames)
        {replies ++ ping, record_compute(match, frames, started)}
    end
  end

  @doc "Timing and result summary, for the log line at the end of a match."
  @spec summary(t()) :: map()
  def summary(%__MODULE__{} = match) do
    %{
      match_id: match.match_id,
      port: match.port,
      lead: match.lead,
      frames: match.frames,
      compute_ms_mean:
        if(match.frames > 0, do: match.compute_us_total / match.frames / 1000, else: 0.0),
      compute_ms_max: match.compute_us_max / 1000,
      rtt_ms: rtt_summary(match.rtt_all),
      result: match.result
    }
  end

  # Every step frame carries the whole match state.
  defp absorb(match, frame) do
    delay = frame["inputDelayFrames"]

    %{
      match
      | port: frame["port"] || match.port,
        match_id: frame["matchId"] || match.match_id,
        input_delay: if(is_integer(delay), do: delay, else: match.input_delay),
        step: frame["step"]
    }
    |> set_lead()
  end

  # The step is sent again after a rejected reply and in answer to a `ping`;
  # the pick is made once and repeated, so a resend can never change it.
  defp step(match, "character", frame) do
    {pick, match} =
      case match.pick do
        nil ->
          {pick, bot_state} = match.bot.pick(frame, match.bot_state)
          {pick, %{match | bot_state: bot_state, pick: pick}}

        pick ->
          {pick, match}
      end

    reply = %{step: "character", character: pick.character, costume: Map.get(pick, :costume, 0)}
    {ping, match} = ping(match)
    {[{:text, Jason.encode!(reply)} | ping], match}
  end

  defp step(match, "ready", frame) do
    match =
      if not match.readied? and function_exported?(match.bot, :ready, 2) do
        %{match | bot_state: match.bot.ready(frame, match.bot_state)}
      else
        match
      end

    {ping, match} = ping(match)
    {[{:text, Jason.encode!(%{step: "ready"})} | ping], %{match | readied?: true}}
  end

  # A `game` frame also comes back in answer to every in-game ping. Only the
  # first one opens the game: sending the neutral cover again would overwrite
  # real pads that are already waiting for their frame.
  defp step(%{game_on?: true} = match, "game", _frame), do: {[], match}

  defp step(match, "game", %{"frame" => frame}) when is_integer(frame) do
    # First in-game ping after half a second, then every two seconds.
    match = %{match | game_on?: true, frames_since_ping: @ping_every_frames - 30}
    # Cover every frame that is already open; one pad per bookend follows.
    last = frame + match.lead
    pads = for f <- (frame + 1)..last//1, do: {:binary, Pad.neutral(f)}
    match = %{match | last_target: max(last, match.last_target || last)}

    # Joined mid-game (a reconnect): the stream resumes without its header.
    if frame > @no_frame_yet and map_size(match.sizes) == 0 do
      {pads ++ [{:text, Jason.encode!(%{step: "replay"})}], %{match | awaiting_replay?: true}}
    else
      {pads, match}
    end
  end

  defp step(match, "over", frame) do
    result = Map.take(frame, ["winner", "endReason", "missed", "players"])

    if function_exported?(match.bot, :over, 2) do
      match.bot.over(frame, match.bot_state)
    end

    {[], %{match | result: result}}
  end

  defp step(match, step, _frame) do
    Logger.warning("[stomped #{match.match_id}] unexpected step #{inspect(step)}")
    {[], match}
  end

  # A `replay` answer holds every finished frame in one message. The bot sees
  # each of them (recurrent policies need the history) but only the pad for
  # the newest frame is worth sending.
  defp play_frames([], match), do: {nil, match}

  defp play_frames(frames, match) do
    last_index = length(frames) - 1

    frames
    |> Enum.with_index()
    |> Enum.reduce({nil, match}, fn {{frame, events}, index}, {_pad, match} ->
      target = frame + match.lead
      ctx = %{port: match.port, lead: match.lead, target: target, live?: index == last_index}
      {inputs, bot_state} = match.bot.handle_frame(frame, events, ctx, match.bot_state)
      match = %{match | bot_state: bot_state}

      if ctx.live? do
        {{target, inputs}, %{match | last_target: target}}
      else
        {nil, match}
      end
    end)
  end

  defp bot_events([], match), do: match

  defp bot_events(events, match) do
    if function_exported?(match.bot, :handle_events, 2) do
      %{match | bot_state: match.bot.handle_events(events, match.bot_state)}
    else
      match
    end
  end

  # Only single-frame messages are timed: a `replay` catch-up is not a frame budget.
  defp record_compute(match, [_one], started) do
    elapsed = System.monotonic_time(:microsecond) - started

    %{
      match
      | frames: match.frames + 1,
        compute_us_total: match.compute_us_total + elapsed,
        compute_us_max: max(match.compute_us_max, elapsed)
    }
  end

  defp record_compute(match, _frames, _started), do: match

  # --- round trip and lead ------------------------------------------------

  @doc """
  The lead (frames between the frame just seen and the pad's target) that a
  round trip of `rtt_ms` needs, given the arena's `inputDelayFrames`.

  Frame F's events are sent once frame `F + input_delay` is computed, about
  #{@engine_ms} ms after that frame's tick, and the pad for `F + k` is read at
  tick `F + k`. So a pad has `(k - input_delay) * 16.67 - #{@engine_ms}` ms to
  make the round trip, our compute included. With no network at all this is
  the arena's own figure: `k = input_delay + 1` and 11 ms.
  """
  @spec lead_for(number(), non_neg_integer()) :: pos_integer()
  def lead_for(rtt_ms, input_delay) do
    needed = rtt_ms + @compute_ms + @margin_ms + @engine_ms
    input_delay + max(1, ceil(needed / @frame_ms))
  end

  defp set_lead(%{lead_extra: :auto, rtt_recent: []} = match),
    do: %{match | lead: 1 + match.input_delay}

  # The slowest recent round trip, so one quick sample cannot shorten the lead.
  defp set_lead(%{lead_extra: :auto} = match),
    do: %{match | lead: lead_for(Enum.max(match.rtt_recent), match.input_delay)}

  defp set_lead(match), do: %{match | lead: 1 + match.input_delay + match.lead_extra}

  # Ask for the current step again and time the answer. The arena allows one
  # ping a second.
  defp ping(%{ping_at: nil} = match) do
    now = System.monotonic_time(:millisecond)

    if match.last_ping_at == nil or now - match.last_ping_at >= @ping_gap_ms do
      match = %{
        match
        | ping_at: now,
          last_ping_at: now,
          ping_step: match.step,
          frames_since_ping: 0
      }

      {[{:text, Jason.encode!(%{step: "ping"})}], match}
    else
      {[], match}
    end
  end

  defp ping(match), do: {[], match}

  # The answer to a ping is the current step's frame. If the step moved on in
  # the meantime, the frame that arrived may be the arena's own broadcast of
  # the new step, sent before our ping got there: that would read as a round
  # trip shorter than the real one, so the sample is dropped.
  defp sample_ping(%{ping_at: nil} = match, _step), do: match

  defp sample_ping(%{ping_step: step} = match, step) do
    rtt = System.monotonic_time(:millisecond) - match.ping_at

    %{
      match
      | ping_at: nil,
        rtt_recent: Enum.take([rtt | match.rtt_recent], @rtt_window),
        rtt_all: [rtt | match.rtt_all]
    }
  end

  defp sample_ping(match, _other_step), do: %{match | ping_at: nil}

  # One live frame per message is the steady state; a `replay` catch-up is not.
  defp game_ping(match, [_one]) do
    match = %{match | frames_since_ping: match.frames_since_ping + 1}

    cond do
      # An unanswered ping (five seconds) is forgotten, not waited on forever.
      match.ping_at != nil and match.frames_since_ping > 300 -> {[], %{match | ping_at: nil}}
      match.frames_since_ping >= @ping_every_frames -> ping(match)
      true -> {[], match}
    end
  end

  defp game_ping(match, _frames), do: {[], match}

  defp rtt_summary([]), do: nil

  defp rtt_summary(samples) do
    sorted = Enum.sort(samples)

    %{
      n: length(sorted),
      min: hd(sorted),
      median: Enum.at(sorted, div(length(sorted), 2)),
      max: List.last(sorted)
    }
  end
end
