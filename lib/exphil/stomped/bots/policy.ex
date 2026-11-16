defmodule ExPhil.Stomped.Bots.Policy do
  @moduledoc """
  Plays a trained checkpoint on the stomped.dev arena.

  Each frame's events go through `ExPhil.Stomped.Decoder` (the state exactly
  as training built it), the state through the warm `ExPhil.Agents.Agent`
  borrowed from an `ExPhil.Stomped.Seat`, and the agent's controller through
  `ExPhil.Stomped.Pad.from_controller/1`.

  Every frame from the first (−123) is fed to the agent: training frames
  include the countdown, and a recurrent policy should enter frame 0 warm.

  ## Measuring what landed

  The stream echoes the controller the game polled for our port, so each
  frame is checked against the pad that was sent for it:

    * `landed` — the game read the pad we meant;
    * `late` — it read something else (the pad arrived after the poll);
    * `unknown` — the pad was neutral, which a late pad also reads as.

  With the server's hold-ahead on, a late pad is covered by a repeat of the
  previous one, so the arena's own `missed` count stays at zero. This count
  is the only measure of lateness, and the one to pick `--lead-extra` from.

  ## The delay the policy plays at

  The agent is told one reaction delay at boot (`start/1`), and the pad target
  the match asks for implies one (`ctx.reaction_delay`, see
  `ExPhil.Stomped.Match`). They must agree, or the policy answers a different
  question than it was trained on: a frame counted under `delay_mismatch` is
  a live pad sent at a lead the agent was not told about. The count is logged
  with the landed/late counts; a match that is not zero here was not a clean
  measurement of the policy.

  ## Options

    * `:seat` — the `ExPhil.Stomped.Seat` holding the agent (required)
    * `:reaction_delay` — the delay the seated agent plays at (from
      `start/1`), checked against every live frame's lead
    * `:trace` — a file path; every scored frame is appended as
      `frame,outcome,arrival_us,sent,echo` (outcome: landed, fresh,
      late_1..late_4, neutral, unmatched, unknown; arrival is this process's
      monotonic clock when the echoing frame came in; sent and echo are the
      pad bytes as `buttons:stick_x:stick_y:c_x:c_y:l:r`), so late pads can
      be read in time and field by field offline
    * `:character`, `:costume` — the pick (default 2 = Fox, costume 0)

  If the seat is busy (a second match while one is live) the bot sends
  neutral pads for that match rather than share recurrent state.
  """

  @behaviour ExPhil.Stomped.Bot

  require Logger

  alias ExPhil.Agents.Agent
  alias ExPhil.Stomped.{Decoder, Pad}

  # The game zeroes a stick axis inside this raw offset and clamps the stick
  # to a circle of this radius.
  @dead_zone 23
  @stick_radius 80
  # Sent pads are only needed until their frame's echo arrives.
  @sent_window 64
  # How many frames back to look for the pad the game read instead.
  @late_search 4

  @doc """
  Load a checkpoint, warm its agent and put it in a seat.

  Call once at server boot; pass the returned `:seat` in the bot's options.

  Options:

    * `:policy` — path to the exported policy (required)
    * `:reaction_delay` — the delay k the agent plays at; the server's lead
      must then be `lead_extra = k - inputDelayFrames` (k - 1 on the arena)
    * `:lead_extra` — alternative to `:reaction_delay`: the server's lead,
      so k = `1 + lead_extra` (the arena's `inputDelayFrames` is 1)
    * `:stateful_step` — O(1) recurrent inference (default true)
    * `:temperature`, `:deterministic` — sampling (default: sampled at 1.0)

  Returns `{:ok, %{agent:, seat:, warmup_ms:, reaction_delay:}}`; pass
  `reaction_delay` on to the bot's options.
  """
  @spec start(keyword()) :: {:ok, map()} | {:error, term()}
  def start(opts) do
    reaction_delay = Keyword.get(opts, :reaction_delay) || 1 + Keyword.get(opts, :lead_extra, 0)

    agent_opts = [
      policy_path: Keyword.fetch!(opts, :policy),
      deterministic: Keyword.get(opts, :deterministic, false),
      temperature: Keyword.get(opts, :temperature, 1.0),
      # The arena delays nothing on our side: the lead is in the pad's frame number.
      frame_delay: 0,
      harness: :stomped,
      harness_knob: max(reaction_delay - 1, 0),
      reaction_delay: reaction_delay,
      stateful_step: Keyword.get(opts, :stateful_step, true)
    ]

    with {:ok, agent} <- Agent.start_link(agent_opts),
         {:ok, warmup_ms} <- Agent.warmup(agent),
         {:ok, seat} <- ExPhil.Stomped.Seat.start_link(agent: agent) do
      {:ok, %{agent: agent, seat: seat, warmup_ms: warmup_ms, reaction_delay: reaction_delay}}
    end
  end

  @impl true
  def init(opts) do
    %{
      seat: Keyword.fetch!(opts, :seat),
      reaction_delay: Keyword.get(opts, :reaction_delay),
      trace: open_trace(Keyword.get(opts, :trace)),
      arrived_us: nil,
      character: Keyword.get(opts, :character, 2),
      costume: Keyword.get(opts, :costume, 0),
      agent: nil,
      decoder: nil,
      header: [],
      sent: %{},
      landed: 0,
      fresh: 0,
      late: 0,
      late_by: %{},
      unknown: 0,
      delay_mismatch: 0,
      errors: 0
    }
  end

  @impl true
  def pick(_step, state) do
    {%{character: state.character, costume: state.costume}, seat(state)}
  end

  @impl true
  def ready(_step, state), do: seat(state)

  @impl true
  def handle_events(events, state) do
    # Game Start is in the header; it is read once the port is known.
    if state.decoder == nil, do: %{state | header: state.header ++ events}, else: state
  end

  @impl true
  def handle_frame(frame, events, ctx, state) do
    state =
      %{state | arrived_us: System.monotonic_time(:microsecond)}
      |> seat()
      |> decoder(ctx)
      |> check_delay(ctx)

    case Decoder.frame(state.decoder, frame, events) do
      {:ok, decoded, decoder} ->
        state = %{state | decoder: decoder} |> score(frame, decoded.controller)
        decide(decoded, ctx, state)

      {:error, reason} ->
        if state.errors == 0 do
          Logger.warning("[stomped policy] frame #{frame} not decodable: #{inspect(reason)}")
        end

        {%{}, %{state | errors: state.errors + 1}}
    end
  end

  @impl true
  def over(_step, state) do
    total = state.landed + state.late

    Logger.info(
      "[stomped policy] pads landed #{state.landed} (#{state.fresh} on a frame where the pad " <>
        "had changed), late #{state.late}" <>
        if(total > 0, do: " (#{Float.round(100 * state.late / total, 2)}% late)", else: "") <>
        ", unknown (neutral) #{state.unknown}, undecodable frames #{state.errors}" <>
        if(state.late > 0, do: "; late pads read as #{late_by_text(state.late_by)}", else: "") <>
        if(state.delay_mismatch > 0,
          do: "; #{state.delay_mismatch} live pads sent at a lead the agent was not told about",
          else: ""
        )
    )

    if is_pid(state.agent), do: ExPhil.Stomped.Seat.release(state.seat)
    if state.trace, do: File.close(state.trace)
    :ok
  end

  @doc "Landed/late/unknown/delay-mismatch counts so far (for tests and tooling)."
  def stats(state),
    do: Map.take(state, [:landed, :fresh, :late, :late_by, :unknown, :delay_mismatch, :errors])

  @doc """
  Does the lead this frame asks for disagree with the delay the agent plays
  at? `false` when the bot was not told the agent's delay.
  """
  @spec delay_mismatch?(map(), map()) :: boolean()
  def delay_mismatch?(%{reaction_delay: k}, %{reaction_delay: lead_k}) when is_integer(k),
    do: k != lead_k

  def delay_mismatch?(_state, _ctx), do: false

  # Count every live frame whose lead disagrees with the agent's delay and say
  # so once: the fix is a pinned `--lead-extra`, not anything in the match.
  defp check_delay(state, %{live?: true} = ctx) do
    if delay_mismatch?(state, ctx) do
      if state.delay_mismatch == 0 do
        Logger.error(
          "[stomped policy] the match asks for pads at lead #{ctx.lead} (reaction delay " <>
            "#{ctx.reaction_delay}) but the agent plays at reaction delay #{state.reaction_delay}; " <>
            "pin --lead-extra #{state.reaction_delay - ctx.input_delay} " <>
            "(reaction delay - inputDelayFrames #{ctx.input_delay}) so the task matches training"
        )
      end

      %{state | delay_mismatch: state.delay_mismatch + 1}
    else
      state
    end
  end

  defp check_delay(state, _ctx), do: state

  defp late_by_text(late_by) do
    late_by
    |> Enum.sort_by(fn {key, _} -> if is_integer(key), do: key, else: 99 end)
    |> Enum.map_join(", ", fn
      {back, n} when is_integer(back) -> "the pad from #{back} frame(s) earlier x#{n}"
      {:neutral, n} -> "neutral x#{n}"
      {:unmatched, n} -> "no recent pad x#{n}"
    end)
  end

  # --- the agent ----------------------------------------------------------

  # Borrow the agent on first need. A reconnect arrives as a new socket in the
  # `game` step, so this cannot live in `pick` alone.
  defp seat(%{agent: nil} = state) do
    case ExPhil.Stomped.Seat.checkout(state.seat) do
      {:ok, agent} ->
        :ok = Agent.reset_buffer(agent)
        %{state | agent: agent}

      :busy ->
        Logger.warning("[stomped policy] the agent is in another match; playing neutral")
        %{state | agent: :busy}
    end
  end

  defp seat(state), do: state

  defp decoder(%{decoder: nil} = state, ctx) do
    %{state | decoder: ctx.port |> Decoder.new() |> Decoder.game_start(state.header), header: []}
  end

  defp decoder(state, _ctx), do: state

  defp decide(_decoded, _ctx, %{agent: :busy} = state), do: {%{}, state}

  # Catching up through a `replay` answer: advance the agent's history with
  # what was actually played, without spending a decision on a dead frame.
  defp decide(decoded, %{live?: false}, state) do
    Agent.observe(state.agent, decoded.game_state, decoded.controller, player_port: 1)
    {%{}, state}
  end

  defp decide(decoded, ctx, state) do
    case Agent.get_controller(state.agent, decoded.game_state, player_port: 1) do
      {:ok, controller} ->
        inputs = Pad.from_controller(controller)
        {inputs, remember(state, ctx.target, inputs)}

      {:error, reason} ->
        if state.errors == 0 do
          Logger.warning(
            "[stomped policy] no decision at frame #{decoded.frame}: #{inspect(reason)}"
          )
        end

        {%{}, %{state | errors: state.errors + 1}}
    end
  end

  # --- what landed --------------------------------------------------------

  defp remember(state, target, inputs) do
    sent = state.sent |> Map.put(target, inputs) |> Map.delete(target - @sent_window)
    %{state | sent: sent}
  end

  defp score(state, frame, echo) do
    case Map.get(state.sent, frame) do
      nil ->
        state

      inputs ->
        read = Pad.from_controller(echo)

        {outcome, state} =
          cond do
            neutral?(inputs) ->
              {:unknown, %{state | unknown: state.unknown + 1}}

            landed?(inputs, read) ->
              if fresh?(state, frame, read),
                do: {:fresh, %{state | landed: state.landed + 1, fresh: state.fresh + 1}},
                else: {:landed, %{state | landed: state.landed + 1}}

            true ->
              key = late_by(state, frame, read)

              {late_key(key),
               %{
                 state
                 | late: state.late + 1,
                   late_by: Map.update(state.late_by, key, 1, &(&1 + 1))
               }}
          end

        trace(state, frame, outcome, inputs, read)
        state
    end
  end

  defp late_key(back) when is_integer(back), do: :"late_#{back}"
  defp late_key(other), do: other

  # A landed pad proves the frame numbering only when the previous frame's pad
  # would NOT also explain the echo. If the game read every pad one frame
  # later than we assume, "landed" would be exactly the frames where the pad
  # did not change, and this count would stay at zero.
  defp fresh?(state, frame, read) do
    case Map.get(state.sent, frame - 1) do
      nil -> false
      earlier -> not landed?(earlier, read)
    end
  end

  defp open_trace(nil), do: nil

  defp open_trace(path) do
    File.mkdir_p!(Path.dirname(path))
    File.open!(path, [:append, :utf8])
  end

  defp trace(%{trace: nil}, _frame, _outcome, _sent, _echo), do: :ok

  defp trace(state, frame, outcome, sent, echo) do
    IO.write(
      state.trace,
      "#{frame},#{outcome},#{state.arrived_us},#{pad_text(sent)},#{pad_text(echo)}\n"
    )
  end

  defp pad_text(pad) do
    [:buttons, :stick_x, :stick_y, :c_x, :c_y, :l, :r]
    |> Enum.map_join(":", &Map.get(pad, &1, 0))
  end

  # What the game read instead: the pad meant for an earlier frame (the pad
  # arrived that many frames late and the hold-ahead copy stood in), nothing
  # at all, or something that matches no recent pad.
  defp late_by(state, frame, read) do
    cond do
      neutral?(read) ->
        :neutral

      true ->
        Enum.find(1..@late_search, :unmatched, fn back ->
          case Map.get(state.sent, frame - back) do
            nil -> false
            earlier -> landed?(earlier, read)
          end
        end)
    end
  end

  @doc false
  def neutral?(inputs) do
    inputs.buttons == 0 and inputs.l == 0 and inputs.r == 0 and
      expected_stick(inputs.stick_x, inputs.stick_y) == {128, 128} and
      expected_stick(inputs.c_x, inputs.c_y) == {128, 128}
  end

  @doc false
  def landed?(sent, echo) do
    sent.buttons == echo.buttons and
      close?(expected_stick(sent.stick_x, sent.stick_y), {echo.stick_x, echo.stick_y}) and
      close?(expected_stick(sent.c_x, sent.c_y), {echo.c_x, echo.c_y})
  end

  # The stick bytes the echo should show for the bytes we sent: the game
  # clamps the stick to a circle, then zeroes each axis inside the dead zone.
  defp expected_stick(x, y) do
    {ox, oy} = {x - 128, y - 128}
    radius = :math.sqrt(ox * ox + oy * oy)

    {ox, oy} =
      if radius > @stick_radius do
        {trunc(ox * @stick_radius / radius), trunc(oy * @stick_radius / radius)}
      else
        {ox, oy}
      end

    {128 + dead(ox), 128 + dead(oy)}
  end

  defp dead(offset) when abs(offset) < @dead_zone, do: 0
  defp dead(offset), do: offset

  # The echo is the game's float rounded back to a byte: allow one step.
  defp close?({ax, ay}, {bx, by}), do: abs(ax - bx) <= 1 and abs(ay - by) <= 1
end
