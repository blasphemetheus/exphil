defmodule ExPhil.Stomped.Bots.Probe do
  @moduledoc """
  A bot that stamps every pad with the frame it is meant for, so the frame
  the game actually applied it at can be read back from the replay.

  The pad for target frame T puts the main stick at a raw byte that encodes
  `rem(T, 50)`; nothing else is pressed. Each frame's pre-frame controller
  event echoes the stick the game read, so decoding it gives the target frame
  of the pad the game applied, modulo 50, and the difference from the frame
  number is the number of frames the pad was applied late:

      offset 0  the pad for T was polled at T          (the spec's rule)
      offset 1  the pad for T - 1 was polled at T      (applied a frame late)

  The policy bot can only tell those two apart when consecutive pads differ;
  here every pad differs from its 49 neighbours, so the reading is exact at
  every frame. `scripts/stomped_probe_report.exs` reads a probe match's
  replay (or this bot's own trace) and prints the offset per frame and its
  time structure. `ExPhil.Stomped.ProbeReplayTest` is the failing test: it
  asserts offset 0 at every frame of a probe replay.

  The character dashes left and right and will lose; that is fine, a probe
  match is a measurement, and it needs to last a few of the 16.7 s cycles
  the measurement is for (4 stocks usually do).

  Options: `:character`, `:costume` (default Fox), `:trace` (file path for
  `frame,offset,arrival_us` lines, the same shape as the policy bot's).
  """

  @behaviour ExPhil.Stomped.Bot

  alias ExPhil.Stomped.{Decoder, Pad}

  require Logger

  # Codes sit above the 23-unit dead zone and inside the 80-unit circle.
  @modulus 50
  @first 24
  @centre 128
  @codes Enum.map(@first..(@first + @modulus - 1), &(&1 + @centre))

  @impl true
  def init(opts) do
    %{
      character: Keyword.get(opts, :character, 2),
      costume: Keyword.get(opts, :costume, 0),
      trace: open_trace(Keyword.get(opts, :trace)),
      decoder: nil,
      header: [],
      offsets: %{},
      neutral: 0,
      frames: 0
    }
  end

  @impl true
  def pick(_step, state), do: {%{character: state.character, costume: state.costume}, state}

  @impl true
  def handle_events(events, state) do
    if state.decoder == nil, do: %{state | header: state.header ++ events}, else: state
  end

  @impl true
  def handle_frame(frame, events, ctx, state) do
    state = decoder(state, ctx)

    state =
      case Decoder.frame(state.decoder, frame, events) do
        {:ok, decoded, decoder} ->
          %{state | decoder: decoder} |> record(frame, offset(decoded.controller, frame))

        {:error, _} ->
          state
      end

    {pad(ctx.target), state}
  end

  @impl true
  def over(_step, state) do
    Logger.info("[stomped probe] " <> summary_text(state))
    if state.trace, do: File.close(state.trace)
    :ok
  end

  @doc "The pad stamped for target frame `t`: main stick x encodes `rem(t, 50)`."
  @spec pad(integer()) :: map()
  def pad(t), do: %{stick_x: @centre + @first + Integer.mod(t, @modulus), stick_y: @centre}

  @doc """
  How many frames late the pad echoed at `frame` was applied: 0 when it is
  the pad stamped for `frame`, `:neutral` when nothing of ours was read, and
  `:other` for a stick byte outside the code range.
  """
  @spec offset(ExPhil.Bridge.ControllerState.t() | map(), integer()) ::
          non_neg_integer() | :neutral | :other
  def offset(%{stick_x: byte}, frame) do
    cond do
      byte == @centre ->
        :neutral

      byte in @codes ->
        decode(byte, frame)

      true ->
        :other
    end
  end

  def offset(controller, frame), do: offset(Pad.from_controller(controller), frame)

  defp decode(byte, frame),
    do: Integer.mod(Integer.mod(frame, @modulus) - (byte - @centre - @first), @modulus)

  @doc "Offsets seen so far, as counts, plus the neutral reads."
  def stats(state), do: %{offsets: state.offsets, neutral: state.neutral, frames: state.frames}

  defp record(state, frame, result) do
    if state.trace,
      do: IO.write(state.trace, "#{frame},#{result},#{System.monotonic_time(:microsecond)}\n")

    state = %{state | frames: state.frames + 1}

    case result do
      :neutral -> %{state | neutral: state.neutral + 1}
      :other -> state
      n -> %{state | offsets: Map.update(state.offsets, n, 1, &(&1 + 1))}
    end
  end

  defp summary_text(state) do
    scored = state.offsets |> Map.values() |> Enum.sum()

    offsets =
      state.offsets
      |> Enum.sort()
      |> Enum.map_join(", ", fn {k, v} -> "#{v} applied #{k} frame(s) late" end)

    "#{state.frames} frames; #{scored} echoed one of our pads: #{offsets}; #{state.neutral} read neutral"
  end

  defp decoder(%{decoder: nil} = state, ctx) do
    %{state | decoder: ctx.port |> Decoder.new() |> Decoder.game_start(state.header), header: []}
  end

  defp decoder(state, _ctx), do: state

  defp open_trace(nil), do: nil

  defp open_trace(path) do
    File.mkdir_p!(Path.dirname(path))
    File.open!(path, [:append, :utf8])
  end
end
