defmodule ExPhil.Stomped.Events do
  @moduledoc """
  Framing for the raw `.slp` event stream the stomped.dev arena sends.

  A binary message is a run of events: a command byte, then that command's
  payload. Payload sizes are not fixed by the format — the stream opens with
  the Event Payloads event (`0x35`), which lists the size of every other
  command, so a parser has to carry that table for the rest of the game.

  This module only splits messages into `{command, payload}` pairs and reads
  the few fields the socket itself needs (the frame number that closes a
  frame). Turning payloads into a game state is the decoder's job.
  """

  @event_payloads 0x35
  @game_start 0x36
  @pre_frame 0x37
  @post_frame 0x38
  @game_end 0x39
  @frame_start 0x3A
  @item_update 0x3B
  @frame_bookend 0x3C
  @message_splitter 0x10

  @type sizes :: %{optional(byte()) => non_neg_integer()}
  @type event :: {command :: byte(), payload :: binary()}

  def event_payloads, do: @event_payloads
  def game_start, do: @game_start
  def pre_frame, do: @pre_frame
  def post_frame, do: @post_frame
  def game_end, do: @game_end
  def frame_start, do: @frame_start
  def item_update, do: @item_update
  def frame_bookend, do: @frame_bookend
  def message_splitter, do: @message_splitter

  @doc "True when the message opens with the stream header (Event Payloads)."
  @spec header?(binary()) :: boolean()
  def header?(<<@event_payloads, _::binary>>), do: true
  def header?(_), do: false

  @doc """
  Split one binary message into events, in stream order.

  Returns the events and the size table (updated when the message carries an
  Event Payloads event). `{:partial, ...}` means the message stopped being
  readable part-way: a command with no known size, or a payload cut short.
  The events before that point are still returned.
  """
  @spec parse(binary(), sizes()) ::
          {:ok, [event()], sizes()}
          | {:partial, [event()], sizes(), reason :: term()}
  def parse(message, sizes) when is_binary(message) and is_map(sizes) do
    do_parse(message, sizes, [])
  end

  defp do_parse(<<>>, sizes, acc), do: {:ok, Enum.reverse(acc), sizes}

  # Event Payloads: a size byte that counts itself, then (command, u16 size) triples.
  defp do_parse(<<@event_payloads, n, rest::binary>>, _sizes, acc)
       when n >= 1 and byte_size(rest) >= n - 1 do
    table_bytes = n - 1
    <<table::binary-size(table_bytes), rest::binary>> = rest
    sizes = for <<cmd, size::big-16 <- table>>, into: %{}, do: {cmd, size}
    do_parse(rest, sizes, [{@event_payloads, table} | acc])
  end

  defp do_parse(<<@event_payloads, _::binary>>, sizes, acc) do
    {:partial, Enum.reverse(acc), sizes, :truncated_event_payloads}
  end

  defp do_parse(<<cmd, rest::binary>>, sizes, acc) do
    case sizes do
      %{^cmd => size} when byte_size(rest) >= size ->
        <<payload::binary-size(size), rest::binary>> = rest
        do_parse(rest, sizes, [{cmd, payload} | acc])

      %{^cmd => _size} ->
        {:partial, Enum.reverse(acc), sizes, {:truncated_payload, cmd}}

      _ ->
        {:partial, Enum.reverse(acc), sizes, {:unknown_command, cmd}}
    end
  end

  @doc """
  Group a message's events into frames.

  Returns `{frames, leftover}`: each frame is `{frame_number, events}` with
  the events from the previous bookend up to and including this frame's
  Frame Bookend; `leftover` is whatever follows the last bookend (the whole
  message for the header and for Game End).
  """
  @spec frames([event()]) :: {[{integer(), [event()]}], [event()]}
  def frames(events) do
    {frames, pending} =
      Enum.reduce(events, {[], []}, fn
        {@frame_bookend, <<frame::big-signed-32, _::binary>>} = event, {frames, pending} ->
          {[{frame, Enum.reverse([event | pending])} | frames], []}

        event, {frames, pending} ->
          {frames, [event | pending]}
      end)

    {Enum.reverse(frames), Enum.reverse(pending)}
  end
end
