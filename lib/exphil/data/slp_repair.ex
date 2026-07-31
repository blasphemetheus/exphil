defmodule ExPhil.Data.SlpRepair do
  @moduledoc """
  Repair truncated `.slp` replays so peppi can parse them.

  A run killed mid-game (the SD flake, a timeout, a crash) leaves a replay
  cut off mid-event: peppi 2.x has no lenient mode and rejects the whole
  file ("failed to fill whole buffer"), silently costing the recording —
  ~20% of eval runs before the 2026-07-28 SD work.

  The fix is mechanical because the format is: a `.slp` is a UBJSON object
  `{ raw: [$U#l <len> <event stream>, metadata: {...} }` where the FIRST
  event (EVENT_PAYLOADS, 0x35) declares the byte size of every other event
  type. So we can walk the stream event by event, cut at the last complete
  one, rewrite the raw length, and close the file with an empty metadata
  element. peppi tolerates a missing GAME_END; it only chokes on a partial
  event.

  `Peppi.parse/2` stays strict; use `parse_lenient/2` (or
  `scripts/repair_slp.exs`) where truncation is expected.
  """

  # {U\x03raw[$U#l  then a 4-byte big-endian length (0 when unfinalized)
  @raw_header <<0x7B, 0x55, 0x03, "raw", 0x5B, 0x24, 0x55, 0x23, 0x6C>>
  @header_size byte_size(@raw_header) + 4

  @event_payloads 0x35

  # U\x08metadata{} }  — minimal valid close for the outer UBJSON object
  @metadata_close <<0x55, 0x08, "metadata", 0x7B, 0x7D, 0x7D>>

  @doc """
  Repair `path` into `out_path` (default: `path <> ".repaired.slp"`).

  Returns `{:ok, out_path, stats}` with `stats.dropped_bytes` /
  `stats.events` / `stats.raw_bytes`, or `{:error, reason}` for files that
  do not look like Slippi replays at all.
  """
  @spec repair(Path.t(), Path.t() | nil) :: {:ok, Path.t(), map()} | {:error, term()}
  def repair(path, out_path \\ nil) do
    out_path = out_path || path <> ".repaired.slp"

    with {:ok, bin} <- File.read(path),
         {:ok, sizes, stream_start} <- payload_sizes(bin) do
      raw_end = declared_raw_end(bin)
      {last_complete, events} = walk(bin, sizes, stream_start, raw_end, 0, stream_start)

      raw_len = last_complete - @header_size

      repaired =
        binary_part(bin, 0, @header_size - 4) <>
          <<raw_len::unsigned-big-32>> <>
          binary_part(bin, @header_size, raw_len) <>
          @metadata_close

      case File.write(out_path, repaired) do
        :ok ->
          {:ok, out_path,
           %{
             raw_bytes: raw_len,
             events: events,
             dropped_bytes: byte_size(bin) - last_complete
           }}

        error ->
          error
      end
    end
  end

  @doc """
  `ExPhil.Data.Peppi.parse/2` with truncation tolerance: strict parse
  first; on failure, repair to a temp file, parse that, and clean up.
  """
  @spec parse_lenient(Path.t(), keyword()) :: {:ok, term()} | {:error, term()}
  def parse_lenient(path, opts \\ []) do
    case ExPhil.Data.Peppi.parse(path, opts) do
      {:ok, parsed} ->
        {:ok, parsed}

      {:error, strict_error} ->
        tmp = Path.join(System.tmp_dir!(), "slp_repair_#{:erlang.unique_integer([:positive])}.slp")

        try do
          case repair(path, tmp) do
            {:ok, ^tmp, _stats} -> ExPhil.Data.Peppi.parse(tmp, opts)
            {:error, _repair_error} -> {:error, strict_error}
          end
        after
          File.rm(tmp)
        end
    end
  end

  # ---------------------------------------------------------------------------

  # EVENT_PAYLOADS: 0x35, u8 size s (counts itself), then (code u8, len u16be)
  # triplets in s-1 bytes. Returns the code=>len map and the offset of the
  # first real event.
  defp payload_sizes(<<@raw_header, _len::32, @event_payloads, s, rest::binary>>)
       when byte_size(rest) >= s - 1 do
    triplets = binary_part(rest, 0, s - 1)

    sizes =
      for <<code, len::unsigned-big-16 <- triplets>>, into: %{} do
        {code, len}
      end

    {:ok, sizes, @header_size + 1 + s}
  end

  defp payload_sizes(_), do: {:error, :not_a_slippi_replay}

  # Where does the declared raw element end? 0 = unfinalized (read to EOF).
  defp declared_raw_end(<<_::binary-size(@header_size - 4), len::unsigned-big-32, _::binary>> = bin) do
    if len == 0, do: byte_size(bin), else: min(@header_size + len, byte_size(bin))
  end

  # FRAME_BOOKEND — the "this frame is finalized" event. Cutting anywhere
  # else can leave a partial frame (pre without post) that panics peppi's
  # frame assembly; the cut point is therefore the end of the LAST bookend,
  # not merely the last complete event.
  @frame_bookend 0x3C

  # Walk complete events; stop at a partial event, an unknown code (the
  # metadata boundary on finalized files), or the declared raw end. Returns
  # {cut_point, complete_events_before_cut} where cut_point is just after
  # the last frame bookend (or the stream start if none was seen — e.g. a
  # replay cut during the very first frame, which has no salvageable data).
  defp walk(bin, sizes, pos, raw_end, events, last_bookend_end) do
    with true <- pos < min(raw_end, byte_size(bin)),
         <<_::binary-size(pos), code, _::binary>> <- bin,
         {:ok, len} <- Map.fetch(sizes, code),
         true <- pos + 1 + len <= min(raw_end, byte_size(bin)) do
      next = pos + 1 + len
      bookend_end = if code == @frame_bookend, do: next, else: last_bookend_end
      walk(bin, sizes, next, raw_end, events + 1, bookend_end)
    else
      _ -> {last_bookend_end, events}
    end
  end

  # FRAME_START — where a frame's event group begins; the only safe place
  # to resume after a mid-stream gap (resuming mid-frame leaves a pre
  # without its post and panics peppi's frame assembly).
  @frame_start 0x3A

  @doc """
  Like `repair/2` but tolerates MID-STREAM corruption, not just truncation:
  short zero/garbage gaps inside the event stream (observed 2026-07-31 on
  mainline-beta ONLINE replays — a ~22-byte zero gap ~61KB in) are excised
  and the walk resyncs at the next FRAME_START whose following event also
  parses. Complete frames on both sides of each gap are kept.
  """
  @spec repair_gaps(Path.t(), Path.t() | nil) :: {:ok, Path.t(), map()} | {:error, term()}
  def repair_gaps(path, out_path \\ nil) do
    out_path = out_path || path <> ".repaired.slp"

    with {:ok, bin} <- File.read(path),
         {:ok, sizes, stream_start} <- payload_sizes(bin) do
      raw_end = declared_raw_end(bin)

      {segments, events, gaps} =
        walk_segments(bin, sizes, stream_start, raw_end, stream_start, stream_start, [], 0, 0)

      spliced =
        segments
        |> Enum.reject(fn {s, e} -> e <= s end)
        |> Enum.map_join(fn {s, e} -> binary_part(bin, s, e - s) end)

      # The payload-declaration event must lead the stream: splice it back
      # in front of the recovered segments (segments start AFTER it).
      head = binary_part(bin, @header_size, stream_start - @header_size)
      raw = head <> spliced
      raw_len = byte_size(raw)

      repaired =
        binary_part(bin, 0, @header_size - 4) <>
          <<raw_len::unsigned-big-32>> <> raw <> @metadata_close

      case File.write(out_path, repaired) do
        :ok ->
          {:ok, out_path,
           %{raw_bytes: raw_len, events: events, gaps: gaps,
             dropped_bytes: byte_size(bin) - raw_len - @header_size}}

        error ->
          error
      end
    end
  end

  defp walk_segments(bin, sizes, pos, raw_end, seg_start, last_bookend, acc, events, gaps) do
    limit = min(raw_end, byte_size(bin))

    with true <- pos < limit,
         <<_::binary-size(pos), code, _::binary>> <- bin,
         {:ok, len} <- Map.fetch(sizes, code),
         true <- pos + 1 + len <= limit do
      next = pos + 1 + len
      bookend = if code == @frame_bookend, do: next, else: last_bookend
      walk_segments(bin, sizes, next, raw_end, seg_start, bookend, acc, events + 1, gaps)
    else
      _ ->
        acc = [{seg_start, last_bookend} | acc]

        case resync(bin, sizes, pos + 1, limit) do
          nil ->
            {Enum.reverse(acc), events, gaps}

          p ->
            walk_segments(bin, sizes, p, raw_end, p, p, acc, events, gaps + 1)
        end
    end
  end

  # Scan forward for a FRAME_START whose immediately following event also
  # parses — two-event validation keeps us from resyncing into garbage
  # that merely starts with the right byte.
  defp resync(bin, sizes, pos, limit) when pos < limit do
    with <<_::binary-size(pos), @frame_start, _::binary>> <- bin,
         {:ok, len} <- Map.fetch(sizes, @frame_start),
         next = pos + 1 + len,
         true <- next < limit,
         <<_::binary-size(next), code2, _::binary>> <- bin,
         {:ok, _} <- Map.fetch(sizes, code2) do
      pos
    else
      _ -> resync(bin, sizes, pos + 1, limit)
    end
  end

  defp resync(_bin, _sizes, _pos, _limit), do: nil
end
