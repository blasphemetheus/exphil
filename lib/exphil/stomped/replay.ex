defmodule ExPhil.Stomped.Replay do
  @moduledoc """
  Turns a stored `.slp` file back into the binary messages the stomped.dev
  arena sent during that game.

  The arena's stream is byte-identical to the replay's `raw` block: the first
  message is everything before the first frame, then one message per frame
  (through its Frame Bookend), then Game End. Re-cutting a file at those
  points gives a faithful offline stand-in for the live socket — the parity
  tests and the offline arena client both read games this way.
  """

  alias ExPhil.Stomped.Events

  # UBJSON: `{` `U` 3 "raw" `[` `$` `U` `#` `l` <i32 length> <bytes>
  @raw_prefix <<"{U", 3, "raw[$U#l">>

  @doc "The `raw` event block of a `.slp` file's bytes."
  @spec raw(binary()) :: {:ok, binary()} | {:error, :not_slp | :truncated}
  def raw(<<@raw_prefix, length::big-32, rest::binary>>) do
    cond do
      # A recording that never closed has length 0: the block runs to the end.
      length == 0 -> {:ok, rest}
      byte_size(rest) >= length -> {:ok, binary_part(rest, 0, length)}
      true -> {:error, :truncated}
    end
  end

  def raw(_), do: {:error, :not_slp}

  @doc "Read a `.slp` file and cut it into the arena's messages."
  @spec read_messages(Path.t()) :: {:ok, [binary()]} | {:error, term()}
  def read_messages(path) do
    with {:ok, bytes} <- File.read(path),
         {:ok, raw} <- raw(bytes) do
      messages(raw)
    end
  end

  @doc """
  Cut a `raw` block into messages: the header, one per frame, then Game End.
  """
  @spec messages(binary()) :: {:ok, [binary()]} | {:error, term()}
  def messages(raw) do
    if Events.header?(raw) do
      <<_cmd, n, _::binary>> = raw

      case Events.parse(binary_part(raw, 0, 1 + n), %{}) do
        {:ok, _events, sizes} -> cut(raw, 1 + n, 0, sizes, [])
        {:partial, _events, _sizes, reason} -> {:error, reason}
      end
    else
      {:error, :no_event_payloads}
    end
  end

  # `start` is where the current message began, `at` the next event to read.
  defp cut(raw, at, start, _sizes, acc) when at >= byte_size(raw) do
    acc = if at > start, do: [binary_part(raw, start, byte_size(raw) - start) | acc], else: acc
    {:ok, Enum.reverse(acc)}
  end

  defp cut(raw, at, start, sizes, acc) do
    cmd = :binary.at(raw, at)
    frame_start = Events.frame_start()
    frame_bookend = Events.frame_bookend()
    game_end = Events.game_end()

    case sizes do
      %{^cmd => size} when at + 1 + size <= byte_size(raw) ->
        next = at + 1 + size

        cond do
          # The header message ends where the first frame begins.
          cmd == frame_start and at > start and not frame_open?(raw, start, frame_start) ->
            cut(raw, at, at, sizes, [binary_part(raw, start, at - start) | acc])

          cmd == frame_bookend ->
            cut(raw, next, next, sizes, [binary_part(raw, start, next - start) | acc])

          cmd == game_end ->
            acc = if at > start, do: [binary_part(raw, start, at - start) | acc], else: acc
            cut(raw, next, next, sizes, [binary_part(raw, at, next - at) | acc])

          true ->
            cut(raw, next, start, sizes, acc)
        end

      %{^cmd => _} ->
        {:error, {:truncated_payload, cmd}}

      _ ->
        {:error, {:unknown_command, cmd}}
    end
  end

  # A message that already begins with Frame Start is a frame in progress.
  defp frame_open?(raw, start, frame_start), do: :binary.at(raw, start) == frame_start
end
