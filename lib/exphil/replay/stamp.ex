defmodule ExPhil.Replay.Stamp do
  @moduledoc """
  Tag bot-produced `.slp` replays so they are identifiable in a corpus.

  Local (offline) games record no player names at all — the replay's
  metadata block carries an empty `names{}` for every player, and the
  in-game nametag is only present if one was selected at the character
  select screen. Stamping writes a name into that metadata slot after the
  game, which every standard tool (Slippi Launcher, peppi, our own
  parser) reads back as the player's netplay name.

  The metadata block is UBJSON with terminator-delimited objects, so
  inserting the name needs no offset fixups; the `raw` element (which IS
  length-prefixed) is never touched.

      iex> ExPhil.Replay.Stamp.stamp_file("game.slp", %{0 => "exph"})
      {:ok, 1}

  Ports are ZERO-indexed here, matching the replay's own metadata keys:
  port 1 is `0`.
  """

  @type port_index :: 0..3
  @type tags :: %{port_index() => String.t()}

  @doc """
  Stamp one replay in place. `tags` maps zero-indexed ports to names.

  Only players whose names are currently empty are stamped, so a real
  netplay name is never clobbered. Returns the number of players stamped.
  """
  @spec stamp_file(Path.t(), tags()) :: {:ok, non_neg_integer()} | {:error, term()}
  def stamp_file(path, tags) do
    with {:ok, content} <- File.read(path),
         {:ok, stamped, count} <- stamp_binary(content, tags),
         :ok <- maybe_write(path, stamped, count) do
      {:ok, count}
    end
  end

  @doc """
  Stamp every `.slp` under `dir` (recursively).

  Returns `{stamped_files, skipped_files}`.
  """
  @spec stamp_dir(Path.t(), tags()) :: {non_neg_integer(), non_neg_integer()}
  def stamp_dir(dir, tags) do
    dir
    |> Path.join("**/*.slp")
    |> Path.wildcard()
    |> Enum.reduce({0, 0}, fn path, {stamped, skipped} ->
      case stamp_file(path, tags) do
        {:ok, n} when n > 0 -> {stamped + 1, skipped}
        _ -> {stamped, skipped + 1}
      end
    end)
  end

  @doc """
  Read back the netplay names recorded in a replay's metadata.

  Used to verify a stamp round-trips.
  """
  @spec names(Path.t()) :: %{port_index() => String.t()}
  def names(path) do
    case File.read(path) do
      {:ok, content} -> names_from_binary(content)
      _ -> %{}
    end
  end

  ## Internals

  @doc false
  @spec stamp_binary(binary(), tags()) :: {:ok, binary(), non_neg_integer()} | {:error, term()}
  def stamp_binary(content, tags) do
    if :binary.match(content, "metadata") == :nomatch do
      {:error, :no_metadata}
    else
      {stamped, count} =
        Enum.reduce(tags, {content, 0}, fn {port, name}, {acc, count} ->
          empty = empty_names(port)

          case :binary.match(acc, empty) do
            :nomatch ->
              {acc, count}

            {idx, len} ->
              replacement = stamped_names(port, name)

              acc =
                binary_part(acc, 0, idx) <>
                  replacement <>
                  binary_part(acc, idx + len, byte_size(acc) - idx - len)

              {acc, count + 1}
          end
        end)

      {:ok, stamped, count}
    end
  end

  # UBJSON: `U<len>"<key>"` for keys, `S U<len>"<value>"` for strings,
  # `{`/`}` delimit objects.
  defp empty_names(port) do
    <<0x55, 1, ?0 + port, ?{, 0x55, 5, "names", ?{, ?}>>
  end

  defp stamped_names(port, name) do
    <<0x55, 1, ?0 + port, ?{, 0x55, 5, "names", ?{, 0x55, 7, "netplay", ?S, 0x55,
      byte_size(name), name::binary, ?}>>
  end

  defp maybe_write(_path, _content, 0), do: :ok
  defp maybe_write(path, content, _count), do: File.write(path, content)

  defp names_from_binary(content) do
    for port <- 0..3, into: %{} do
      prefix = <<0x55, 1, ?0 + port, ?{, 0x55, 5, "names", ?{, 0x55, 7, "netplay", ?S, 0x55>>

      case :binary.match(content, prefix) do
        :nomatch ->
          {port, ""}

        {idx, len} ->
          <<name_len>> = binary_part(content, idx + len, 1)
          {port, binary_part(content, idx + len + 1, name_len)}
      end
    end
  end
end
