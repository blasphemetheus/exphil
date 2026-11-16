#!/usr/bin/env elixir
# Read a probe match (ExPhil.Stomped.Bots.Probe) and print, per frame, how
# many frames late the game applied our pad, and the time structure of it.
#
#   mix run scripts/stomped_probe_report.exs path/to/match.slp [port]
#
# The replay is the arena's own record, so this needs nothing from our side
# but the encoding: the pad stamped for frame T carries rem(T, 50) in the
# main stick's x byte. Port is our Slippi player index (0 or 1); when
# omitted, the port whose stick bytes decode as probe codes is used.
alias ExPhil.Stomped.{Decoder, Events, Replay}
alias ExPhil.Stomped.Bots.Probe

[path | rest] = System.argv()

decode = fn port ->
  {:ok, [header | messages]} = Replay.read_messages(path)
  {:ok, header_events, sizes} = Events.parse(header, %{})
  decoder = port |> Decoder.new() |> Decoder.game_start(header_events)

  {out, _} =
    Enum.reduce(messages, {[], decoder}, fn message, {acc, decoder} ->
      {:ok, events, _} = Events.parse(message, sizes)

      case Events.frames(events) do
        {[{frame, frame_events}], []} ->
          {:ok, decoded, decoder} = Decoder.frame(decoder, frame, frame_events)
          {[{frame, Probe.offsets(decoded.controller, frame)} | acc], decoder}

        _ ->
          {acc, decoder}
      end
    end)

  Enum.reverse(out)
end

port =
  case rest do
    [p] ->
      String.to_integer(p)

    [] ->
      Enum.max_by([0, 1], fn p -> Enum.count(decode.(p), fn {_, o} -> is_integer(o.main) end) end)
  end

all = decode.(port)

# The other encoded fields, when the probe used them (fields: :all).
for field <- [:c, :button, :trigger] do
  per = Enum.map(all, fn {f, o} -> {f, o[field]} end)
  scored = Enum.filter(per, fn {_, o} -> is_integer(o) end)

  if length(scored) > 100 do
    counts = scored |> Enum.map(&elem(&1, 1)) |> Enum.frequencies() |> Enum.sort()

    IO.puts(
      "#{field}: #{length(scored)} frames echoed a code; " <>
        Enum.map_join(counts, ", ", fn {k, n} ->
          "#{n} applied #{k} late (#{Float.round(100 * n / length(scored), 1)}%)"
        end)
    )
  end
end

offsets = Enum.map(all, fn {f, o} -> {f, o.main} end)
scored = Enum.filter(offsets, fn {_, o} -> is_integer(o) end)
counts = scored |> Enum.map(&elem(&1, 1)) |> Enum.frequencies() |> Enum.sort()

IO.puts(
  "#{Path.basename(path)}, port #{port}: #{length(offsets)} frames, #{length(scored)} echoed a probe pad"
)

for {k, n} <- counts do
  IO.puts("  applied #{k} frame(s) late: #{n} (#{Float.round(100 * n / length(scored), 1)}%)")
end

IO.puts(
  "  neutral: #{Enum.count(offsets, fn {_, o} -> o == :neutral end)}, other: #{Enum.count(offsets, fn {_, o} -> o == :other end)}"
)

# Runs of frames applied late, and how often they start (frames between starts).
late = Enum.map(scored, fn {_, o} -> o > 0 end)
runs = late |> Enum.chunk_by(& &1) |> Enum.filter(&hd(&1)) |> Enum.map(&length/1)

if runs != [] do
  IO.puts(
    "  late runs: #{length(runs)}, mean #{Float.round(Enum.sum(runs) / length(runs), 1)} frames, max #{Enum.max(runs)}"
  )

  starts =
    scored
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [{_, a}, {_, b}] -> a == 0 and b > 0 end)
    |> Enum.map(fn [_, {f, _}] -> f end)

  gaps =
    starts
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [a, b] -> b - a end)
    |> Enum.filter(&(&1 > 100))

  if gaps != [] do
    med = Enum.at(Enum.sort(gaps), div(length(gaps), 2))

    IO.puts(
      "  late stretches start every #{med} frames (median; #{Float.round(med / 59.94, 1)} s at 59.94 Hz) over #{length(gaps)} cycles"
    )
  end
end

IO.puts("\nper 60 frames, share applied late:")

scored
|> Enum.chunk_every(60)
|> Enum.map(fn chunk ->
  round(100 * Enum.count(chunk, fn {_, o} -> o > 0 end) / length(chunk))
end)
|> Enum.chunk_every(30)
|> Enum.each(fn row ->
  IO.puts("  " <> Enum.map_join(row, " ", &String.pad_leading("#{&1}", 3)))
end)
