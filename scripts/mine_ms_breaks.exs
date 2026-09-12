# Mine multishine CHAIN BREAKS from rollout replays into a scenario manifest
# (closed-loop correction validation, HANDOFF_2026-09-12 §3 / Astra's note).
#
# A break = the subject was in the shine family (ground/air reflector) and
# then spent >= --gap frames without re-entering it. The manifest handoff is
# placed --lead frames BEFORE the last shine-family frame, so the driver
# (scenario_suite.exs --driver teacher|policy|neutral) takes port 1 at the
# decision that failed, with the recorded history intact.
#
#   mix run scripts/mine_ms_breaks.exs --replays "glob,glob" \
#     [--port 1] [--gap 30] [--lead 6] [--min-frame 300] [--max-per-replay 6] \
#     [--out scenarios/ms_breaks_manifest.json]

alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, gap: :integer, lead: :integer, min_frame: :integer, max_per_replay: :integer, out: :string]
  )

globs = opts[:replays] || raise "--replays required"
port = opts[:port] || 1
gap = opts[:gap] || 30
lead = opts[:lead] || 6
min_frame = opts[:min_frame] || 300
max_per = opts[:max_per_replay] || 6
out = opts[:out] || "scenarios/ms_breaks_manifest.json"

paths = globs |> String.split(",", trim: true) |> Enum.flat_map(&Path.wildcard/1) |> Enum.map(&Path.expand/1)

shine? = fn a -> ShineChain.family(a) in [:ground_reflect, :air_reflect] end

entries =
  Enum.flat_map(paths, fn path ->
    case ExPhil.Data.Peppi.parse(path, player_port: port) do
      {:ok, replay} ->
        frames =
          replay.frames
          |> Enum.map(fn f -> {f.frame_number, f.players[port] && trunc(f.players[port].action || 0)} end)
          |> Enum.reject(fn {_, a} -> is_nil(a) end)

        # frame numbers where the subject leaves the shine family and stays out >= gap
        {breaks, _} =
          frames
          |> Enum.chunk_every(gap + 1, 1, :discard)
          |> Enum.reduce({[], -1_000}, fn [{fnum, a} | rest], {acc, last_break} ->
            was_shine = shine?.(a)
            next_a = rest |> List.first() |> elem(1)

            out_for_gap =
              was_shine and not shine?.(next_a) and
                not Enum.any?(rest, fn {_, b} -> shine?.(b) end)

            if out_for_gap and fnum >= min_frame and fnum - last_break > gap,
              do: {[fnum | acc], fnum},
              else: {acc, last_break}
          end)

        breaks
        |> Enum.reverse()
        |> Enum.take(max_per)
        |> Enum.map(fn fnum ->
          %{
            "slp" => Path.relative_to_cwd(path),
            "frame" => fnum - lead,
            "type" => "multishine_reentry",
            "note" => "chain break: last shine-family frame #{fnum}, no shine for #{gap}f; handoff #{lead}f earlier"
          }
        end)

      _ ->
        []
    end
  end)

File.mkdir_p!(Path.dirname(out))

File.write!(
  out,
  Jason.encode!(
    %{
      "comment" => "Multishine chain-break handoffs mined by scripts/mine_ms_breaks.exs (#{Date.utc_today()}); gap #{gap}, lead #{lead}",
      "entries" => entries
    },
    pretty: true
  )
)

Output.success("#{length(entries)} break moments from #{length(paths)} replay(s) -> #{out}")
