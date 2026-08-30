# E2 — rare-event coverage: how many EXPERT examples exist of the behaviours
# the bot lacks, and how many of them the training recipe actually presents
# as targets (EVAL_DIRECTIONS E2).
#
# Behaviours (from A1, eval_runs/0829_situation_hist):
#   dash_init      standing/walk/turn -> DASHING next frame (the flick)
#   dashdance      >= 3 dash flips within 24 f (Options :dashdance)
#   throw_from_grab GRAB_WAIT/PUMMEL -> THROW_* (the pummel-loop answer)
#   grab_from_dash  DASHING/RUNNING -> GRAB_RUNNING (grab out of movement)
#   upb_recovery    FIREFOX start while offstage / below ledge
#   sideb_recovery  ILLUSION start while offstage
#   airdodge_offstage  AIRDODGE while offstage (the bot's substitute)
#
# Recipe presentation (fox_gen_v1: temporal window 60, stride 5): a window's
# label is its LAST frame, so a single-frame event is a training target only
# when its index lands on the stride grid ≈ 1 in 5. Reported as
# "targets/epoch ≈ count / stride". Sampling weights are NOT modelled here.
#
#   mix run scripts/rare_event_coverage.exs --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --port 1 --stride 5 --out eval_runs/0829_rare_events/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, limit_files: :integer, stride: :integer,
             concurrency: :integer, out: :string]
  )

glob = opts[:replays] || raise("--replays required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
stride = opts[:stride] || 5
files = glob |> Path.wildcard() |> Enum.sort()
files = if opts[:limit_files], do: Enum.take(files, opts[:limit_files]), else: files
if files == [], do: raise("no replays matched")

Output.banner("Rare-event coverage (E2)")
Output.config([{"Replays", "#{length(files)} files"}, {"Port", port}, {"Stride", stride}])

dashing = 20
running = 21
standing_ish = [14, 15, 16, 17, 18]
grab_hold = [216, 217]
throws = [219, 220, 221, 222]
grab_running = 214
firefox = [353, 354, 355, 356]
illusion = [350, 351, 352]
airdodge = 236

scan = fn path ->
  try do
    with {:ok, replay} <- Peppi.parse(path, player_port: port) do
      states =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.map(& &1.game_state)

      n = length(states)

      if n < 300 do
        nil
      else
        arr = List.to_tuple(states)
        sits = Situations.label_states(states, port, as: :set) |> List.to_tuple()
        act = fn i -> trunc(elem(arr, i).players[port].action || 0) end
        off = fn i -> MapSet.member?(elem(sits, i), :offstage) or MapSet.member?(elem(sits, i), :below_ledge) end
        events = Options.events(states, port)
        dd = Enum.count(events, &(&1.option == :dashdance))

        counts =
          Enum.reduce(1..(n - 1), %{}, fn i, acc ->
            a = act.(i)
            pa = act.(i - 1)
            bump = fn acc, k, cond? -> if cond?, do: Map.update(acc, k, 1, &(&1 + 1)), else: acc end

            acc
            |> bump.(:dash_init, a == dashing and pa in standing_ish)
            |> bump.(:throw_from_grab, a in throws and pa in grab_hold)
            |> bump.(:grab_from_dash, a == grab_running and pa in [dashing, running])
            |> bump.(:upb_recovery, a in firefox and pa not in firefox and off.(i))
            |> bump.(:sideb_recovery, a in illusion and pa not in illusion and off.(i))
            |> bump.(:airdodge_offstage, a == airdodge and pa != airdodge and off.(i))
            |> bump.(:grab_hold_frames, a in grab_hold)
            |> bump.(:grab_entries, a in [212, 214] and pa not in [212, 213, 214, 215])
          end)

        Map.merge(counts, %{dashdance: dd, frames: n, files: 1})
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

total =
  files
  |> Task.async_stream(scan, max_concurrency: opts[:concurrency] || 12, timeout: :infinity, ordered: false)
  |> Stream.with_index(1)
  |> Enum.reduce(%{}, fn {{:ok, r}, i}, acc ->
    if rem(i, 200) == 0, do: Output.progress_bar(i, length(files), label: "replays")
    if r, do: Map.merge(acc, r, fn _, x, y -> x + y end), else: acc
  end)

Output.progress_done()

g = fn k -> Map.get(total, k, 0) end
games = g.(:files)
mins = g.(:frames) / 3600
f1 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 1) end

row = fn label, k ->
  c = g.(k)
  "| #{label} | #{c} | #{f1.(c / max(games, 1))} | #{f1.(c / max(mins, 1))} | #{div(c, stride)} |"
end

report = """
# Rare-event coverage (E2) — #{glob}

#{games} games, #{f1.(mins)} minutes of play, subject port #{port}. Temporal recipe
stride #{stride} → a single-frame event is a window label ≈ 1 in #{stride} times.

| behaviour | count | per game | per minute | ≈ targets / epoch (÷#{stride}) |
|---|---:|---:|---:|---:|
#{row.("dash initiation (stand/walk/turn → DASH)", :dash_init)}
#{row.("dash-dance bursts (≥3 flips / 24 f)", :dashdance)}
#{row.("grab out of dash/run", :grab_from_dash)}
#{row.("grab entries (any)", :grab_entries)}
#{row.("throw from a held grab", :throw_from_grab)}
#{row.("up-B (firefox) start while offstage", :upb_recovery)}
#{row.("side-B (illusion) start while offstage", :sideb_recovery)}
#{row.("airdodge while offstage", :airdodge_offstage)}

Grab-hold frames (GRAB_WAIT + PUMMEL): #{g.(:grab_hold_frames)} → mean hold #{f1.(g.(:grab_hold_frames) / max(g.(:grab_entries), 1))} f per grab entry.

## Read

- "Dozens" would mean a data problem; anything in the thousands per epoch
  means the recipe sees the behaviour many times and still does not
  reproduce it — the loss is downstream (label weighting, window/stride
  alignment, discretization, or the model). Compare to the bot's rates in
  `eval_runs/0829_situation_hist/README.md`.
- Throw-from-grab vs airdodge-offstage ratio is the direct denominator for
  A1 §2/§3.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
