# Neutral-range scorecard (F3c, 09-01) — Bradley's pushback on the F3b
# read: "Fox long-range game = retreat/laser zone is too simple; the expert
# data would bear that out." This measures it: BEHAVIORAL stick direction
# and option usage in NEUTRAL frames as a function of opponent distance,
# same detector over the expert corpus and the bot's live-look replays.
#
# Per |dx| bucket (neutral frames only):
#   toward      mean signed stick-x toward the opponent (+1 = full toward)
#   %twd / %awy fraction of frames past a 0.3 deadzone each way
#   dash%       frames in dash/run/turn (18,20,21)
#   laserG/A%   frames in grounded (341-344) / aerial (345-348) blaster states
#   jump%       frames in jumpsquat/jump rise (24,25,26)
#
#   mix run scripts/neutral_range_scorecard.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --set bot_v12ar='eval_runs/0831_livelook_v12ar/2026-08-Mainline/*.slp:1' \
#     --limit-files 40 --out eval_runs/0901_neutral_range/RESULTS.md
#
# --set NAME=GLOB[:PORT] — with :PORT the subject port is pinned (bot
# replays); without, the port is resolved by --char-id (default 2 = Fox),
# dittos skipped. --min-frames 120.
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, limit_files: :integer, char_id: :integer, out: :string]
  )

sets =
  Keyword.get_values(opts, :set)
  |> Enum.map(fn s ->
    [name, rest] = String.split(s, "=", parts: 2)

    case String.split(rest, ":") do
      [glob, port] -> {name, glob, String.to_integer(port)}
      [glob] -> {name, glob, :auto}
    end
  end)

if sets == [], do: raise("--set NAME=GLOB[:PORT] required")
limit_files = opts[:limit_files] || 40
char_id = opts[:char_id] || 2

buckets = [{0, 20}, {20, 40}, {40, 70}, {70, 100}, {100, 140}, {140, 999}]
bucket_of = fn d -> Enum.find_index(buckets, fn {lo, hi} -> d >= lo and d < hi end) end

dash_states = MapSet.new([18, 20, 21])
laser_ground = MapSet.new(341..344)
laser_air = MapSet.new(345..348)
jump_states = MapSet.new([24, 25, 26])

Output.banner("Neutral-range scorecard (F3c)")

resolve = fn path, port_spec ->
  case port_spec do
    :auto ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == char_id)) do
            [%{port: p}] -> {:ok, p}
            _ -> :skip
          end

        _ -> :skip
      end

    p -> {:ok, p}
  end
end

score_set = fn {name, glob, port_spec} ->
  files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files * 3)

  picked =
    files
    |> Enum.flat_map(fn f ->
      case resolve.(f, port_spec) do
        {:ok, p} -> [{f, p}]
        _ -> []
      end
    end)
    |> Enum.take(limit_files)

  Output.puts("== #{name}: #{length(picked)} files")

  rows =
    picked
    |> Enum.flat_map(fn {path, port} ->
      opp = if port == 1, do: 2, else: 1

      case Peppi.parse(path) do
        {:ok, replay} ->
          frames =
            replay
            |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
            |> Enum.reject(&(&1.game_state.frame < 0))

          sits = Situations.label_states(Enum.map(frames, & &1.game_state), port, as: :set)

          Enum.zip(frames, sits)
          |> Enum.filter(fn {_f, set} -> MapSet.member?(set, :neutral) end)
          |> Enum.map(fn {f, _} ->
            self = f.game_state.players[port]
            o = f.game_state.players[opp]
            dx = o.x - self.x
            stick = (f.controller.main_stick.x - 0.5) * 2.0
            toward = if dx < 0, do: -stick, else: stick
            act = self.action

            {bucket_of.(abs(dx)), toward, act}
          end)

        _ -> []
      end
    end)

  by_bucket = Enum.group_by(rows, &elem(&1, 0))

  stats =
    Enum.map(0..(length(buckets) - 1), fn bi ->
      rs = Map.get(by_bucket, bi, [])
      n = length(rs)

      if n == 0 do
        %{bucket: bi, n: 0}
      else
        towards = Enum.map(rs, &elem(&1, 1))
        acts = Enum.map(rs, &elem(&1, 2))

        %{
          bucket: bi, n: n,
          toward: Enum.sum(towards) / n,
          pct_twd: Enum.count(towards, &(&1 > 0.3)) / n,
          pct_awy: Enum.count(towards, &(&1 < -0.3)) / n,
          dash: Enum.count(acts, &MapSet.member?(dash_states, &1)) / n,
          laser_g: Enum.count(acts, &MapSet.member?(laser_ground, &1)) / n,
          laser_a: Enum.count(acts, &MapSet.member?(laser_air, &1)) / n,
          jump: Enum.count(acts, &MapSet.member?(jump_states, &1)) / n
        }
      end
    end)

  {name, stats}
end

results = Enum.map(sets, score_set)

pct = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
f3 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

tables =
  Enum.map_join(results, "\n\n", fn {name, stats} ->
    rows =
      Enum.map_join(stats, "\n", fn s ->
        {lo, hi} = Enum.at(buckets, s.bucket)
        label = if hi > 500, do: "#{lo}+", else: "#{lo}-#{hi}"

        if s.n == 0 do
          "| #{label} | 0 | — | — | — | — | — | — | — |"
        else
          "| #{label} | #{s.n} | #{f3.(s.toward)} | #{pct.(s.pct_twd)} | #{pct.(s.pct_awy)} | " <>
            "#{pct.(s.dash)} | #{pct.(s.laser_g)} | #{pct.(s.laser_a)} | #{pct.(s.jump)} |"
        end
      end)

    """
    ### #{name}

    | \\|dx\\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|
    #{rows}
    """
  end)

report = """
# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(#{Enum.map_join(sets, ", ", fn {n, _, p} -> "#{n}:#{inspect(p)}" end)}).
toward = mean signed stick-x toward the opponent (deadzone-free mean);
%twd/%awy past a 0.3 deadzone; option columns = share of neutral frames in
those action states (dash/run/turn; grounded blaster 341-344; aerial
blaster 345-348; jumpsquat+rise).

#{tables}

Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
