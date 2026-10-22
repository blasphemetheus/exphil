# Engine page export (2026-10-22): a replay the model has "read" — for every
# frame the agreement eval scored (scripts/agreement_eval.exs --rows), the
# model's press probabilities, its top guess vs the real input, and its
# surprise (−log p of the real input). The page (priv/viewer/engine/) plays
# the replay beside those lanes: the first Stockfish-style read-out.
#
#   mix run scripts/engine_export.exs --report eval_runs/agreement/s906_ep2.json \
#     --rows eval_runs/agreement/s906_ep2_rows.jsonl --out eval_runs/1022_engine
#
# Writes per game: <game>.msltrace.json (the whole replay, viewer index =
# game frame) + <game>.rows.json (the eval rows for that game), and
# manifest.json (games, the report's headline numbers, the policy).

alias ExPhil.Data.Peppi
alias ExPhil.Sim.Trace
alias ExPhil.Training.Output

{opts, _, bad} = OptionParser.parse(System.argv(), strict: [report: :string, rows: :string, out: :string, games: :integer])
if bad != [], do: raise("bad options: #{inspect(bad)}")
report_path = opts[:report] || raise("--report required")
report = report_path |> File.read!() |> Jason.decode!()
rows_path = opts[:rows] || String.replace(report_path, ~r/\.json$/, "_rows.jsonl")
out = opts[:out] || "eval_runs/#{Date.utc_today() |> Calendar.strftime("%m%d")}_engine"
File.mkdir_p!(out)

split = report["split"]
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation") |> Enum.take(length(report["games"]))
files = if opts[:games], do: Enum.take(files, opts[:games]), else: files

Output.banner("Engine export: #{Path.basename(report["policy"])} over #{length(files)} games")

rows_by_game =
  rows_path
  |> File.stream!()
  |> Stream.map(&Jason.decode!/1)
  |> Enum.group_by(& &1["g"])

games =
  files
  |> Enum.with_index()
  |> Enum.map(fn {path, gi} ->
    game = Path.basename(path, ".slp")
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    opp = Enum.find(meta.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(path, player_port: own.port)

    states =
      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.game_state)

    # viewer index = game frame: the states are contiguous from frame 0
    first = List.first(states).frame
    unless first == 0, do: Output.warning("  #{game}: first frame #{first}, viewer index = frame - #{first}")

    chars = [Trace.char_id(own.character_name), Trace.char_id(opp.character_name)]
    id = game |> String.slice(0, 40) |> String.replace(~r/[^A-Za-z0-9_-]+/, "_")

    Trace.from_game_states(states, label: "ENGINE #{game} (red Fox = the expert the model reads; port #{own.port} vs #{opp.character_name} port #{opp.port})", chars: chars)
    |> Trace.write!(Path.join(out, "#{id}.msltrace.json"))

    rows =
      (rows_by_game[gi] || [])
      |> Enum.sort_by(& &1["frame"])
      |> Enum.map(fn r -> r |> Map.drop(["g"]) |> Map.put("i", r["frame"] - first) end)

    File.write!(Path.join(out, "#{id}.rows.json"), Jason.encode!(%{game: game, id: id, frames: length(states), first_frame: first, opponent: opp.character_name, own_port: own.port, rows: rows}))
    mean_nll = if rows == [], do: nil, else: Float.round(Enum.sum(Enum.map(rows, & &1["nll"])) / length(rows), 3)
    Output.puts("  #{gi + 1}/#{length(files)} #{game}: #{length(states)} frames, #{length(rows)} scored, surprise #{mean_nll}")
    %{game: game, id: id, frames: length(states), rows: length(rows), opponent: opp.character_name, surprise: mean_nll, trace: "#{id}.msltrace.json", sidecar: "#{id}.rows.json"}
  end)

manifest = %{
  created: DateTime.utc_now() |> DateTime.to_iso8601(),
  policy: report["policy"], stride: report["stride"], games: games,
  overall: report["overall"], change: report["change"], hold: report["hold"], press_calibration: report["press_calibration"],
  levels: report["levels"]
}

File.write!(Path.join(out, "manifest.json"), Jason.encode!(manifest))
Output.puts("RESULT engine export: #{length(games)} games -> #{out}/manifest.json")
