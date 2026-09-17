# Read and preserve whole recordings before choosing any training split.
# elixir -pa '_build/dev/lib/*/ebin' scripts/inventory_mewtwo_demonstrations.exs OUT REPLAY...
alias ExPhil.Data.Peppi

[out | paths] = System.argv()
if paths == [], do: raise("Supply an output directory and replay paths")
File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)
File.mkdir!(Path.join(out, "replays"))

sha = fn bytes -> :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower) end

runs =
  Enum.map(paths, fn path ->
    bytes = File.read!(path)
    {:ok, replay} = Peppi.parse(path)
    true = bytes == File.read!(path)
    [subject] = Enum.filter(replay.metadata.players, &(&1.character_name == "Mewtwo"))
    [opponent] = Enum.reject(replay.metadata.players, &(&1.port == subject.port))
    true = opponent.character_name == "Fox"

    rows = replay.frames |> Enum.filter(&(&1.frame_number >= 0))
    actions = Enum.map(rows, &trunc(&1.players[subject.port].action))

    onsets =
      rows
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [a, b] ->
        b.frame_number == a.frame_number + 1 and
          a.players[subject.port].action != b.players[subject.port].action
      end)
      |> Enum.frequencies_by(fn [_, b] -> trunc(b.players[subject.port].action) end)

    destination = Path.join([out, "replays", Path.basename(path)])
    File.write!(destination, bytes, [:exclusive])

    %{
      source: Path.expand(path),
      preserved_replay: Path.expand(destination),
      sha256: sha.(bytes),
      subject_port: subject.port,
      opponent_port: opponent.port,
      stage: Melee.Enums.Stage.from_external(replay.metadata.stage),
      playable_frames: length(rows),
      approximate_seconds: length(rows) / 60,
      split: "unassigned",
      opponent_controller: "awaiting_user_description",
      action_onsets: %{
        down_tilt: onsets[57] || 0,
        standing_grab: onsets[212] || 0,
        dash_grab: onsets[214] || 0,
        jumpsquat: onsets[24] || 0,
        airdodge: onsets[236] || 0,
        special_landing: onsets[43] || 0,
        aerials:
          Map.new(
            [nair: 65, fair: 66, bair: 67, uair: 68, dair: 69],
            fn {name, action} -> {name, onsets[action] || 0} end
          )
      },
      shield_action_frames: Enum.count(actions, &(&1 in 178..182))
    }
  end)

manifest = %{
  created_at: DateTime.to_iso8601(DateTime.utc_now()),
  purpose: "Human Mewtwo demonstrations for teacher validation; no training split assigned",
  counts_are: "Action-state entries, not successful openings, short hops, or verified wavedashes",
  parser_sha256: sha.(File.read!("native/exphil_peppi/src/lib.rs")),
  runs: runs
}

File.write!(Path.join(out, "inventory.json"), JSON.encode!(manifest), [:exclusive])
Enum.each(runs, &IO.inspect(Map.take(&1, [:stage, :playable_frames, :action_onsets])))
