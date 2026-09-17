# Export only passed teacher sequences, preserving delivered controller inputs.
# mix run --no-start scripts/build_mewtwo_neutral_recordings.exs AUDIT_DIR OUTPUT
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Labels, RecordedFrames}
[dir, output | flags] = System.argv()
unless flags in [[], ["--approved-only"]], do: raise("Unknown export flags")
summary = File.read!(Path.join(dir, "summary.json")) |> Jason.decode!()
approval = File.read!(Path.join(dir, "approval.json")) |> Jason.decode!()
approved_ids = for r <- approval["runs"], r["passed"], do: {r["kind"], r["side"]}

{selected, rejected} =
  Enum.split_with(summary["runs"], &({&1["kind"], &1["side"]} in approved_ids))

if rejected != [] and flags == [],
  do: raise("Teacher battery has failures; explicit --approved-only required to exclude them")

if selected == [], do: raise("No approved teacher sequences")

{lists, sources} =
  selected
  |> Enum.map(fn run ->
    path = Path.join(dir, "#{run["kind"]}_#{run["side"]}.states")
    bytes = File.read!(path)
    states = :erlang.binary_to_term(bytes)
    [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "**/*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    # Continue the already-started technique through landing; the collector
    # sends no new attacks after the first contact. Never include finalization.
    finish = min(run["first_contact_frame"] + 60, run["trial_end"])
    # Use replay PHYSICAL inputs: the live snapshot's shoulder field is
    # processed and becomes 1.0 for digital L, or 0.35 for Z. Those aren't
    # physical analog-trigger labels. Peppi provides the causal raw inputs.
    pairs =
      Peppi.to_training_frames(replay)
      |> Enum.filter(&(&1.game_state.frame >= hd(states).frame and &1.game_state.frame <= finish))
      |> Enum.map(&Map.put(&1, :input_only, &1.game_state.frame < run["trial_start"]))
      |> Labels.tag(:recorded)

    {pairs,
     %{
       path: Path.expand(path),
       sha256: Base.encode16(:crypto.hash(:sha256, bytes), case: :lower),
       replay: replay_path,
       replay_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(replay_path)), case: :lower),
       kind: run["kind"],
       side: run["side"],
       frames: length(pairs)
     }}
  end)
  |> Enum.unzip()

report = %{
  teacher: "MewtwoNeutralTeacher",
  stage: "final_destination",
  opponent: "fox",
  excluded_failed_cases:
    Enum.map(rejected, &Map.take(&1, ["kind", "side", "executed", "contacted", "opening"])),
  label_source: "actual next-frame physical controller inputs from Peppi",
  sources: sources,
  audit: Path.expand(Path.join(dir, "summary.json")),
  scope:
    "Controlled neutral openings plus completion of the initiating technique; no combo labels",
  human_recordings:
    "Preserved and excluded from this scripted baseline; no whole-game split leakage"
}

payload = RecordedFrames.envelope(lists, report)
RecordedFrames.validate!(payload)
File.mkdir_p!(Path.dirname(output))
File.write!(output, :erlang.term_to_binary(payload), [:exclusive])
File.write!(output <> ".json", Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(%{lists: length(lists), frames: Enum.sum(Enum.map(lists, &length/1)), output: output})
