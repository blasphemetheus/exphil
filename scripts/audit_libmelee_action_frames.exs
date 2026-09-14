alias ExPhil.Data.{ActionFrameConvention, Peppi}
{opts, [], []} = OptionParser.parse(System.argv(), strict: [scores: :string, out: :string])
path = Keyword.fetch!(opts, :scores)
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
scores = File.read!(path) |> Jason.decode!()

pairs =
  Enum.flat_map(scores["runs"], fn run ->
    unless run["timing_valid"] and run["diverged"] == false, do: raise("invalid run")
    [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    frames = Peppi.to_training_frames(replay) |> Map.new(&{&1.game_state.frame, &1})

    Enum.map(run["policy_inputs"], fn trace ->
      player = Map.fetch!(frames, trace["frame"]).game_state.players[1]
      unless player.action == trace["action"], do: raise("state alignment mismatch")
      {player.character, player.action, trace["action_frame"], player.action_frame}
    end)
  end)

samples =
  pairs
  |> Enum.frequencies()
  |> Enum.sort()
  |> Enum.map(fn {{character, action, raw, parsed}, count} ->
    %{character: character, action: action, libmelee: raw, parsed: parsed, count: count}
  end)

mismatches =
  Enum.filter(
    samples,
    &(ActionFrameConvention.libmelee_to_parsed(&1.character, &1.action, &1.libmelee) != &1.parsed)
  )

report = %{
  scores: path,
  scores_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower),
  observations: length(pairs),
  mismatches: mismatches,
  samples: samples
}

File.write!(out, Jason.encode!(report, pretty: true), [:exclusive])
unless mismatches == [], do: raise("producer contract does not match recorded AF")
