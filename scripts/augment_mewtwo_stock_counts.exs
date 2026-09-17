# Whole-history stock-count augmentation for the stock-independent scripted teacher.
# Human decisions are deliberately not augmented by this script.
alias ExPhil.Training.RecordedFrames
[output | paths] = System.argv()
if paths == [], do: raise("Supply scripted-teacher recordings")

lists =
  Enum.flat_map(paths, fn path ->
    payload = path |> File.read!() |> :erlang.binary_to_term()
    report = Jason.decode!(payload.teacher_validation_json)

    unless report["teacher"] == "MewtwoNeutralTeacher" or
             report["source_type"] == "scripted_teacher",
           do: raise("Stock augmentation requires an explicitly scripted teacher source")

    RecordedFrames.validate!(payload)
  end)

augmented =
  for {frames, index} <- Enum.with_index(lists), own <- 1..4 do
    for port <- [1, 2] do
      counts = Enum.map(frames, & &1.game_state.players[port].stock) |> Enum.uniq()

      unless length(counts) == 1 and hd(counts) > 0,
        do: raise("Cannot augment a sequence containing a stock transition")
    end

    opponent = rem(own + index, 4) + 1

    for frame <- frames do
      frame
      |> put_in([:game_state, Access.key(:players), 1, Access.key(:stock)], own)
      |> put_in([:game_state, Access.key(:players), 2, Access.key(:stock)], opponent)
    end
  end

report = %{
  operation: "Whole-sequence stock-count augmentation; physical input labels unchanged",
  justification:
    "MewtwoNeutralTeacher does not condition its choices on stock count; accepted sequences contain no stock transitions",
  assignments:
    "Own counts 1..4; opponent count rotated by sequence index to cover equal and unequal stocks",
  sources:
    Enum.map(paths, fn p ->
      %{
        path: Path.expand(p),
        sha256: Base.encode16(:crypto.hash(:sha256, File.read!(p)), case: :lower)
      }
    end),
  original_lists: length(lists),
  augmented_lists: length(augmented),
  frames: Enum.sum(Enum.map(augmented, &length/1))
}

payload = RecordedFrames.envelope(augmented, report)
RecordedFrames.validate!(payload)
File.mkdir_p!(Path.dirname(output))
File.write!(output, :erlang.term_to_binary(payload), [:exclusive])
File.write!(output <> ".json", Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(Map.take(report, [:original_lists, :augmented_lists, :frames]))
