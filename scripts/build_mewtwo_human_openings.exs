# Audit successful neutral openings from one whole human training replay.
# Usage: elixir -pa '_build/dev/lib/*/ebin' this.exs REPLAY OUTPUT
alias ExPhil.Data.Peppi
alias ExPhil.Eval.NeutralExchange
alias ExPhil.Training.{Labels, RecordedFrames}
[path, output | flags] = System.argv()
unless flags in [[], ["--scripted-teacher"]], do: raise("Unknown source flag")
scripted? = flags == ["--scripted-teacher"]
{:ok, replay} = Peppi.parse(path)
[%{port: subject}] = Enum.filter(replay.metadata.players, &(&1.character_name == "Mewtwo"))

[%{port: opponent, character_name: "Fox"}] =
  Enum.reject(replay.metadata.players, &(&1.port == subject))

unless replay.metadata.stage == 32, do: raise("This round uses the whole FD game only")
score = NeutralExchange.rows(replay, subject) |> NeutralExchange.score()
by_frame = Map.new(replay.frames, &{&1.frame_number, &1})

training =
  Peppi.to_training_frames(replay,
    player_port: subject,
    opponent_port: opponent,
    remap_ports: true
  )

audits =
  for event <- score.events, event.outcome == :subject_opens do
    frame = by_frame[event.first_contact]
    me = frame.players[subject]
    fox = frame.players[opponent]
    previous = by_frame[event.first_contact - 1].players[opponent]
    action = trunc(me.action)
    capture = fox.action in 223..228 and me.action in 212..225

    active =
      if action in 44..69,
        do:
          Melee.FrameData.attack_state(
            :mewtwo,
            Melee.Enums.Action.from_id(action),
            trunc(me.action_frame)
          ) == :attacking,
        else: false

    melee_contact =
      active and me.hitlag_left > 0 and fox.hitlag_left > 0 and fox.percent > previous.percent

    category =
      cond do
        capture -> :grab
        melee_contact and action == 57 -> :down_tilt
        melee_contact and action in 65..69 -> :aerial
        melee_contact -> :other_ground_attack
        true -> :unattributed
      end

    %{
      start: max(0, event.start - 29),
      contact: event.first_contact,
      action: action,
      action_frame: me.action_frame,
      category: category,
      accepted: category != :unattributed,
      attacker_hitlag: me.hitlag_left,
      defender_hitlag: fox.hitlag_left,
      gap: abs(me.x - fox.x),
      defender_action: fox.action
    }
  end

lists =
  for audit <- audits, audit.accepted do
    training
    |> Enum.filter(
      &(&1.game_state.frame >= audit.start - 20 and &1.game_state.frame < audit.contact)
    )
    |> Enum.map(&Map.put(&1, :input_only, &1.game_state.frame < audit.start))
    |> Labels.tag(:recorded)
  end

report = %{
  source: Path.expand(path),
  sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower),
  split: :train_whole_game,
  subject_port: subject,
  opponent_port: opponent,
  source_type: if(scripted?, do: :scripted_teacher, else: :human),
  opponent: if(scripted?, do: "CPU 6", else: "CPU 6 or 9; per-game level unknown"),
  audits: audits,
  selection:
    "Successful exclusive neutral openings; capture or simultaneous hitlag and an active melee attack",
  limitation:
    "Attack attribution inferred from replay states; successful demonstration data is not proof of an optimal decision",
  targets: "Include the clean lead-in and stop before first contact state; no combo continuation",
  accepted: Enum.count(audits, & &1.accepted),
  excluded: Enum.count(audits, &(not &1.accepted)),
  categories: audits |> Enum.filter(& &1.accepted) |> Enum.frequencies_by(& &1.category)
}

payload = RecordedFrames.envelope(lists, report)
RecordedFrames.validate!(payload)
File.mkdir_p!(Path.dirname(output))
File.write!(output, :erlang.term_to_binary(payload), [:exclusive])
File.write!(output <> ".json", Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(Map.take(report, [:accepted, :excluded, :categories]))
IO.inspect(%{lists: length(lists), frames: Enum.sum(Enum.map(lists, &length/1))})
