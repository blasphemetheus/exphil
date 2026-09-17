alias ExPhil.Eval.NeutralExchange
[dir] = System.argv()
summary = File.read!(Path.join(dir, "summary.json")) |> Jason.decode!()

runs =
  for run <- summary["runs"] do
    states =
      File.read!(Path.join(dir, "#{run["kind"]}_#{run["side"]}.states"))
      |> :erlang.binary_to_term()

    rows =
      for g <- states do
        p = fn n ->
          p = g.players[n]

          %{
            action: p.action,
            percent: p.percent,
            stock: p.stock,
            hitstun: p.action in 75..91,
            offstage: abs(p.position.x) > 85.57 or p.position.y < -5
          }
        end

        %{frame: g.frame, subject: p.(1), opponent: p.(2)}
      end

    score = NeutralExchange.score(rows, lead: 20)
    first = List.first(score.events)

    first_valid =
      first.outcome == :subject_opens and first.first_contact == run["first_contact_frame"]

    players = Enum.filter(states, &(&1.frame >= run["trial_start"])) |> Enum.map(& &1.players[1])
    actions = Enum.map(players, & &1.action)
    jump = Enum.find_index(actions, &(&1 == 24))
    dodge = Enum.find_index(actions, &(&1 == 236))
    land = Enum.find_index(actions, &(&1 == 43))
    # An airdodge aimed into the floor can enter special landing in the
    # same update: require airborne -> delivered diagonal L/R -> landing.
    instant_dodge =
      if land != nil and land > 0 do
        before = Enum.at(players, land - 1)
        landed = Enum.at(players, land)
        {x, y} = landed.controller_state.main_stick

        not before.on_ground and
          (landed.controller_state.button.l or landed.controller_state.button.r) and
          y < 0.4 and abs(x - 0.5) > 0.2
      else
        false
      end

    wave =
      jump != nil and land != nil and jump < land and
        (instant_dodge or (dodge != nil and jump < dodge and dodge < land))

    displacement =
      if wave,
        do:
          Enum.at(players, min(land + 10, length(players) - 1)).position.x -
            hd(players).position.x

    aerial = Enum.any?(actions, &(&1 in [65, 66]))
    canceled = Enum.any?(players, &(&1.l_cancel == 1 and &1.action in 70..74))
    apex = Enum.max(Enum.map(players, & &1.position.y))

    movement_valid =
      if run["kind"] == "neutral_wavedash",
        do: wave and displacement * run["side"] > 15,
        else: true

    aerial_valid = if aerial, do: canceled and apex > 1 and apex < 20, else: true

    %{
      kind: run["kind"],
      side: run["side"],
      opening_matches_exclusive_scorer: first_valid,
      actual_wavedash: wave,
      displacement: displacement,
      apex: apex,
      aerial: aerial,
      l_cancel: canceled,
      passed: run["passed"] and first_valid and movement_valid and aerial_valid
    }
  end

report = %{
  runs: runs,
  approved: Enum.all?(runs, & &1.passed),
  scope:
    "These controlled Fox/FD scenarios only; live action-state damage and percent/capture audit",
  teacher_sha256:
    :crypto.hash(:sha256, File.read!("lib/exphil/agents/mewtwo_neutral_teacher.ex"))
    |> Base.encode16(case: :lower)
}

File.write!(Path.join(dir, "approval.json"), Jason.encode!(report, pretty: true))
Enum.each(runs, &IO.inspect/1)
unless report.approved, do: System.halt(1)
