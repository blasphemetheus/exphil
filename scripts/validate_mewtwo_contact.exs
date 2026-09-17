# Execute primitives or the stateful neutral teacher in live Mewtwo/Fox FD trials.
# Approval remains limited to the tested setup and opponent behavior.
# devenv shell -- mix run scripts/validate_mewtwo_contact.exs NEW_OUT_DIR
Code.require_file("../libmelee_ex/test/support/probe.ex")

defmodule MewtwoContactProbe do
  alias Melee.{Probe, Controller, Enums}
  @kinds ~w(down_tilt run_up_grab nair fair neutral_down neutral_grab neutral_nair
    neutral_fair neutral_move neutral_attack neutral_wavedash neutral_edge_defense
    neutral_interruption neutral_recenter neutral_turnaround)a
  def kinds, do: @kinds

  def run(kind, side, out) do
    base =
      %{
        neutral_down: :down_tilt,
        neutral_grab: :run_up_grab,
        neutral_fair: :fair,
        neutral_nair: :nair,
        neutral_move: :run_up_grab,
        neutral_attack: :down_tilt,
        neutral_wavedash: :down_tilt,
        neutral_edge_defense: :down_tilt,
        neutral_interruption: :down_tilt,
        neutral_recenter: :down_tilt,
        neutral_turnaround: :down_tilt
      }[kind] || kind

    neutral? = base != kind
    home = Path.join(out, "home_#{kind}_#{side}") |> Path.expand()
    # The ExiAI binary is a playback build: it writes through the
    # regenerated-replay setting, independently of SaveReplays.
    File.mkdir_p!(Path.join(home, "Config"))

    File.write!(
      Path.join(home, "Config/Dolphin.ini"),
      "[Core]\nSlippiRegenerateReplays = True\nSlippiReplayRegenerateDir = #{home}/replays\n"
    )

    port =
      52_110 +
        2 *
          Enum.find_index(
            @kinds,
            &(&1 == kind)
          ) + if side == -1, do: 1, else: 0

    probe =
      Probe.start!(
        path: Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
        iso_path: Path.expand("~/isos/melee.iso"),
        home: home,
        slippi_port: port,
        headless: true,
        gfx_backend: "Null",
        blocking_input: true,
        replay_dir: Path.join(home, "replays"),
        boot_rules: [stock: 1, time_limit: 8],
        exi_inputs: true,
        ffw: true,
        ports: [1, 2]
      )

    try do
      p1 = [
        port: 1,
        character: Enums.Character.to_id(:mewtwo),
        stage: Enums.Stage.to_id(:final_destination)
      ]

      p2 = [
        port: 2,
        character: Enums.Character.to_id(:fox),
        stage: Enums.Stage.to_id(:final_destination)
      ]

      probe =
        Probe.drive!(
          probe,
          fn p ->
            gs = Probe.gamestate(p)
            Melee.GameState.in_game?(gs) and gs.frame >= 0
          end,
          fn p -> [Keyword.put(p1, :autostart, Probe.autostart?(p, [p2])), p2] end,
          timeout_frames: 20_000
        )

      probe = Probe.idle!(probe, 90)
      {center, ""} = Float.parse(System.get_env("MEWTWO_CENTER_SHIFT", "0.0"))
      fox_x = side * center + if(kind == :neutral_edge_defense, do: -side * 27.0, else: 0.0)
      probe = place(probe, 2, fox_x)

      gap =
        case kind do
          :neutral_recenter ->
            {recenter_gap, ""} = Float.parse(System.get_env("MEWTWO_RECENTER_GAP", "75.0"))
            recenter_gap

          :neutral_attack ->
            30.0

          :neutral_wavedash ->
            {wave_gap, ""} = Float.parse(System.get_env("MEWTWO_WAVE_GAP", "60.0"))
            wave_gap

          :neutral_edge_defense ->
            23.0

          :neutral_nair ->
            5.0

          _ ->
            %{down_tilt: 16.0, run_up_grab: 40.0, nair: 9.0, fair: 9.0}[base]
        end

      probe =
        if side == -1 do
          p = place(probe, 1, fox_x - 25.0)
          Controller.press_button(p.controllers[1], :x)
          p = Enum.reduce(1..10, p, fn _, p -> Probe.step!(p) end)
          Controller.release_button(p.controllers[1], :x)
          place(p, 1, fox_x + 10.0, 1.0)
        else
          probe
        end

      probe = place(probe, 1, fox_x - side * gap)
      # Face the opponent without carrying a run into the down-tilt test.
      facing_side = if kind == :neutral_turnaround, do: -side, else: side

      Controller.tilt_analog(
        probe.controllers[1],
        :main,
        if(facing_side == 1, do: 0.65, else: 0.35),
        0.5
      )

      probe = Probe.step!(probe)
      Controller.release_all(probe.controllers[1])

      Controller.tilt_analog(
        probe.controllers[2],
        :main,
        if(side == 1, do: 0.35, else: 0.65),
        0.5
      )

      probe = Probe.step!(probe)
      Controller.release_all(probe.controllers[2])
      probe = Probe.idle!(probe, 30)

      {probe, context} =
        Enum.reduce(1..20, {probe, []}, fn _, {p, rows} ->
          gs = Probe.gamestate(p)
          {Probe.idle!(p, 1), [gs | rows]}
        end)

      context = Enum.reverse(context)

      {probe, context} =
        if kind in [:neutral_attack, :neutral_edge_defense] do
          before_attack = Probe.gamestate(probe)
          Controller.press_button(probe.controllers[2], :a)
          {Probe.step!(probe), context ++ [before_attack]}
        else
          {probe, context}
        end

      start = Probe.gamestate(probe)
      actual_gap = start.players[2].position.x - start.players[1].position.x
      setup_valid = actual_gap * side > 0 and abs(abs(actual_gap) - gap) < 4.0

      setup_valid =
        setup_valid and
          (kind != :neutral_turnaround or start.players[1].facing != (side == 1)) and
          (kind != :neutral_recenter or abs(start.players[1].position.x) > 65)

      tech =
        if neutral? do
          ExPhil.Agents.MewtwoNeutralTeacher.new()
        else
          if kind in [:nair, :fair] do
            opts =
              [aerial: kind] ++
                if(kind == :fair, do: [drift: if(side == 1, do: :right, else: :left)], else: [])

            Melee.Tech.new(:shffl, :mewtwo, opts)
          end
        end

      {finished, trace, _, snapshots} =
        Enum.reduce(0..359, {probe, [], tech, []}, fn n, {p, trace, tech, snapshots} ->
          gs = Probe.gamestate(p)
          me = gs.players[1]
          fox = gs.players[2]

          opened? =
            Enum.any?(
              trace,
              &(&1.opponent_percent > start.players[2].percent or
                  (&1.subject_percent > start.players[1].percent and kind != :neutral_interruption) or
                  &1.opponent_action in 226..228)
            )

          unless tech, do: Controller.release_all(p.controllers[1])
          Controller.release_all(p.controllers[2])

          tech =
            if tech do
              if neutral? do
                {updated, commands} =
                  if opened? and tech.tech == nil do
                    {tech, [:release_all]}
                  else
                    ExPhil.Agents.MewtwoNeutralTeacher.step(tech, me, fox)
                  end

                Enum.each(commands, &command(p.controllers[1], &1))
                updated
              else
                {_, updated} = Melee.Tech.step(tech, me, p.controllers[1])
                updated
              end
            end

          cond do
            opened? ->
              :ok

            kind == :neutral_interruption and n == 0 ->
              Controller.press_button(p.controllers[2], :a)

            kind == :neutral_grab ->
              Controller.press_button(p.controllers[2], :r)

            kind == :neutral_move ->
              direction = if rem(div(n, 25), 2) == 0, do: side, else: -side

              Controller.tilt_analog(
                p.controllers[2],
                :main,
                if(direction > 0, do: 0.65, else: 0.35),
                0.5
              )

            kind in [:neutral_attack, :neutral_edge_defense] and rem(n, 20) == 0 ->
              Controller.press_button(p.controllers[2], :a)

            kind == :down_tilt and n == 0 ->
              # Same crouch + A input used by MewtwoPunishExpert's selected tilt.
              Controller.tilt_analog(p.controllers[1], :main, 0.5, 0.2)
              Controller.press_button(p.controllers[1], :a)

            kind == :run_up_grab and
                not Enum.any?(trace, &(&1.subject_action in [212, 214, 213, 215])) ->
              Controller.press_button(p.controllers[2], :r)
              dx = fox.position.x - me.position.x
              Controller.tilt_analog(p.controllers[1], :main, if(dx > 0, do: 1.0, else: 0.0), 0.5)
              if abs(dx) <= 12.0, do: Controller.press_button(p.controllers[1], :z)

            kind == :run_up_grab ->
              Controller.press_button(p.controllers[2], :r)

            true ->
              :ok
          end

          p = Probe.step!(p)
          gs = Probe.gamestate(p)
          me = gs.players[1]
          fox = gs.players[2]

          entry = %{
            frame: gs.frame,
            subject_action: me.action,
            opponent_action: fox.action,
            subject_action_frame: me.action_frame,
            subject_y: me.position.y,
            opponent_y: fox.position.y,
            subject_facing: me.facing,
            l_cancel: me.l_cancel,
            subject_x: me.position.x,
            opponent_x: fox.position.x,
            opponent_percent: fox.percent,
            subject_percent: me.percent,
            opponent_hitlag: fox.hitlag_left,
            teacher_choice: if(neutral?, do: inspect(tech.choice), else: nil),
            opponent_last_hit_by: fox.last_hit_by
          }

          {p, [entry | trace], tech, [gs | snapshots]}
        end)

      trace = Enum.reverse(trace)

      intended =
        if kind in [:neutral_move, :neutral_attack, :neutral_edge_defense],
          do: [57, 65, 66, 212, 214],
          else: %{down_tilt: [57], run_up_grab: [212, 214], nair: [65], fair: [66]}[base]

      executed = Enum.any?(trace, &(&1.subject_action in intended))

      contacted =
        if base != :run_up_grab or kind == :neutral_move,
          do: Enum.any?(trace, &(&1.opponent_percent > start.players[2].percent)),
          else: Enum.any?(trace, &(&1.opponent_action in 226..228))

      first_contact =
        Enum.find(trace, fn row ->
          row.opponent_percent > start.players[2].percent or
            row.subject_percent > start.players[1].percent or row.opponent_action in 226..228
        end)

      opening =
        cond do
          first_contact == nil ->
            :timeout

          first_contact.subject_percent > start.players[1].percent and
              first_contact.opponent_percent > start.players[2].percent ->
            :trade

          first_contact.subject_percent > start.players[1].percent ->
            :fox

          true ->
            :mewtwo
        end

      recovered =
        kind == :neutral_interruption and opening == :fox and
          Enum.any?(
            trace,
            &(&1.frame > first_contact.frame and &1.subject_action in intended and
                &1.opponent_percent > start.players[2].percent)
          )

      # Preserve raw observations, including actual input readback. Training
      # pairs S[t] with the recorded input from S[t+1], after validation.
      File.write!(
        Path.join(out, "#{kind}_#{side}.states"),
        :erlang.term_to_binary(context ++ [start] ++ Enum.reverse(snapshots))
      )

      # End the one-stock match so Slippi flushes its replay. These frames
      # are outside the trial and must never become training targets.
      Controller.release_all(finished.controllers[1])
      Controller.release_all(finished.controllers[2])

      _ended =
        Probe.until!(
          finished,
          fn p -> not Melee.GameState.in_game?(Probe.gamestate(p)) end,
          fn p ->
            Controller.tilt_analog(p.controllers[1], :main, 1.0, 0.5)
            p
          end,
          timeout_frames: 2400
        )

      %{
        kind: kind,
        side: side,
        setup_valid: setup_valid,
        actual_gap: actual_gap,
        initial_facing: start.players[1].facing,
        initial_x: start.players[1].position.x,
        trial_start: start.frame,
        opening: opening,
        trial_end: List.last(trace).frame,
        replay_dir: Path.join(home, "replays"),
        first_contact_frame: if(first_contact, do: first_contact.frame),
        interruption_recovered: recovered,
        executed: executed,
        contacted: contacted,
        passed: setup_valid and executed and contacted and (opening == :mewtwo or recovered),
        trace: trace
      }
    after
      Probe.stop(probe)
    end
  end

  defp command(c, :release_all), do: Controller.release_all(c)
  defp command(c, {:press, b}), do: Controller.press_button(c, b)
  defp command(c, {:release, b}), do: Controller.release_button(c, b)
  defp command(c, {:tilt, s, x, y}), do: Controller.tilt_analog(c, s, x, y)

  defp place(probe, port, target, amount \\ 0.65) do
    Controller.release_all(probe.controllers[port])

    probe =
      Probe.until!(
        probe,
        fn p -> abs(Probe.gamestate(p).players[port].position.x - target) < 1.5 end,
        fn p ->
          x = Probe.gamestate(p).players[port].position.x

          Controller.tilt_analog(
            p.controllers[port],
            :main,
            if(x < target, do: amount, else: 1.0 - amount),
            0.5
          )

          p
        end,
        timeout_frames: 1200
      )

    Controller.release_all(probe.controllers[port])
    Probe.idle!(probe, 45)
  end
end

[out | selected] = System.argv()
File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)

kinds =
  Enum.filter(
    MewtwoContactProbe.kinds(),
    fn kind ->
      selected == [] or Atom.to_string(kind) in selected
    end
  )

if kinds == [], do: raise("No recognized techniques selected")

runs =
  for kind <- kinds, side <- [1, -1] do
    result = MewtwoContactProbe.run(kind, side, out)
    File.write!(Path.join(out, "#{kind}_#{side}.json"), Jason.encode!(result, pretty: true))
    IO.inspect(Map.drop(result, [:trace]))
    result
  end

File.write!(Path.join(out, "summary.json"), Jason.encode!(%{runs: runs}, pretty: true))
unless Enum.all?(runs, & &1.passed), do: System.halt(1)
