Code.require_file("../libmelee_ex/test/support/probe.ex")
alias Melee.{Probe, Controller, Enums}
alias ExPhil.Agents.MewtwoNeutralTeacher, as: Teacher
[out | args] = System.argv()
{opts, [], []} = OptionParser.parse(args, strict: [mode: :string, runs: :integer])
mode = opts[:mode] || "cpu"
count = opts[:runs] || 9
unless mode in ["cpu", "stand"] and count > 0, do: raise("Invalid teacher match protocol")

defmodule MewtwoTeacherReplay do
  def read(path, remaining \\ 60) do
    case ExPhil.Data.Peppi.parse(path) do
      {:ok, replay} ->
        replay

      {:error, _reason} when remaining > 0 ->
        Process.sleep(50)
        read(path, remaining - 1)

      {:error, reason} ->
        raise("Replay did not finalize: #{inspect(reason)}")
    end
  end
end

File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)

runs =
  for n <- 1..count do
    home = Path.expand(Path.join(out, "run_#{n}"))
    File.mkdir_p!(Path.join(home, "Config"))

    File.write!(
      Path.join(home, "Config/Dolphin.ini"),
      "[Core]\nSlippiRegenerateReplays = True\nSlippiReplayRegenerateDir = #{home}/replays\n"
    )

    p =
      Probe.start!(
        path: Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
        iso_path: Path.expand("~/isos/melee.iso"),
        home: home,
        slippi_port: 52_180 + n,
        headless: true,
        gfx_backend: "Null",
        blocking_input: true,
        replay_dir: Path.join(home, "replays"),
        exi_inputs: true,
        ffw: true,
        ports: [1, 2]
      )

    try do
      a = [
        port: 1,
        character: Enums.Character.to_id(:mewtwo),
        stage: Enums.Stage.to_id(:final_destination)
      ]

      b = [
        port: 2,
        character: Enums.Character.to_id(:fox),
        stage: Enums.Stage.to_id(:final_destination)
      ]

      b = if mode == "cpu", do: Keyword.put(b, :cpu_level, 6), else: b

      p =
        Probe.drive!(
          p,
          fn p ->
            gs = Probe.gamestate(p)
            Melee.GameState.in_game?(gs) and gs.frame >= 0
          end,
          fn p -> [Keyword.put(a, :autostart, Probe.autostart?(p, [b])), b] end,
          timeout_frames: 20_000
        )

      {p, _, states, choices} =
        Enum.reduce(0..1799, {p, Teacher.new(), [], []}, fn _, {p, t, states, choices} ->
          gs = Probe.gamestate(p)
          if mode == "stand", do: Controller.release_all(p.controllers[2])
          {t, commands} = Teacher.step(t, gs.players[1], gs.players[2])

          Enum.each(commands, fn
            :release_all -> Controller.release_all(p.controllers[1])
            {:press, b} -> Controller.press_button(p.controllers[1], b)
            {:release, b} -> Controller.release_button(p.controllers[1], b)
            {:tilt, s, x, y} -> Controller.tilt_analog(p.controllers[1], s, x, y)
          end)

          {Probe.step!(p), t, [gs | states], [inspect(t.choice) | choices]}
        end)

      states = Enum.reverse(states)
      File.write!(Path.join(home, "states.bin"), :erlang.term_to_binary(states))
      File.write!(Path.join(home, "choices.json"), Jason.encode!(Enum.reverse(choices)))
      Controller.release_all(p.controllers[1])

      _ =
        Probe.until!(
          p,
          fn p -> not Melee.GameState.in_game?(Probe.gamestate(p)) end,
          fn p ->
            Controller.tilt_analog(p.controllers[1], :main, 1.0, 0.5)
            p
          end,
          timeout_frames: 7200
        )

      [replay_path] = Path.wildcard(Path.join(home, "replays/**/*.slp"))
      replay = MewtwoTeacherReplay.read(replay_path)
      metrics = ExPhil.Eval.MewtwoNeutralBenchmark.score(replay, List.last(states).frame)

      result = %{
        run: n,
        metrics: metrics,
        replay: replay_path,
        choices: Enum.frequencies(choices),
        opening_gate: (metrics.neutral.outcomes[:subject_opens] || 0) > 0
      }

      File.write!(Path.join(home, "result.json"), Jason.encode!(result, pretty: true))
      IO.inspect(%{run: n, outcomes: metrics.neutral.outcomes, deaths: metrics.deaths})
      result
    after
      Probe.stop(p)
    end
  end

File.write!(
  Path.join(out, "summary.json"),
  Jason.encode!(%{mode: mode, runs: runs, passed: Enum.all?(runs, & &1.opening_gate)},
    pretty: true
  )
)
