# Drive Dolphin LIVE from a scripted expert (the "teacher"), not a policy.
#
# Answers the question a policy run can't: *can the teacher itself actually
# do the thing?* If the expert can't sustain a grounded multishine, no
# amount of imitation will teach one — the target behavior isn't in the
# supervision. (GOALS.md Track B, rung 1: "fix what the expert teaches".)
#
#   mix run scripts/demo_expert.exs --expert multishine \
#     --dolphin ~/.config/Slippi\ Launcher/netplay --iso ~/isos/melee.iso
#
# Options:
#   --expert NAME     multishine | mewtwo_fair | mewtwo_combo | mewtwo_techchase
#                     | fox_recovery   (default multishine)
#   --fixture PATHS   comma-separated fixture replays (default: the expert's own)
#   --character NAME  (default: fox for multishine, mewtwo otherwise)
#   --stage NAME      (default final_destination — flat ground for multishine)
#   --port N          expert's port (default 1)
#   --dummy NAME      port-2 dummy (default tech_random so no controller is needed)
#   --seconds N       stop after N seconds of in-game time (default 60)
#   --dolphin/--iso   paths
#   --headless        no window (measurement only)
#   --replay-dir DIR  where Dolphin writes the .slp (default logs/expert_demo/<ts>)
#
# On exit it prints an ExPhil.Eval.ShineChain summary of what the EXPERT
# actually produced, so "did the teacher sustain?" is answered numerically,
# not just visually.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.MeleePort
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      expert: :string,
      fixture: :string,
      character: :string,
      stage: :string,
      port: :integer,
      dummy: :string,
      seconds: :integer,
      dolphin: :string,
      iso: :string,
      headless: :boolean,
      replay_dir: :string
    ]
  )

expert_name = opts[:expert] || "multishine"

{expert_mod, default_fixture, default_char} =
  case expert_name do
    "multishine" ->
      {ExPhil.Agents.MultishineExpert, "test/fixtures/replays/fox_multishine_closed.slp", :fox}

    "mewtwo_fair" ->
      {ExPhil.Agents.MewtwoFairExpert, "test/fixtures/replays/mewtwo_fair_chains.slp", :mewtwo}

    "mewtwo_combo" ->
      {ExPhil.Agents.MewtwoComboExpert, ExPhil.Agents.FixtureSets.mewtwo_combo_csv(), :mewtwo}

    "mewtwo_techchase" ->
      {ExPhil.Agents.MewtwoTechChaseExpert, nil, :mewtwo}

    "fox_recovery" ->
      {ExPhil.Agents.FoxRecoveryExpert, nil, :fox}

    other ->
      Output.error("Unknown expert #{inspect(other)}")
      System.halt(1)
  end

port = opts[:port] || 1
seconds = opts[:seconds] || 60
max_frames = seconds * 60

character =
  case opts[:character] do
    nil -> default_char
    s -> String.to_atom(s)
  end

stage = String.to_atom(opts[:stage] || "final_destination")
ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
replay_dir = opts[:replay_dir] || "logs/expert_demo/#{ts}"
File.mkdir_p!(replay_dir)

# Build the expert exactly as the drill does: table from the fixture(s).
fixture_paths =
  (opts[:fixture] || default_fixture || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

expert =
  if fixture_paths == [] do
    # Rules-only experts (fox_recovery, techchase) need no table
    expert_mod.from_frames([], player_port: port)
  else
    frames =
      Enum.flat_map(fixture_paths, fn p ->
        case ExPhil.Data.Peppi.parse(p) do
          {:ok, replay} ->
            replay
            |> ExPhil.Data.Peppi.to_training_frames(
              player_port: port,
              opponent_port: if(port == 1, do: 2, else: 1)
            )
            |> Enum.reject(&(&1.game_state.frame < 0))

          _ ->
            []
        end
      end)

    expert_mod.from_frames(frames, player_port: port)
  end

Output.banner("Expert Demo — #{expert_name} (the TEACHER, not a policy)")

Output.config([
  {"Expert", inspect(expert_mod)},
  {"Fixtures", Enum.map(fixture_paths, &Path.basename/1)},
  {"Character", character},
  {"Stage", stage},
  {"Seconds", seconds},
  {"Replay dir", replay_dir}
])

{:ok, bridge} = MeleePort.start_link()

elixir_dummy =
  case opts[:dummy] || "tech_random" do
    "tech_random" -> ExPhil.Agents.Dummies.TechRandom
    "none" -> nil
    _ -> ExPhil.Agents.Dummies.TechRandom
  end

config = %{
  dolphin_path: Path.expand(opts[:dolphin] || "~/.config/Slippi Launcher/netplay"),
  iso_path: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  controller_port: port,
  opponent_port: if(port == 1, do: 2, else: 1),
  character: character,
  stage: stage,
  online_delay: 0,
  dummy_mode: if(elixir_dummy, do: "external", else: "none"),
  dummy_character: "fox",
  dummy_cpu_level: 0,
  no_audio: opts[:headless] || false,
  headless: opts[:headless] || false,
  emulation_speed: if(opts[:headless], do: 0.0, else: 1.0),
  replay_dir: replay_dir,
  slippi_port: 51700
}

neutral_input = %{
  main_stick: %{x: 0.5, y: 0.5},
  c_stick: %{x: 0.5, y: 0.5},
  shoulder: 0.0,
  buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
}

to_input = fn cs ->
  %{
    main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
    c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
    shoulder: cs.l_shoulder + cs.r_shoulder,
    buttons: %{
      a: cs.button_a,
      b: cs.button_b,
      x: cs.button_x,
      y: cs.button_y,
      z: cs.button_z,
      l: cs.button_l,
      r: cs.button_r,
      d_up: cs.button_d_up
    }
  }
end

Output.puts("Booting Dolphin...")

case MeleePort.init_console(bridge, config, 180_000) do
  {:ok, _} -> Output.success("connected")
  :ok -> Output.success("connected")
  {:error, reason} -> Output.error("init failed: #{inspect(reason)}") && System.halt(1)
end

# Drive: each frame, ask the expert what a perfect player would press.
# `prev` is the expert's OWN last emitted input — the same "input that
# actually landed" convention the DAgger labels use, which the press-edge
# recovery rules key off.
loop = fn loop, steps, prev, actions ->
  cond do
    steps > max_frames ->
      {:done, actions}

    true ->
      case MeleePort.step(bridge, auto_menu: true) do
        {:ok, gs} ->
          player = gs.players[port]

          if player do
            action = trunc(player.action || 0)

            {input, prev_cs} =
              case expert_mod.label(expert, player, prev, gs.players[if(port == 1, do: 2, else: 1)]) do
                {:ok, cs} -> {to_input.(cs), cs}
                :skip -> {neutral_input, prev}
              end

            MeleePort.send_controller(bridge, input)
            loop.(loop, steps + 1, prev_cs, [action | actions])
          else
            MeleePort.send_controller(bridge, neutral_input)
            loop.(loop, steps + 1, prev, actions)
          end

        {:menu, _} ->
          loop.(loop, steps + 1, prev, actions)

        {:postgame, _} ->
          {:done, actions}

        {:game_ended, _} ->
          {:done, actions}

        {:error, reason} ->
          Output.warning("step error: #{inspect(reason)}")
          {:done, actions}
      end
  end
end

{:done, actions_rev} = loop.(loop, 0, nil, [])
actions = Enum.reverse(actions_rev)

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> :ok
end

Output.puts("")
Output.divider()

if expert_name == "multishine" do
  s = ShineChain.summary(actions)
  Output.puts("TEACHER shine chains: #{inspect(s.length_histogram)}")
  Output.puts("  chains=#{s.chains} shines=#{s.shines} mean=#{inspect(s.mean_length)} max=#{inspect(s.max_length)}")
  Output.puts("  sustained(>=5)=#{s.sustained} empty_hops=#{s.empty_hops} ended_by=#{inspect(s.ended_by)}")

  cond do
    s.chains == 0 ->
      Output.error("The TEACHER never shined — the target behavior is not being produced at all.")

    s.max_length && s.max_length < 5 ->
      Output.warning(
        "The TEACHER itself cannot sustain (max chain #{s.max_length}). " <>
          "No imitation of this expert will multishine — fix the expert/fixture first."
      )

    true ->
      Output.success("The TEACHER sustains (max #{s.max_length}) — the target IS in the supervision.")
  end
else
  Output.puts("frames observed: #{length(actions)}")
end

Output.success("replay -> #{replay_dir}")

case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} ->
    out |> String.split("\n", trim: true) |> Enum.each(fn pid -> System.cmd("kill", [pid]) end)

  _ ->
    :ok
end
