# Side-by-side: a trained POLICY on port 1 vs the scripted EXPERT (teacher)
# on port 2, same match. Watch the model's imperfect multishine next to the
# teacher's perfect one — the visual complement to the ShineChain numbers
# (fixture max 73, teacher live max 370, policy drops after a few).
#
#   mix run scripts/demo_vs_teacher.exs \
#     --policy checkpoints/multishine_daggerloop_20260708_112952_i1_policy.bin \
#     --expert multishine \
#     --dolphin ~/.config/Slippi\ Launcher/netplay --iso ~/isos/melee.iso
#
# Options:
#   --policy PATH     trained policy for port 1 (required)
#   --expert NAME     scripted expert for port 2 (default multishine)
#   --fixture PATHS   expert fixture(s) (default: the expert's own)
#   --character NAME  both players' character (default fox)
#   --stage NAME      (default final_destination)
#   --seconds N       stop after N in-game seconds (default 90)
#   --deterministic   argmax the policy (default: sampled)
#   --dolphin/--iso   paths (default the netplay GUI build, windowed)
#   --headless        no window (measurement only)
#
# On exit it prints ShineChain summaries for BOTH ports so the gap is
# numeric, not just visual.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      expert: :string,
      fixture: :string,
      character: :string,
      stage: :string,
      seconds: :integer,
      deterministic: :boolean,
      dolphin: :string,
      iso: :string,
      headless: :boolean
    ]
  )

unless opts[:policy] do
  Output.error("--policy is required (the port-1 model)")
  System.halt(1)
end

expert_name = opts[:expert] || "multishine"

{expert_mod, default_fixture} =
  case expert_name do
    "multishine" -> {ExPhil.Agents.MultishineExpert, "test/fixtures/replays/fox_multishine_closed.slp"}
    other ->
      Output.error("demo_vs_teacher currently supports --expert multishine (got #{inspect(other)})")
      System.halt(1)
  end

character = String.to_atom(opts[:character] || "fox")
stage = String.to_atom(opts[:stage] || "final_destination")
seconds = opts[:seconds] || 90
max_frames = seconds * 60
ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
replay_dir = "logs/vs_teacher/#{ts}"
File.mkdir_p!(replay_dir)

# Expert table from the fixture (port 1 of the fixture, as usual).
fixture_paths =
  (opts[:fixture] || default_fixture)
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

expert_frames =
  Enum.flat_map(fixture_paths, fn p ->
    case ExPhil.Data.Peppi.parse(p) do
      {:ok, r} ->
        r
        |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      _ ->
        []
    end
  end)

expert = expert_mod.from_frames(expert_frames, player_port: 1)

Output.banner("Model (P1) vs Teacher (P2) — #{expert_name}")

Output.config([
  {"Policy (P1)", opts[:policy]},
  {"Expert (P2)", inspect(expert_mod)},
  {"Character", character},
  {"Stage", stage},
  {"Seconds", seconds},
  {"Mode", if(opts[:deterministic], do: "deterministic", else: "sampled")}
])

{:ok, agent} =
  Agent.start_link(policy_path: opts[:policy], deterministic: opts[:deterministic] || false)

case Agent.warmup(agent) do
  {:ok, ms} -> Output.success("policy warmed up (#{ms}ms)")
  {:error, r} -> Output.warning("warmup: #{inspect(r)}")
end

{:ok, bridge} = MeleePort.start_link()

config = %{
  dolphin_path: Path.expand(opts[:dolphin] || "~/.config/Slippi Launcher/netplay"),
  iso_path: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  controller_port: 1,
  opponent_port: 2,
  character: character,
  # Port 2 is driven manually by the expert, but the bridge still needs a
  # character for it; "external" dummy_mode hands us the port.
  dummy_mode: "external",
  dummy_character: to_string(character),
  dummy_cpu_level: 0,
  stage: stage,
  online_delay: 0,
  no_audio: opts[:headless] || false,
  headless: opts[:headless] || false,
  emulation_speed: if(opts[:headless], do: 0.0, else: 1.0),
  replay_dir: replay_dir,
  slippi_port: 51720
}

neutral = %{
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
      a: cs.button_a, b: cs.button_b, x: cs.button_x, y: cs.button_y,
      z: cs.button_z, l: cs.button_l, r: cs.button_r, d_up: cs.button_d_up
    }
  }
end

Output.puts("Booting Dolphin...")

case MeleePort.init_console(bridge, config, 180_000) do
  {:ok, _} -> Output.success("connected")
  :ok -> Output.success("connected")
  {:error, reason} -> Output.error("init failed: #{inspect(reason)}") && System.halt(1)
end

# Each frame: P1 from the policy, P2 from the expert. prev2 is the expert's
# own last emitted input (the press-edge recovery keys off it).
loop = fn loop, steps, prev2, a1, a2 ->
  if steps > max_frames do
    {:done, a1, a2}
  else
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, gs} ->
        p1 = gs.players[1]
        p2 = gs.players[2]

        # Port 1 — the trained policy
        case Agent.get_controller(agent, gs, player_port: 1) do
          {:ok, c} -> MeleePort.send_controller(bridge, to_input.(c))
          {:error, _} -> MeleePort.send_controller(bridge, neutral)
        end

        # Port 2 — the scripted teacher
        {p2_input, prev2_next} =
          if p2 do
            case expert_mod.label(expert, p2, prev2, p1) do
              {:ok, cs} -> {to_input.(cs), cs}
              :skip -> {neutral, prev2}
            end
          else
            {neutral, prev2}
          end

        MeleePort.send_controller(bridge, Map.put(p2_input, :port, 2))

        a1 = if p1, do: [trunc(p1.action || 0) | a1], else: a1
        a2 = if p2, do: [trunc(p2.action || 0) | a2], else: a2
        loop.(loop, steps + 1, prev2_next, a1, a2)

      {:menu, _} ->
        loop.(loop, steps + 1, prev2, a1, a2)

      s when elem(s, 0) in [:postgame, :game_ended] ->
        {:done, a1, a2}

      {:error, reason} ->
        Output.warning("step error: #{inspect(reason)}")
        {:done, a1, a2}
    end
  end
end

{:done, a1_rev, a2_rev} = loop.(loop, 0, nil, [], [])

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> :ok
end

s1 = ShineChain.summary(Enum.reverse(a1_rev))
s2 = ShineChain.summary(Enum.reverse(a2_rev))

Output.puts("")
Output.divider()
Output.puts("P1 MODEL:   chains=#{s1.chains} max=#{inspect(s1.max_length)} mean=#{inspect(s1.mean_length)} sustained(>=5)=#{s1.sustained} empty_hops=#{s1.empty_hops}")
Output.puts("            length histogram #{inspect(Enum.sort(s1.length_histogram))}")
Output.puts("P2 TEACHER: chains=#{s2.chains} max=#{inspect(s2.max_length)} mean=#{inspect(s2.mean_length)} sustained(>=5)=#{s2.sustained} empty_hops=#{s2.empty_hops}")
Output.puts("            length histogram #{inspect(Enum.sort(s2.length_histogram))}")
Output.success("replay -> #{replay_dir}")

GenServer.stop(agent)

case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} -> out |> String.split("\n", trim: true) |> Enum.each(&System.cmd("kill", [&1]))
  _ -> :ok
end
