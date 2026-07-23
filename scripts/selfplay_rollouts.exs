# Seeded self-play rollouts (DATA_FLYWHEEL_DESIGN 2026-07-23, stage B3).
#
# Ghost-P2 drills can't teach neutral — neutral is interactive. This puts
# TWO policies in a both-neutral moment (input-prefix replay, exactly the
# scenario_suite mechanics) and lets them play out an interactive window.
# Finalized .slp rollouts + seed_meta.json land in a seed dir.
#
# DATA USE: RL/PPO only — self-play trajectories reinforce the policy's own
# habits and are NOT behavior-cloning demonstrations (the launcher's
# NEWERA8_SEED_ROLLOUTS_DIR path expert-relabels; PPO consumption is the
# r17 approach-shaping branch).
#
#   mix run scripts/selfplay_rollouts.exs \
#     --policy checkpoints/mewtwo_combo_newera_r16_r13_policy.bin \
#     --manifest scenarios/manifest.json --types idle_deadlock \
#     --runs 3 --out seeds/selfplay_r16
#
# Options:
#   --policy PATH        Port-1 policy (required)
#   --policy2 PATH       Port-2 policy (default: same — mirror self-play;
#                        pass an older checkpoint for league diversity)
#   --manifest PATH      Scenario manifest (default scenarios/manifest.json)
#   --types a,b          Entry types to seed from (default idle_deadlock —
#                        the both-neutral situations)
#   --only N,M           Entry indices after type filter
#   --runs N             Attempts per entry (default 3)
#   --temperature T      Sampling temperature BOTH policies (default 0.8;
#                        deterministic mirror self-play would be pointless)
#   --window N           Interactive frames after handoff (default 600)
#   --no-finalize        Skip the SD finalize (rollouts stay unparseable!)
#   --out DIR            Seed dir (default seeds/selfplay_<ts>)
#   --dolphin PATH       Headless ExiAI (default ~/.local/share/slippi/exi-ai/dolphin-emu-headless)
#   --iso PATH           Melee ISO (default ~/isos/melee.iso)
#   --slippi-port N      Base port (default 51600, +1 per run)
#   --input-offset N     (default 1)  --drift-tolerance F (default 3.0)
#   --quiet
#
# Prefix mechanics, input conversion, and finalize are duplicated from
# scenario_suite.exs (see the PrefixReplay extraction note in
# DRILLS_DESIGN_2026-07-23.md).

require Logger

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ScenarioScan
alias ExPhil.Training.Output

defmodule SelfplayRollouts do
  @moduledoc false

  @neutral %{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
  }

  @suicide %{@neutral | main_stick: %{x: 0.0, y: 0.5}}
  @menu_step_limit 12_000
  @finalize_step_limit 1_200
  @trigger_digital_threshold 0.31

  def prepare_replay(slp, opts \\ []) do
    shim = Keyword.get(opts, :pipe_shim, true)
    {:ok, replay} = Peppi.parse(Path.expand(slp))

    Enum.reduce(replay.frames, %{}, fn f, inputs ->
      p1 = f.players[1]
      p2 = f.players[2]

      if p1 && p2 do
        Map.put(inputs, f.frame_number, {rec_input(p1.controller, shim), rec_input(p2.controller, shim)})
      else
        inputs
      end
    end)
  end

  defp rec_input(c, pipe_shim) do
    %{
      main_stick: %{x: c.main_stick_x, y: c.main_stick_y},
      c_stick: %{x: c.c_stick_x, y: c.c_stick_y},
      shoulder: max(c.l_trigger, c.r_trigger),
      buttons: %{
        a: c.button_a,
        b: c.button_b,
        x: c.button_x,
        y: c.button_y,
        z: c.button_z,
        l: c.button_l or (pipe_shim and c.l_trigger > @trigger_digital_threshold),
        r: c.button_r or (pipe_shim and c.r_trigger > @trigger_digital_threshold),
        d_up: c.button_d_up
      }
    }
  end

  defp controller_to_input(cs) do
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

  def run_one(entry, inputs, agents, seq, opts) do
    t0 = System.monotonic_time(:millisecond)
    run_dir = Path.join(opts[:run_base], "run#{String.pad_leading(to_string(seq), 3, "0")}")
    File.mkdir_p!(run_dir)

    {:ok, bridge} = MeleePort.start_link()

    config = %{
      dolphin_path: opts[:dolphin],
      iso_path: opts[:iso],
      controller_port: 1,
      opponent_port: 2,
      character: :mewtwo,
      stage: :final_destination,
      online_delay: 0,
      dummy_mode: "external",
      dummy_character: "fox",
      dummy_cpu_level: 0,
      no_audio: true,
      headless: true,
      emulation_speed: 0.0,
      replay_dir: run_dir,
      slippi_port: opts[:slippi_port] + seq
    }

    {agent1, agent2} = agents
    Agent.reset_buffer(agent1)
    Agent.reset_buffer(agent2)

    st = %{
      phase: :menu,
      menu_steps: 0,
      handoff: entry.frame,
      window: opts[:window],
      offset: opts[:input_offset],
      inputs: inputs,
      agent1: agent1,
      agent2: agent2,
      finalize: opts[:finalize],
      finalize_steps: 0,
      truncated: nil
    }

    result =
      case MeleePort.init_console(bridge, config, 180_000) do
        {:ok, _} -> loop(bridge, st)
        :ok -> loop(bridge, st)
        {:error, reason} -> %{error: "init failed: #{inspect(reason)}"}
      end

    try do
      MeleePort.stop(bridge)
    catch
      :exit, _ -> :ok
    end

    if Process.alive?(bridge), do: GenServer.stop(bridge, :normal, 5_000)

    Map.merge(
      %{
        slp: entry.slp,
        frame: entry.frame,
        type: entry.type,
        run_dir: run_dir,
        wall_s: Float.round((System.monotonic_time(:millisecond) - t0) / 1000, 1)
      },
      result
    )
  rescue
    e -> %{slp: entry.slp, frame: entry.frame, error: Exception.message(e)}
  end

  defp loop(bridge, st) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:menu, _} ->
        out_of_game(bridge, st)

      {:postgame, _} ->
        out_of_game(bridge, st)

      {:game_ended, _reason} ->
        case st.phase do
          :finalize -> %{truncated: :finalized}
          :selfplay -> %{truncated: :game_ended}
          _ -> %{error: "game ended during #{st.phase}"}
        end

      {:ok, gs} ->
        st = if st.phase == :menu, do: %{st | phase: :prefix}, else: st

        case st.phase do
          :prefix -> prefix_frame(bridge, gs, st)
          :selfplay -> selfplay_frame(bridge, gs, st)
          :finalize -> finalize_frame(bridge, gs, st)
        end

      {:error, reason} ->
        %{error: "step error: #{inspect(reason)}"}
    end
  end

  defp out_of_game(bridge, st) do
    case st.phase do
      :menu ->
        if st.menu_steps > @menu_step_limit do
          %{error: "stuck in menus"}
        else
          loop(bridge, %{st | menu_steps: st.menu_steps + 1})
        end

      :finalize ->
        %{truncated: :finalized}

      :selfplay ->
        %{truncated: :game_ended}

      :prefix ->
        %{error: "game ended during prefix (divergence?)"}
    end
  end

  defp prefix_frame(bridge, gs, st) do
    f = gs.frame

    if f >= st.handoff do
      selfplay_frame(bridge, gs, %{st | phase: :selfplay})
    else
      case st.inputs[f + st.offset] do
        {p1, p2} ->
          MeleePort.send_controller(bridge, p1)
          MeleePort.send_controller(bridge, Map.put(p2, :port, 2))

        nil ->
          MeleePort.send_controller(bridge, @neutral)
          MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      end

      loop(bridge, st)
    end
  end

  defp selfplay_frame(bridge, gs, st) do
    f = gs.frame

    if f >= st.handoff + st.window do
      if st.finalize do
        finalize_frame(bridge, gs, %{st | phase: :finalize})
      else
        %{truncated: :window_over}
      end
    else
      # BOTH ports are live policies — the interactive part.
      case Agent.get_controller(st.agent1, gs, player_port: 1) do
        {:ok, c} -> MeleePort.send_controller(bridge, controller_to_input(c))
        {:error, _} -> MeleePort.send_controller(bridge, @neutral)
      end

      case Agent.get_controller(st.agent2, gs, player_port: 2) do
        {:ok, c} -> MeleePort.send_controller(bridge, Map.put(controller_to_input(c), :port, 2))
        {:error, _} -> MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      end

      loop(bridge, st)
    end
  end

  defp finalize_frame(bridge, _gs, st) do
    if st.finalize_steps >= @finalize_step_limit do
      %{truncated: :finalize_timeout}
    else
      MeleePort.send_controller(bridge, @suicide)
      MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      loop(bridge, %{st | finalize_steps: st.finalize_steps + 1})
    end
  end
end

# ============================================================================
# CLI
# ============================================================================

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      policy2: :string,
      manifest: :string,
      types: :string,
      only: :string,
      runs: :integer,
      temperature: :float,
      window: :integer,
      uncertainty_log: :string,
      no_finalize: :boolean,
      out: :string,
      dolphin: :string,
      iso: :string,
      slippi_port: :integer,
      input_offset: :integer,
      drift_tolerance: :float,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:policy] do
  Output.error("--policy is required")
  System.halt(1)
end

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_dir = Path.expand(opts[:out] || "seeds/selfplay_#{ts}")
temperature = opts[:temperature] || 0.8
window = opts[:window] || 600
runs = opts[:runs] || 3

run_opts = [
  dolphin: Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
  iso: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  slippi_port: opts[:slippi_port] || 51600,
  input_offset: opts[:input_offset] || 1,
  window: window,
  finalize: not (opts[:no_finalize] || false),
  run_base: Path.join(out_dir, "runs")
]

manifest_path = opts[:manifest] || "scenarios/manifest.json"
manifest = manifest_path |> File.read!() |> Jason.decode!()
type_by_name = Map.new(ScenarioScan.types(), fn t -> {to_string(t), t} end)

wanted_types =
  case opts[:types] do
    nil -> [:idle_deadlock]
    s -> s |> String.split(",", trim: true) |> Enum.map(&Map.fetch!(type_by_name, &1))
  end

entries =
  manifest["entries"]
  |> Enum.map(fn e ->
    %{slp: e["slp"], frame: e["frame"], type: Map.fetch!(type_by_name, e["type"]), note: e["note"]}
  end)
  |> Enum.filter(&(&1.type in wanted_types))

entries =
  case opts[:only] do
    nil ->
      entries

    s ->
      idx = s |> String.split(",") |> Enum.map(&String.to_integer/1) |> MapSet.new()
      entries |> Enum.with_index() |> Enum.filter(fn {_, i} -> i in idx end) |> Enum.map(&elem(&1, 0))
  end

if entries == [] do
  Output.error("No manifest entries after filtering (types: #{inspect(wanted_types)})")
  System.halt(1)
end

Output.banner("Seeded Self-Play Rollouts")

Output.config([
  {"Policy P1", opts[:policy]},
  {"Policy P2", opts[:policy2] || "(same — mirror)"},
  {"Entries", length(entries)},
  {"Runs/entry", runs},
  {"Temperature", temperature},
  {"Window", window},
  {"Finalize", run_opts[:finalize]},
  {"Out", out_dir}
])

# A4: P1's confidences only — P2 mirrors the same policy in mirror mode,
# so logging both would double-count the same uncertainty surface.
{:ok, agent1} =
  Agent.start_link(
    policy_path: opts[:policy],
    temperature: temperature,
    uncertainty_log: opts[:uncertainty_log]
  )

{:ok, agent2} = Agent.start_link(policy_path: opts[:policy2] || opts[:policy], temperature: temperature)

case Agent.warmup(agent1) do
  {:ok, ms} -> Output.success("Agent 1 warmed up (#{ms}ms)")
  {:error, r} -> Output.warning("Agent 1 warmup failed: #{inspect(r)}")
end

case Agent.warmup(agent2) do
  {:ok, ms} -> Output.success("Agent 2 warmed up (#{ms}ms)")
  {:error, r} -> Output.warning("Agent 2 warmup failed: #{inspect(r)}")
end

preps =
  entries
  |> Enum.map(& &1.slp)
  |> Enum.uniq()
  |> Map.new(fn slp -> {slp, SelfplayRollouts.prepare_replay(slp)} end)

work = for entry <- entries, run_idx <- 1..runs, do: {entry, run_idx}

results =
  work
  |> Enum.with_index()
  |> Enum.map(fn {{entry, run_idx}, seq} ->
    Output.puts(
      "-- [#{seq + 1}/#{length(work)}] #{entry.type} @ #{Path.basename(entry.slp)}:#{entry.frame} (run #{run_idx})"
    )

    result = SelfplayRollouts.run_one(entry, preps[entry.slp], {agent1, agent2}, seq, run_opts)

    cond do
      result[:error] -> Output.error("   #{result[:error]}")
      true -> Output.success("   #{result.truncated} #{result.wall_s}s -> #{result[:run_dir]}")
    end

    result
  end)

# Collect finalized rollouts into the seed dir + seed_meta.json (the same
# contract build_seed_dir.exs produces, so dagger_drill's slicing and the
# launcher's NEWERA8_SEED_ROLLOUTS_DIR wiring both apply).
File.mkdir_p!(out_dir)

meta_path = Path.join(out_dir, "seed_meta.json")

existing_meta =
  case File.read(meta_path) do
    {:ok, bin} -> Jason.decode!(bin)
    _ -> %{}
  end

{meta, kept} =
  results
  |> Enum.with_index()
  |> Enum.reduce({existing_meta, 0}, fn {r, seq}, {meta, kept} ->
    with false <- is_binary(r[:error]),
         t when t in [:finalized, :game_ended] <- r[:truncated],
         [slp] <- Path.wildcard(Path.join(r.run_dir, "*.slp")) do
      name = "selfplay_#{ts}_s#{seq}.slp"
      File.cp!(slp, Path.join(out_dir, name))

      entry = %{
        "handoff" => r.frame,
        "window" => window,
        "type" => "selfplay_#{r.type}",
        "score" => nil,
        "pass" => nil,
        "source_slp" => r.slp,
        "selfplay" => true
      }

      {Map.put(meta, name, entry), kept + 1}
    else
      _ -> {meta, kept}
    end
  end)

File.write!(meta_path, Jason.encode!(meta, pretty: true))

errored = Enum.count(results, & &1[:error])
Output.puts("")
Output.puts("rollouts: #{kept} kept, #{errored} errored, #{length(results) - kept - errored} unfinalized")
Output.success("seed dir -> #{out_dir} (#{map_size(meta)} total seeds)")

GenServer.stop(agent1)
GenServer.stop(agent2)

case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} ->
    pids = String.split(out, "\n", trim: true)
    Enum.each(pids, fn pid -> System.cmd("kill", [pid]) end)
    if pids != [], do: Output.warning("Killed #{length(pids)} orphaned Dolphin(s)")

  _ ->
    :ok
end
