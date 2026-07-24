# Measure the bridge's true input->effect latency, empirically.
#
# From a neutral standing Fox, send a single-frame down+B (shine) and record
# how many frames later the action state becomes the grounded reflector
# (360..363). That number is the round-trip latency of send_controller ->
# console.step flush -> Dolphin poll -> observed action, and it's the
# foundation for any latency compensation. Repeat a few times for a stable
# estimate.
#
#   MULTISHINE_HEADLESS=1 mix run scripts/latency_probe.exs \
#     --dolphin ~/.local/share/slippi/exi-ai/dolphin-emu-headless --iso ~/isos/melee.iso

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.MeleePort
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [dolphin: :string, iso: :string, trials: :integer])

trials = opts[:trials] || 4

neutral = %{
  main_stick: %{x: 0.5, y: 0.5},
  c_stick: %{x: 0.5, y: 0.5},
  shoulder: 0.0,
  buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
}

# down + B (shine input)
shine = %{neutral | main_stick: %{x: 0.5, y: 0.0}, buttons: %{neutral.buttons | b: true}}

{:ok, bridge} = MeleePort.start_link()

{:ok, _} =
  MeleePort.init_console(bridge, %{
    dolphin_path: Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
    iso_path: Path.expand(opts[:iso] || "~/isos/melee.iso"),
    controller_port: 1,
    opponent_port: 2,
    character: :fox,
    stage: :final_destination,
    dummy_mode: "stand",
    dummy_character: "fox",
    dummy_cpu_level: 0,
    headless: System.get_env("MULTISHINE_HEADLESS") == "1",
    no_audio: true
  })

Output.banner("Bridge input-latency probe")

action = fn gs -> gs.players[1] && trunc(gs.players[1].action || 0) end
reflector? = fn a -> a in 360..363 end

# Advance to a settled in-game standing state.
settle = fn f ->
  Enum.reduce_while(1..6000, nil, fn _, _ ->
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, gs} ->
        a = action.(gs)
        # Wait (14) or standing, on ground, not already in reflector
        if a && a in [14] && gs.players[1].on_ground and gs.frame > f do
          {:halt, {:ok, gs.frame}}
        else
          MeleePort.send_controller(bridge, neutral)
          {:cont, nil}
        end

      _ ->
        {:cont, nil}
    end
  end)
end

# One trial: send a single-frame shine, then neutral, counting steps until
# the action becomes a reflector. Returns the latency in frames.
trial = fn ->
  # send the shine on THIS step's input slot
  MeleePort.send_controller(bridge, shine)

  Enum.reduce_while(1..30, {0, :sent}, fn _, {n, phase} ->
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, gs} ->
        a = action.(gs)

        cond do
          reflector?.(a) ->
            {:halt, {:latency, n}}

          true ->
            # keep neutral after the one-frame shine
            MeleePort.send_controller(bridge, neutral)
            {:cont, {n + 1, phase}}
        end

      _ ->
        {:halt, {:error, :step}}
    end
  end)
end

Output.puts("settling to a standing state...")

results =
  Enum.map(1..trials, fn i ->
    settle.(0)

    case trial.() do
      {:latency, n} ->
        Output.puts("trial #{i}: shine appeared #{n} step(s) after send")
        n

      other ->
        Output.warning("trial #{i}: #{inspect(other)}")
        nil
    end
  end)
  |> Enum.reject(&is_nil/1)

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> :ok
end

Output.divider()

if results != [] do
  mean = Enum.sum(results) / length(results)
  Output.puts("INPUT LATENCY: #{inspect(results)} steps  (mean #{Float.round(mean, 2)})")
  Output.puts("=> an input decided from frame N lands on frame N+#{round(mean)}")
else
  Output.error("no clean trials")
end

case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} -> out |> String.split("\n", trim: true) |> Enum.each(&System.cmd("kill", [&1]))
  _ -> :ok
end
