# Controlled experiment: can a grounded shine come out of jumpsquat at all,
# and if so, what input produces it? (Bridge latency measured at 0, so this
# isolates the MECHANIC, not timing.)
#
# Sequence: shine on the ground -> jump-cancel (X at reflector af>=4) ->
# during jumpsquat, attempt the next shine. We run several jumpsquat-shine
# VARIANTS and log every frame's action + on_ground + the input sent, so we
# can SEE which (if any) yields a grounded reflector instead of a full jump.
#
# Variants (--variant):
#   held      : stick held DOWN through jumpsquat, B pressed (the closed_loop
#               approach that fails)
#   flick     : stick NEUTRAL for 1 frame entering jumpsquat, then a fresh
#               DOWN+B flick (tests "held stick doesn't re-trigger")
#   ystick    : shine via stick-down + B with C-stick neutral, but jump via
#               Y held one extra frame
#   cstick_jc : jump-cancel with a full-hop then down+B every frame
#
#   MULTISHINE_HEADLESS=1 mix run scripts/jumpsquat_probe.exs \
#     --dolphin ~/.local/share/slippi/exi-ai/dolphin-emu-headless --iso ~/isos/melee.iso \
#     --variant flick

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.MeleePort
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [dolphin: :string, iso: :string, variant: :string, cycles: :integer])

variant = opts[:variant] || "flick"
cycles = opts[:cycles] || 6

neutral = %{
  main_stick: %{x: 0.5, y: 0.5},
  c_stick: %{x: 0.5, y: 0.5},
  shoulder: 0.0,
  buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
}

down = fn -> %{neutral | main_stick: %{x: 0.5, y: 0.0}} end
downb = fn -> %{down.() | buttons: %{neutral.buttons | b: true}} end
downx = fn -> %{down.() | buttons: %{neutral.buttons | x: true}} end

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

pa = fn gs -> p = gs.players[1]; {trunc((p && p.action) || 0), trunc((p && p.action_frame) || 0), p && p.on_ground} end

name = fn a ->
  cond do
    a in 360..363 -> "gShine"; a in 365..368 -> "aShine"; a == 24 -> "jsquat"
    a in [25, 26, 27] -> "JUMP"; a == 14 -> "wait"; a == 12 -> "fall"; a == 85 -> "sfall"
    true -> "a#{a}"
  end
end

# advance to a standing wait
Enum.reduce_while(1..8000, nil, fn _, _ ->
  case MeleePort.step(bridge, auto_menu: true) do
    {:ok, gs} ->
      {a, _, g} = pa.(gs)
      if a == 14 and g, do: {:halt, :ready}, else: (MeleePort.send_controller(bridge, neutral); {:cont, nil})
    _ -> {:cont, nil}
  end
end)

Output.banner("Jumpsquat-cancel probe — variant=#{variant}")

# The per-frame input policy, a pure function of the observed state, driving
# ONE multishine cycle at a time and logging each frame.
send_and_log = fn input, gs ->
  {a, af, g} = pa.(gs)
  b = input.buttons.b
  x = input.buttons.x
  sy = input.main_stick.y
  Output.puts("  #{name.(a)} af=#{af} gnd=#{g} | send: B=#{b} X=#{x} stickY=#{sy}")
  MeleePort.send_controller(bridge, input)
end

# state machine per variant. prev = last sent input (for edge logic).
downbx = fn -> %{down.() | buttons: %{neutral.buttons | b: true, x: true}} end

# Track whether B was pressed last frame so we can PULSE it (press/release/
# press) to create fresh edges in the air — a held B gives no edge.
{:ok, edge_agent} = Agent.start_link(fn -> false end)
last_b = fn -> Agent.get(edge_agent, & &1) end
set_b = fn v -> Agent.update(edge_agent, fn _ -> v end) end

policy = fn a, af, g, _ ->
  in_ref = a in 360..368

  input =
    cond do
      # grounded reflector, JC window
      in_ref and g and af >= 4 ->
        case variant do
          # BUFFER: jump AND shine on the same JC frame. The buffered B,
          # held into jumpsquat, lands B-down on jumpsquat frame 1 (the
          # earliest possible) — the frame the "react after observing" path
          # misses by one.
          "buffer" -> downbx.()
          _ -> downx.()
        end

      in_ref and g ->
        downb.()

      # jumpsquat — the crux.
      a == 24 ->
        case variant do
          "held" -> downb.()
          "flick" -> if af <= 1, do: down.(), else: downb.()
          # buffer: B already down from the JC frame; keep it down.
          "buffer" -> downb.()
          _ -> downb.()
        end

      # airborne without reflector: PULSE down+B (a held B gives no edge).
      not g and not in_ref ->
        if last_b.(), do: down.(), else: downb.()

      a in 365..368 ->
        down.()

      true ->
        downb.()
    end

  set_b.(input.buttons.b)
  input
end

Enum.reduce_while(1..(cycles * 12), nil, fn _, _ ->
  case MeleePort.step(bridge, auto_menu: true) do
    {:ok, gs} ->
      {a, af, g} = pa.(gs)
      input = policy.(a, af, g, nil)
      send_and_log.(input, gs)
      {:cont, nil}
    _ -> {:halt, :ended}
  end
end)

try do MeleePort.stop(bridge) catch :exit, _ -> :ok end
case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} -> out |> String.split("\n", trim: true) |> Enum.each(&System.cmd("kill", [&1]))
  _ -> :ok
end
