# Drive the multishine POLICY live and capture, per frame, its action AND the
# controller input the AGENT emitted — then find its longest GROUNDED shine
# run and dump exactly what it pressed. The policy runs through the SAME
# bridge my probes did, so if it grounded-multishines better than my
# hand-crafted attempts, its inputs reveal the pattern I was missing.
#
#   MULTISHINE_HEADLESS=1 mix run scripts/analyze_policy_shine.exs \
#     --policy checkpoints/multishine_daggerloop_20260708_112952_i1_policy.bin \
#     --dolphin ~/.local/share/slippi/exi-ai/dolphin-emu-headless --iso ~/isos/melee.iso

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, dolphin: :string, iso: :string, seconds: :integer, deterministic: :boolean])

seconds = opts[:seconds] || 40

neutral = %{
  main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, shoulder: 0.0,
  buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
}

to_input = fn cs ->
  %{
    main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
    c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
    shoulder: cs.l_shoulder + cs.r_shoulder,
    buttons: %{a: cs.button_a, b: cs.button_b, x: cs.button_x, y: cs.button_y,
      z: cs.button_z, l: cs.button_l, r: cs.button_r, d_up: cs.button_d_up}
  }
end

{:ok, agent} = Agent.start_link(policy_path: opts[:policy], deterministic: opts[:deterministic] || false)
Agent.warmup(agent)
{:ok, bridge} = MeleePort.start_link()

{:ok, _} =
  MeleePort.init_console(bridge, %{
    dolphin_path: Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
    iso_path: Path.expand(opts[:iso] || "~/isos/melee.iso"),
    controller_port: 1, opponent_port: 2, character: :fox, stage: :final_destination,
    dummy_mode: "stand", dummy_character: "fox", dummy_cpu_level: 0,
    headless: System.get_env("MULTISHINE_HEADLESS") == "1", no_audio: true
  })

Output.banner("Analyze POLICY multishine (live capture)")

# Collect {action, af, on_ground, sent_input} per in-game frame.
loop = fn loop, n, acc ->
  if n > seconds * 60 do
    Enum.reverse(acc)
  else
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, gs} ->
        p = gs.players[1]

        input =
          case Agent.get_controller(agent, gs, player_port: 1) do
            {:ok, c} -> to_input.(c)
            _ -> neutral
          end

        MeleePort.send_controller(bridge, input)
        # P2 stand dummy holds itself
        rec = if p, do: {trunc(p.action || 0), trunc(p.action_frame || 0), p.on_ground, input}, else: nil
        loop.(loop, n + 1, (if rec, do: [rec | acc], else: acc))

      s when elem(s, 0) in [:menu] ->
        loop.(loop, n + 1, acc)

      _ ->
        Enum.reverse(acc)
    end
  end
end

recs = loop.(loop, 0, [])
try do MeleePort.stop(bridge) catch :exit, _ -> :ok end

acts = Enum.map(recs, fn {a, _, _, _} -> a end)
s = ShineChain.summary(acts)
Output.divider()
Output.puts("POLICY: grounded_frac=#{inspect(s.grounded_fraction)} clean_max=#{inspect(s.max_length)} sustained=#{s.sustained}")
Output.puts("chain histogram=#{inspect(Enum.sort(s.length_histogram))}  ended_by=#{inspect(s.ended_by)}")

name = fn a -> cond do
  a in 360..363 -> "gShine"; a in 365..368 -> "aShine"; a==24 -> "jsquat"
  a in [25,26,27] -> "JUMP"; a==14 -> "wait"; a==12 -> "fall"; a==85 -> "sfall"
  a in [40,41,42,43] -> "land"; true -> "a#{a}" end end

# Find the longest run of grounded-shine/jumpsquat and dump its inputs.
in_loop = fn a -> a in 360..363 or a == 24 end
indexed = Enum.with_index(recs)
runs =
  indexed
  |> Enum.chunk_by(fn {{a, _, _, _}, _} -> in_loop.(a) end)
  |> Enum.filter(fn [{{a, _, _, _}, _} | _] -> in_loop.(a) end)

best = Enum.max_by(runs, &length/1, fn -> [] end)

if best != [] do
  {_, start_i} = hd(best)
  Output.puts("\n=== inputs during the policy's longest grounded run (len #{length(best)}) ===")
  recs
  |> Enum.slice(max(start_i - 3, 0), 24)
  |> Enum.each(fn {a, af, g, c} ->
    b = c.buttons
    Output.puts("#{String.pad_trailing(name.(a), 8)} af=#{af} gnd=#{g} | B=#{b.b} X=#{b.x} Y=#{b.y} Z=#{b.z} sY=#{Float.round(c.main_stick.y, 2)} cY=#{Float.round(c.c_stick.y, 2)}")
  end)
else
  Output.puts("no grounded shine run found")
end

GenServer.stop(agent)
case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} -> out |> String.split("\n", trim: true) |> Enum.each(&System.cmd("kill", [&1]))
  _ -> :ok
end
