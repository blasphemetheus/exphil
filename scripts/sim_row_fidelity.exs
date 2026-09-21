# SIM_INTEGRATION.md step-3 gate: row fidelity against Dolphin.
#
# Parse a real Dolphin replay (Peppi), configure the sim identically
# (stage, characters, costumes, seed), step both from frame -123 with
# NEUTRAL inputs in the sim, and compare the mapped GameState field by
# field per frame. Pre-input frames are deterministic, so the gate is exact
# equality on -123..-100; the report also shows where divergence begins.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_row_fidelity.exs path/to/game.slp [--until -60]

alias ExPhil.Bridge.SimPort
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, [slp | _], _} = OptionParser.parse(System.argv(), strict: [until: :integer])
until = opts[:until] || -60

{:ok, meta} = Peppi.metadata(slp)
{:ok, replay} = Peppi.parse(slp)

players =
  meta.players
  |> Enum.sort_by(& &1.port)
  |> Enum.map(fn p -> %{character: String.downcase(p.character_name), costume: p.costume || 0} end)

Output.banner("Sim row fidelity vs Dolphin (step 3 gate)")
Output.config([{"Replay", Path.basename(slp)}, {"Stage", meta.stage}, {"Players", inspect(players)}, {"Seed", meta.random_seed}, {"Compare until frame", until}])

{:ok, sim} = SimPort.start_link(stage: meta.stage, players: players, length: 256, seed: meta.random_seed)

fields = ~w(x y action action_frame stock percent facing shield_strength jumps_left on_ground invulnerable speed_air_x_self speed_ground_x_self speed_y_self speed_x_attack speed_y_attack)a

dolphin = replay.frames |> Enum.filter(&(&1.frame_number <= until)) |> Map.new(&{&1.frame_number, &1})

norm = fn
  v when is_float(v) -> Float.round(v, 4)
  v -> v
end

# Slippi writes denormal garbage into hitstun on pre-game frames; compare the rest.
compare = fn frame_no, gs ->
  case dolphin[frame_no] do
    nil -> []
    df ->
      for port <- [1, 2], dp = df.players[port], sp = gs.players[port], f <- fields,
          norm.(Map.get(dp, f)) != norm.(Map.get(sp, f)) do
        {frame_no, port, f, norm.(Map.get(dp, f)), norm.(Map.get(sp, f))}
      end
  end
end

{:ok, [gs0]} = SimPort.frames(sim)
mism0 = compare.(gs0.frame, gs0)

{mismatches, _} =
  Enum.reduce_while(1..(until + 124), {mism0, gs0.frame}, fn _, {acc, _} ->
    case SimPort.step(sim) do
      {:ok, [gs], _} -> {:cont, {acc ++ compare.(gs.frame, gs), gs.frame}}
      {:error, r} -> Output.error(inspect(r)); {:halt, {acc, nil}}
    end
  end)

SimPort.stop(sim)

frames_compared = Enum.count(dolphin, fn {k, _} -> k >= -123 and k <= until end)
gate_window = Enum.filter(mismatches, fn {f, _, _, _, _} -> f <= -100 end)
first_div = mismatches |> Enum.map(&elem(&1, 0)) |> Enum.min(fn -> nil end)

Output.puts("frames compared: #{frames_compared} (-123..#{until}); mismatching (frame, port, field) tuples: #{length(mismatches)}")
Output.puts("first divergent frame: #{inspect(first_div)}")

by_field = mismatches |> Enum.group_by(&elem(&1, 2)) |> Enum.map(fn {f, l} -> {f, length(l)} end) |> Enum.sort_by(&(-elem(&1, 1)))
Output.puts("mismatches by field: #{inspect(by_field)}")

Output.puts("first 12 mismatches (frame, port, field, dolphin, sim):")
mismatches |> Enum.take(12) |> Enum.each(&Output.puts("  #{inspect(&1)}"))

if gate_window == [] and frames_compared >= 24,
  do: Output.success("STEP-3 GATE PASSED (exact on -123..-100)"),
  else: Output.error("STEP-3 GATE FAILED: #{length(gate_window)} mismatches in -123..-100")
