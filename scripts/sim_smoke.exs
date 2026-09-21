# SIM_INTEGRATION.md step-2 gate: drive a Fox ditto on FD through the sim
# worker for N frames with scripted inputs, count protocol errors, measure
# per-frame round-trip latency, and report the mapped state.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_smoke.exs [--frames 1800] [--batch 1]
#
# Requires the sim main clone built (~/git/msl-main, or EXPHIL_SIM_ROOT).

alias ExPhil.Bridge.{ControllerState, SimPort}
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [frames: :integer, batch: :integer])
frames = opts[:frames] || 1800
batch = opts[:batch] || 1

Output.banner("Sim worker smoke (step 2 gate)")
Output.config([{"Frames", frames}, {"Batch", batch}, {"Sim root", System.get_env("EXPHIL_SIM_ROOT") || "~/git/msl-main"}])

{:ok, sim} =
  SimPort.start_link(
    stage: "final_destination",
    players: [%{character: "fox"}, %{character: "fox", costume: 1}],
    batch_size: batch,
    length: 256,
    seed: 7
  )

{:ok, [gs0 | _]} = SimPort.frames(sim)
p1 = gs0.players[1]
Output.puts("reset: frame #{gs0.frame} stage #{gs0.stage} p1 char #{p1.character} action #{p1.action} at (#{p1.x}, #{p1.y}) facing #{p1.facing}; p2 at #{gs0.players[2].x}")

neutral = %ControllerState{
  main_stick: %{x: 0.0, y: 0.0}, c_stick: %{x: 0.0, y: 0.0}, l_shoulder: 0.0, r_shoulder: 0.0,
  button_a: false, button_b: false, button_x: false, button_y: false, button_z: false,
  button_l: false, button_r: false, button_d_up: false
}

# Scripted P1: dash right for 30 frames, short hop + nair, then dash-dance; P2 idles.
script = fn t ->
  cond do
    t < 30 -> %{neutral | main_stick: %{x: 1.0, y: 0.0}}
    t in 30..31 -> %{neutral | button_y: true}
    t in 36..38 -> %{neutral | button_a: true}
    rem(div(t, 12), 2) == 0 -> %{neutral | main_stick: %{x: 1.0, y: 0.0}}
    true -> %{neutral | main_stick: %{x: -1.0, y: 0.0}}
  end
end

{errors, latencies, actions, last} =
  Enum.reduce(0..(frames - 1), {0, [], %{}, nil}, fn t, {errs, lats, acts, _} ->
    controllers = for _ <- 1..batch, do: [script.(t), neutral]
    t0 = System.monotonic_time(:microsecond)

    case SimPort.step(sim, controllers) do
      {:ok, [gs | _], _term} ->
        dt = System.monotonic_time(:microsecond) - t0
        a = gs.players[1].action
        {errs, [dt | lats], Map.update(acts, a, 1, &(&1 + 1)), gs}

      {:error, reason} ->
        Output.error("step #{t}: #{inspect(reason)}")
        {errs + 1, lats, acts, nil}
    end
  end)

sorted = Enum.sort(latencies)
pct = fn p -> Enum.at(sorted, min(length(sorted) - 1, trunc(p * length(sorted)))) end
mean = if sorted == [], do: 0, else: div(Enum.sum(sorted), length(sorted))

Output.puts("frames stepped: #{length(latencies)}  protocol errors: #{errors}")
Output.puts("round-trip latency (us): mean #{mean}  p50 #{pct.(0.5)}  p95 #{pct.(0.95)}  p99 #{pct.(0.99)}  max #{List.last(sorted)}")

if last do
  p = last.players[1]
  Output.puts("final: frame #{last.frame} p1 action #{p.action} af #{p.action_frame} at (#{Float.round(p.x, 2)}, #{Float.round(p.y, 2)}) on_ground #{p.on_ground} jumps #{p.jumps_left}; distance #{Float.round(last.distance, 2)}")
end

top = actions |> Enum.sort_by(fn {_, n} -> -n end) |> Enum.take(8)
Output.puts("p1 action histogram (top 8): #{inspect(top)}")

{:ok, saved} = SimPort.save(sim, 0)
{:ok, _} = SimPort.restore(sim, 0, saved)
Output.puts("save/restore round trip: #{byte_size(saved)} bytes")

gate = errors == 0 and length(latencies) == frames and map_size(actions) > 3
if gate, do: Output.success("STEP-2 GATE PASSED"), else: Output.error("STEP-2 GATE FAILED")
SimPort.stop(sim)
