# First per-port INPUT and STATE mismatch between two replays of "the same" game
# (a source and the scenario suite's re-recording of its prefix), frames -39..UPTO.
#   mix run --no-start scripts/compare_replay_inputs.exs SOURCE.slp RERUN.slp 340
# 2026-09-14: showed CPU-AI stick floats are unreplayable (0.0039 -> 0.0062 at
# frame -39) while the bot's bucketed inputs match on every frame.

alias ExPhil.Data.Peppi
[a, b, upto] = System.argv()
upto = String.to_integer(upto)
load = fn p -> {:ok, r} = Peppi.parse(p); Map.new(r.frames, &{&1.frame_number, &1}) end
fa = load.(a); fb = load.(b)
ctl = fn c -> {Float.round(c.main_stick_x, 4), Float.round(c.main_stick_y, 4), Float.round(c.c_stick_x, 4), Float.round(c.c_stick_y, 4), Float.round(c.l_trigger, 3), Float.round(c.r_trigger, 3), c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r} end
st = fn p -> {Float.round(p.x, 2), Float.round(p.y, 2), trunc(p.action), trunc(p.action_frame)} end
for port <- [1, 2] do
  first_input = Enum.find(-39..upto, fn f -> fa[f] && fb[f] && ctl.(fa[f].players[port].controller) != ctl.(fb[f].players[port].controller) end)
  first_state = Enum.find(-39..upto, fn f -> fa[f] && fb[f] && st.(fa[f].players[port]) != st.(fb[f].players[port]) end)
  IO.puts("port #{port}: first INPUT mismatch at #{inspect(first_input)}, first STATE mismatch at #{inspect(first_state)}")
  for f <- [first_input, first_state], f != nil do
    IO.puts("  f#{f} src  in=#{inspect(ctl.(fa[f].players[port].controller))} st=#{inspect(st.(fa[f].players[port]))}")
    IO.puts("  f#{f} rerun in=#{inspect(ctl.(fb[f].players[port].controller))} st=#{inspect(st.(fb[f].players[port]))}")
  end
end
