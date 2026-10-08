# Label hazards inside a DAgger set (mix-free, CPU only — no EXLA):
#   elixir -pa _build/dev/lib/*/ebin scripts/dagger_set_label_hazards.exs data/silent_fall/sim_dagger_expert_r1.frames
# Answers "does the relabelled set carry the recovery decisions on the bot's
# own states?": per relabelled frame, label-vs-prev edges for jump (jump in
# hand, y < -20), stick-up onset and B edge (jump spent, y < -20), next to
# the policy's actual input on the same frames when the set kept it.
[path | _] = System.argv()
set = path |> File.read!() |> :erlang.binary_to_term()
frames = set.frame_lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true))

own = fn f -> f.game_state.players[1] end
jump? = fn c -> c.button_x or c.button_y end
up? = fn c -> (c.main_stick[:y] || 0.5) >= 0.75 end
b? = fn c -> c.button_b end
edge? = fn f, pick, g -> g.(pick.(f)) and not g.(f.prev_controller) end
pct = fn k, d -> if d == 0, do: "-", else: "#{Float.round(100 * k / d, 2)} % (#{k}/#{d})" end
label = & &1.controller

spent = Enum.filter(frames, fn f -> p = own.(f); (p.jumps_left || 0) == 0 and (p.y || 0.0) < -20.0 and not up?.(f.prev_controller) end)
in_hand = Enum.filter(frames, fn f -> p = own.(f); (p.jumps_left || 0) > 0 and (p.y || 0.0) < -20.0 and not jump?.(f.prev_controller) end)
deep = Enum.filter(spent, fn f -> (own.(f).y || 0.0) < -40.0 end)

IO.puts("RESULT set #{path}: #{length(frames)} relabelled frames; spent y<-20 #{length(spent)} (deep <-40 #{length(deep)}), jump-in-hand y<-20 #{length(in_hand)}")
IO.puts("RESULT label jump edge (in hand, y<-20): #{pct.(Enum.count(in_hand, &edge?.(&1, label, jump?)), length(in_hand))}   expert held-out 8.76 %")
IO.puts("RESULT label stick-up onset (spent, y<-20): #{pct.(Enum.count(spent, &edge?.(&1, label, up?)), length(spent))}   expert held-out 3.1 %")
IO.puts("RESULT label stick-up onset (spent, y<-40): #{pct.(Enum.count(deep, &edge?.(&1, label, up?)), length(deep))}")
IO.puts("RESULT label B edge (spent, y<-20): #{pct.(Enum.count(spent, &edge?.(&1, label, b?)), length(spent))}   expert held-out 0.24 %")
IO.puts("RESULT label stick-up share (spent, y<-20): #{pct.(Enum.count(spent, &up?.(label.(&1))), length(spent))}   expert loop 42.5 %")
