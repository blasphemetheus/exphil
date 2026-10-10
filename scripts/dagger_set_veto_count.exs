# Count the VETO frames of a DAgger set rolled with --keep-actual (2026-10-10):
# supervised frames where the bot's own input pressed X/Y/B against its own
# previous input and the expert label holds it released — the frames
# `--veto-weight` lifts (SilentFallWeighting.veto?/3). Also the near-edge
# facing-out subset (the on-stage Illusion precursor, INPUT_COHERENCE
# "10-10 09:10") so the queue log carries the number the mechanism rests on.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_veto_count.exs SET.frames
alias ExPhil.Training.SilentFallWeighting, as: SFW

[path] = System.argv()
set = path |> File.read!() |> :erlang.binary_to_term()
edge = ExPhil.Sim.GA.stage_edge(32)
target? = fn f -> f[:input_only] != true end

vetoes =
  for frames <- set.frame_lists, [a, b] <- Enum.chunk_every(frames, 2, 1, :discard),
      target?.(b), b.game_state.frame == a.game_state.frame + 1,
      SFW.veto?(a[:actual], b[:actual], b.controller),
      do: b

near_out? = fn f ->
  p = f.game_state.players[1]
  sign = if (p.x || 0.0) >= 0, do: 1, else: -1
  d = abs(p.x || 0.0) - edge
  d > -25 and d < 5 and (p.facing || 1) * sign > 0
end

n_t = set.frame_lists |> List.flatten() |> Enum.count(target?)
with_actual = set.frame_lists |> List.flatten() |> Enum.count(&(target?.(&1) and &1[:actual] != nil))
by_b = Enum.count(vetoes, &(&1.actual.button_b and not &1.controller.button_b))
near = Enum.filter(vetoes, near_out?)
IO.puts("RESULT veto count #{Path.basename(path)}: targets #{n_t} (with actual #{with_actual}); veto frames #{length(vetoes)} (B #{by_b}, jump #{length(vetoes) - by_b}); near-edge facing-out #{length(near)} (B #{Enum.count(near, & &1.actual.button_b)})")
