# Keep only the NEAR-EDGE labels of a wide-window DAgger set (2026-10-10
# 04:30, queue 41): every target frame that the v3 window would also have
# labelled (offstage, airborne — `labelable?/2`) becomes input-only context,
# so what is left supervised is exactly what the wide window added
# (grounded / above-stage states within near_edge of the edge). Re-cut with
# dagger_set_split_gated.exs afterwards (input-only frames now sit mid-trip).
# Why: dag6w (r6w alone) gave the best closed loop of the program but lost
# the offstage press that r345g's three rounds carry; adding all of r6w to
# r345g is 256k = past the dose cliff. r345g + the near-edge frames of a few
# seeds stays at the ×1 dose.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_keep_near_edge.exs IN OUT
for app <- [:nx], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L

[input, output] = System.argv()
set = input |> File.read!() |> :erlang.binary_to_term()
edge = ExPhil.Sim.GA.stage_edge(32)
target? = fn f -> f[:input_only] != true end

lists =
  Enum.map(set.frame_lists, fn frames ->
    Enum.map(frames, fn f ->
      if target?.(f) and L.labelable?(f.game_state.players[1], edge), do: Map.put(f, :input_only, true), else: f
    end)
  end)

n_in = set.frame_lists |> List.flatten() |> Enum.count(target?)
n_out = lists |> List.flatten() |> Enum.count(target?)
grounded = lists |> List.flatten() |> Enum.count(&(target?.(&1) and &1.game_state.players[1].on_ground == true))
out = set |> Map.put(:frame_lists, lists) |> Map.put(:near_edge_only_from, input)
File.write!(output, :erlang.term_to_binary(out, [:compressed]))
IO.puts("RESULT keep near-edge: targets #{n_in} -> #{n_out} (#{grounded} grounded); offstage labels -> input-only -> #{output}")
