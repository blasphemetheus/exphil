# Relabel an existing keep-actual DAgger set with another index / labeler
# version (2026-10-16): the bot's states and its own presses (`:actual`) are
# the rollout; only the labels change. Each frame list of the input (one
# relabel_history run: input-only context + a labelled run, gated frames
# inside) is rebuilt as a history `[{game_state, actual}]` and relabelled
# with `ExpertRecoveryLabeler.relabel_history/4`, so a new labeler version
# (v8: offstage dim; `--prev actual`: no label chain) is measured on the SAME
# r9o states without a fresh rollout. The lists keep `:actual`.
#
#   mix run --no-compile scripts/dagger_set_relabel.exs --index IDX.bin --in SET.frames --out NEW.frames \
#     [--max-d2 G] [--max-run 90] [--prev chain|actual] [--label-seed 12] [--report r.json]
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [index: :string, in: :string, out: :string, max_d2: :float, max_run: :integer, prev: :string, label_seed: :integer, report: :string])
index = L.load(opts[:index] || raise("--index required"))
input = opts[:in] || raise("--in required")
out = opts[:out] || raise("--out required")
edge = GA.stage_edge(32)
prev_mode = String.to_atom(opts[:prev] || "chain")
seed = opts[:label_seed] || 12
:rand.seed(:exsss, {seed, seed, seed})
Output.banner("DAgger set relabel (#{index.n} rows, window #{index.window}, prev #{prev_mode})")

set = input |> File.read!() |> :erlang.binary_to_term()
if set[:keep_actual] != true and not Enum.all?(List.flatten(set.frame_lists), &Map.has_key?(&1, :actual)), do: raise("#{input} has no :actual presses (not a keep-actual set)")
Output.config([{"In", input}, {"Lists", length(set.frame_lists)}, {"max_d2", inspect(opts[:max_d2])}, {"max_run", inspect(opts[:max_run])}, {"prev", prev_mode}, {"Out", out}])

n_lists = length(set.frame_lists)
# every list as a history (newest first, own player = port 1 — the set's
# game_states carry own_port 1), all relabelled in one lockstep
histories = Enum.map(set.frame_lists, fn l -> l |> Enum.map(&{&1.game_state, &1[:actual] || &1.controller}) |> Enum.reverse() end)
per_history = L.relabel_histories(histories, index, edge, max_d2: opts[:max_d2], max_run: opts[:max_run], prefix: 90, prev: prev_mode)

# A list's context frames (its first 90 f, input-only, not gated) are the
# previous list's run in the same env history — relabelling them here would
# label every frame twice. Labels stay only on the list's ORIGINAL run span
# (labelled or gated frames); context frames go back to input-only with the
# bot's own press. Lists left with no labelled frame are dropped.
lists =
  Enum.zip(set.frame_lists, per_history)
  |> Enum.flat_map(fn {orig, new_lists} ->
    run_frames = orig |> Enum.filter(&(&1[:input_only] != true or &1[:gated] == true)) |> MapSet.new(& &1.game_state.frame)

    new_lists
    |> Enum.map(fn l ->
      [nil | l]
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, f] ->
        if MapSet.member?(run_frames, f.game_state.frame),
          do: f,
          else: f |> Map.merge(%{input_only: true, controller: f.actual, prev_controller: if(a, do: a.actual, else: f.prev_controller)}) |> Map.delete(:gated)
      end)
    end)
    |> Enum.filter(fn l -> Enum.any?(l, &(&1[:input_only] != true)) end)
  end)

flat = List.flatten(lists)
total = Enum.count(flat, &(&1[:input_only] != true))
gated = Enum.count(flat, &(&1[:gated] == true))
holds = Enum.count(flat, fn f -> f[:input_only] != true and f.controller == f.prev_controller end)
Output.puts("RESULT relabel #{Path.basename(input)}: #{n_lists} lists -> #{length(lists)}; labelled #{total} (was #{Enum.count(List.flatten(set.frame_lists), &(&1[:input_only] != true))}), gated #{gated} (#{Float.round(100 * gated / max(total + gated, 1), 1)} %), hold share #{Float.round(holds / max(total, 1), 3)}")

new = set |> Map.merge(%{frame_lists: lists, relabelled_from: input, index: opts[:index], prev: prev_mode, max_d2: opts[:max_d2], max_run: opts[:max_run], label_seed: seed, keep_actual: true})
File.write!(out, :erlang.term_to_binary(new, [:compressed]))
Output.puts("-> #{out}")
if opts[:report], do: File.write!(opts[:report], Jason.encode!(%{in: input, out: out, index: opts[:index], prev: prev_mode, lists: length(lists), labelled: total, gated: gated, hold_share: holds / max(total, 1)}, pretty: true))
