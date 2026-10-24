# Dose-matched NULL set (2026-10-24): the same DAgger windows as a relabelled
# set, with the LABEL replaced by the bot's own press (`:actual`) — so an
# epoch mixed with it has the exact dose of the relabel arm and none of the
# expert labels. If the null arm regresses like the relabel arm did
# (r9o/dag9o, 10-16), the harm is the mixing mechanics, not the labels.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_null.exs IN_split.frames OUT_null_lean.frames
#
# Reads a keep-actual `_split.frames` (the lean sets stripped :actual), sets
# controller := actual on every frame that has one, prev_controller := the
# previous frame's actual (the bot's own history, which is what its state
# saw), strips :prev_player / :actual like dagger_set_strip.exs, and writes.
# CPU only, no Mix, no GPU.

[input, out] = System.argv()
set = :erlang.binary_to_term(File.read!(input))
lists = set.frame_lists

{lists2, n_lab, n_changed} =
  Enum.reduce(lists, {[], 0, 0}, fn list, {acc, n_lab, n_changed} ->
    {rev, _prev, nl, nc} =
      Enum.reduce(list, {[], nil, 0, 0}, fn f, {rev, prev_actual, nl, nc} ->
        actual = Map.get(f, :actual)
        labelled = Map.get(f, :input_only) != true
        f2 =
          if actual != nil do
            f
            |> Map.put(:controller, actual)
            |> then(fn m -> if prev_actual, do: Map.put(m, :prev_controller, prev_actual), else: m end)
          else
            f
          end
        f2 = Map.drop(f2, [:prev_player, :actual])
        changed = actual != nil and f[:controller] != actual
        {[f2 | rev], actual || prev_actual, nl + if(labelled, do: 1, else: 0), nc + if(labelled and changed, do: 1, else: 0)}
      end)
    {[Enum.reverse(rev) | acc], n_lab + nl, n_changed + nc}
  end)

out_set =
  set
  |> Map.put(:frame_lists, Enum.reverse(lists2))
  |> Map.put(:stripped, [:prev_player, :actual])
  |> Map.put(:stripped_from, input)
  |> Map.put(:null_labels, true)
  |> Map.put(:expert, "null: label = the bot's own press (dose-matched control)")
  |> Map.put(:exported_at, DateTime.utc_now() |> DateTime.to_iso8601())

File.write!(out, :erlang.term_to_binary(out_set, [:compressed]))
IO.puts("RESULT null set: #{length(lists)} lists, #{n_lab} labelled frames, #{n_changed} labels differed from the actual press (#{Float.round(100 * n_changed / max(n_lab, 1), 1)} %) -> #{out} (#{div(File.stat!(out).size, 1_000_000)} MB)")
