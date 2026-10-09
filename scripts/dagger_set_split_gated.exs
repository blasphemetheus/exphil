# Re-cut a DAgger set so every frame list meets the loader's contract
# (`Data.from_frame_lists/2`: an input-only prefix, then a nonempty target
# suffix with no input-only frame inside it). The `--max-d2` coverage gate
# marks frames input-only in the MIDDLE of a trip, which that contract
# rejects ("input-only frames must precede a nonempty target suffix" —
# dag3g_x4_e3, 10-09 00:38). Each maximal run of target frames becomes its
# own list: everything before the run (the trip's own earlier frames, gated
# or not) is kept as input-only context, so the run sees the same history
# it had in the rollout; lists with no target frame are dropped.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_split_gated.exs IN OUT
for app <- [:nx], do: Application.ensure_all_started(app)

[input, output] = System.argv()
set = input |> File.read!() |> :erlang.binary_to_term()

target? = fn f -> f[:input_only] != true end

lists =
  Enum.flat_map(set.frame_lists, fn frames ->
    indexed = Enum.with_index(frames)
    # start indices of maximal target runs
    runs =
      indexed
      |> Enum.chunk_by(fn {f, _} -> target?.(f) end)
      |> Enum.filter(fn [{f, _} | _] -> target?.(f) end)

    Enum.map(runs, fn run ->
      {_, first} = hd(run)
      prefix = frames |> Enum.take(first) |> Enum.map(&Map.put(&1, :input_only, true))
      prefix ++ Enum.map(run, &elem(&1, 0))
    end)
  end)

n_in = set.frame_lists |> List.flatten() |> Enum.count(target?)
n_out = lists |> List.flatten() |> Enum.count(target?)
ctx = lists |> List.flatten() |> Enum.count(&(not target?.(&1)))

out = set |> Map.put(:frame_lists, lists) |> Map.put(:split_gated_from, input)
File.write!(output, :erlang.term_to_binary(out, [:compressed]))

IO.puts("RESULT split gated: #{length(set.frame_lists)} lists -> #{length(lists)} lists; targets #{n_in} -> #{n_out}; input-only context #{ctx} -> #{output}")
