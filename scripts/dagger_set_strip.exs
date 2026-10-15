# Strip relabel-time fields from a DAgger set before training (2026-10-15):
# `:prev_player` (a full Player struct per frame, only the labeler reads it)
# and, unless --keep-actual, `:actual` (the bot's own press — needed only by
# --veto-weight / SilentFallWeighting). r9o's five split parts were 5.25M
# frames; the training beam reached 50 GB RSS loading them. Mix-free.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_strip.exs IN.frames OUT.frames [--keep-actual]
{opts, [input, output], _} = OptionParser.parse(System.argv(), strict: [keep_actual: :boolean])
set = input |> File.read!() |> :erlang.binary_to_term()
drop = if opts[:keep_actual], do: [:prev_player], else: [:prev_player, :actual]
lists = Enum.map(set.frame_lists, fn l -> Enum.map(l, &Map.drop(&1, drop)) end)
out = set |> Map.put(:frame_lists, lists) |> Map.put(:stripped, drop) |> Map.put(:stripped_from, input)
size = :erlang.external_size(out)
if size >= 4_294_967_295, do: raise("set too large to write compressed (#{div(size, 1_000_000)} MB)")
File.write!(output, :erlang.term_to_binary(out, [:compressed]))
IO.puts("RESULT strip #{inspect(drop)}: #{lists |> List.flatten() |> length()} frames, #{div(:erlang.external_size(set), 1_000_000)} -> #{div(size, 1_000_000)} MB -> #{output}")
