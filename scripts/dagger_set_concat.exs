# Concatenate DAgger sets (same export shape: %{frame_lists: [...], ...}) into
# one --mix-frames file; mix-free, CPU only.
#   elixir -pa _build/dev/lib/*/ebin scripts/dagger_set_concat.exs IN1 IN2 ... OUT
args = System.argv()
if length(args) < 2, do: raise("usage: dagger_set_concat.exs IN1 [IN2 ...] OUT")
{ins, [out]} = Enum.split(args, -1)
sets = Enum.map(ins, fn p -> p |> File.read!() |> :erlang.binary_to_term() end)
[first | _] = sets
for s <- sets, s.label_convention != first.label_convention, do: raise("label conventions differ")
lists = Enum.flat_map(sets, & &1.frame_lists)
merged = %{first | frame_lists: lists, exported_at: DateTime.utc_now() |> DateTime.to_iso8601()}
  |> Map.put(:sources, ins)
File.mkdir_p!(Path.dirname(out))
File.write!(out, :erlang.term_to_binary(merged, [:compressed]))
n = lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true)) |> length()
IO.puts("RESULT concat: #{length(ins)} sets -> #{length(lists)} runs, #{n} relabelled frames -> #{out}")
