# Merge a trained PPO head back into its frozen prior and write a PLAYABLE
# policy checkpoint — the artifact you hand to `play_dolphin_async.exs`.
#
#   devenv shell -- mix run scripts/ppo_export_policy.exs \
#     --policy checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin \
#     --head eval_runs/0924_mewtwo_ppo/v1/head_iter150.bin \
#     --out checkpoints/mewtwo_ppo_v1_iter150_policy.bin
#
# The head is only the autoregressive `ar_*` layers; the trunk is untouched, so
# the export is the prior's params with those layers replaced. Provenance on the
# head is checked against --policy first: merging a head into the wrong trunk
# produces a file that loads fine and plays nothing like either parent.

alias ExPhil.Sim.PPO
alias ExPhil.Training.{Checkpoint, Output}

{opts, _, invalid} =
  OptionParser.parse(System.argv(), strict: [policy: :string, head: :string, out: :string])

if invalid != [], do: raise("Invalid options: #{inspect(invalid)}")
policy = opts[:policy] || raise("--policy required")
head_path = opts[:head] || raise("--head required")
out = opts[:out] || raise("--out required")

head = head_path |> File.read!() |> :erlang.binary_to_term()
ar = head[:ar] || raise("no :ar layers in #{head_path}")

if head[:policy] && head[:policy] != policy do
  raise "head was trained on #{head[:policy]} but --policy is #{policy}"
end

Output.banner("Export PPO head → playable policy")
Output.config([
  {"Prior", policy},
  {"Head", "#{Path.basename(head_path)} (iter #{head[:iter]}, #{map_size(ar)} ar_* layers)"},
  {"Character", head[:character] || "unrecorded"},
  {"Out", out}
])

{:ok, export} = Checkpoint.load_policy(policy)

merged =
  case export.params do
    %Axon.ModelState{} = ms -> %{ms | data: Map.merge(ms.data, ar)}
    map -> Map.merge(map, ar)
  end

File.mkdir_p!(Path.dirname(out))
File.write!(out, :erlang.term_to_binary(%{export | params: PPO.to_backend(merged, Nx.BinaryBackend)}))

# Reload it: a checkpoint that cannot be read back is not an artifact.
{:ok, _} = Checkpoint.load_policy(out)
Output.success("#{Float.round(File.stat!(out).size / 1_048_576, 1)} MB → #{out} (reloads cleanly)")
