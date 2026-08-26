# Export per-epoch policy .bin snapshots from trainer .axon checkpoints,
# for behavioral sweeps over a train.exs run (which saves trainer .axon
# per epoch but only exports _policy.bin for best/final).
#
# Usage:
#   mix run scripts/export_epoch_policies.exs checkpoints/fox_gen_v1_20260825_205855
#
# For each <prefix>_epochN.axon writes <prefix>_epN.bin (skips ones that
# already exist). The embed config is rebuilt from the checkpoint's own
# training config via the full-opts pass-through (Embeddings.config/1
# whitelists internally) — this matters: a nil embed_config would make
# the canary embed at DEFAULT layout (288) while the params are e.g.
# 296-wide, and the live Agent prefers canary length. Guard #6 in
# export_policy validates the width against the params either way.
#
# NO-MIX LAW: run only when no training beam is live.

alias ExPhil.Training.Output

prefix =
  case System.argv() do
    [p | _] -> p
    _ ->
      IO.puts(:stderr, "usage: mix run scripts/export_epoch_policies.exs <checkpoint-prefix>")
      System.halt(2)
  end

paths = Path.wildcard("#{prefix}_epoch*.axon") |> Enum.sort()

if paths == [] do
  IO.puts(:stderr, "no #{prefix}_epoch*.axon found")
  System.halt(1)
end

Output.banner("Epoch policy export")
Output.puts("#{length(paths)} trainer checkpoints at #{prefix}")

for path <- paths do
  [ep] = Regex.run(~r/epoch(\d+)\.axon$/, path, capture: :all_but_first)
  out = "#{prefix}_ep#{ep}.bin"

  if File.exists?(out) do
    Output.puts("  ep#{ep}: exists, skipping")
  else
    %{policy_params: policy_params, config: config} =
      path |> File.read!() |> :erlang.binary_to_term()

    embed_config = ExPhil.Embeddings.config(Map.to_list(config))

    trainer = %{policy_params: policy_params, config: config, embed_config: embed_config}

    case ExPhil.Training.Imitation.Checkpointing.export_policy(trainer, out) do
      :ok -> Output.success("  ep#{ep} -> #{out}")
      other -> Output.error("  ep#{ep} FAILED: #{inspect(other)}")
    end
  end
end
