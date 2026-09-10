# Re-stamp a policy's layout metadata (2026-09-09).
#
# Why: a policy exported before the source-channel / canary fix carries no
# `with_projectiles` stamp and a canary fingerprinted at the DEFAULT layout,
# so every consumer (Agent, probes, eval_model) rebuilds a 296-wide embedding
# for a 264-wide model (v16f). This rewrites the metadata the way
# Imitation.Checkpoint.export_policy does now, from the checkpoint's own
# embed_size, and re-fingerprints the canary at the corrected layout.
#
#   mix run scripts/restamp_policy.exs --policy checkpoints/X_best_policy.bin \
#     [--with-projectiles false] [--action-frame-buckets N] [--out PATH]
#
# Without --out the file is rewritten in place after a .bak copy.

alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, out: :string, with_projectiles: :string, action_frame_buckets: :integer]
  )

path = opts[:policy] || raise "--policy required"
out = opts[:out] || path

{params, metadata} = Edifice.Checkpoint.load(path, return_metadata: true)
config = metadata[:config] || %{}

with_proj =
  case opts[:with_projectiles] do
    nil -> Map.get(config, :with_projectiles, false)
    s -> s in ["true", "1"]
  end

afb = opts[:action_frame_buckets] || Map.get(config, :action_frame_buckets) || 0

default = ExPhil.Embeddings.Game.Config.default()

canary_config = %{
  default
  | queue_depth: Map.get(config, :queue_depth) || 1,
    with_delay_id: Map.get(config, :with_delay_id) || false,
    stage_internals: Map.get(config, :stage_internals) || false,
    with_projectiles: with_proj,
    with_items: Map.get(config, :with_items) || false,
    player: %{default.player | action_frame_buckets: afb}
}

canary = ExPhil.Embeddings.Canary.fingerprint_batched(canary_config)
width = ExPhil.Embeddings.embedding_size(canary_config)

if config[:embed_size] && config[:embed_size] != width do
  Output.error(
    "corrected layout is #{width} wide but the checkpoint's embed_size is #{config[:embed_size]} — " <>
      "the flags do not describe this model; nothing written"
  )

  System.halt(1)
end

new_config =
  config
  |> Map.put(:with_projectiles, with_proj)
  |> Map.put(:with_items, Map.get(config, :with_items) || false)
  |> Map.put(:provided_channels, ExPhil.Data.Peppi.provides())
  |> Map.put(:action_frame_buckets, afb)
  |> Map.put(:embed_canary, canary)

if out == path, do: File.cp!(path, path <> ".bak")

spec = Edifice.Spec.new(:exphil_policy, Map.to_list(new_config), external: true)
Edifice.Checkpoint.save(params, out, spec: spec, metadata: %{config: new_config})

Output.success(
  "re-stamped #{Path.basename(out)}: with_projectiles=#{with_proj} action_frame_buckets=#{afb} " <>
    "canary #{length(config[:embed_canary] || [])} -> #{length(canary)} dims (embed_size #{config[:embed_size]})"
)
