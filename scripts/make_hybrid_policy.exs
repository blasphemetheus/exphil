# Head/trunk swap surgery (mechanism hunt for the d1 delay-preference
# inversion, 2026-07-29): splice the recurrent TRUNK of one exported
# policy with the controller HEADS of another.
#
# The encode-horizon probe (interp_d1_timing.exs) found NO trunk-level
# timing shift between the champion and the d1-DAgger policy — pointing
# at the heads (the d1 teacher's rules are the same {action, af} features
# with shifted thresholds, which a head decision boundary can implement
# alone). The decisive test: if champion-trunk + dagger-heads inherits
# the delay preference, the adaptation is head-level.
#
#   mix run scripts/make_hybrid_policy.exs \
#     --trunk-from checkpoints/ms_open_z.bin \
#     --heads-from checkpoints/ms_d1_dagger3_policy.bin \
#     --out checkpoints/hybrid_champTrunk_d1Heads.bin
#
# Trunk = "input_ln" + "<backbone>_*" layers (same partition the agent's
# stateful step path uses — see agent.ex trunk_step_params). Heads =
# everything else. Donors must agree on architecture (validated key-by-key
# on shapes); config/spec are taken from the TRUNK donor.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.Checkpoint
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [trunk_from: :string, heads_from: :string, out: :string]
  )

trunk_path = opts[:trunk_from] || raise "--trunk-from required"
heads_path = opts[:heads_from] || raise "--heads-from required"
out_path = opts[:out] || raise "--out required"

Output.banner("Hybrid policy splice")
Output.config([{"Trunk from", trunk_path}, {"Heads from", heads_path}, {"Out", out_path}])

load = fn path ->
  case Checkpoint.load_policy(path) do
    {:ok, export} -> export
    %{params: _} = export -> export
    {:error, reason} -> raise "load failed for #{path}: #{inspect(reason)}"
  end
end

trunk_export = load.(trunk_path)
heads_export = load.(heads_path)

raw = fn
  %Axon.ModelState{data: data} -> data
  %{} = m -> m
end

trunk_raw = raw.(trunk_export.params)
heads_raw = raw.(heads_export.params)

backbone = trunk_export.config[:backbone] || trunk_export.config["backbone"] || :gru
prefix = "#{backbone}_"

trunk_key? = fn k -> is_binary(k) and (k == "input_ln" or String.starts_with?(k, prefix)) end

# The trunks may differ structurally (e.g. champion 1 GRU layer, drill
# default 2) — the hybrid takes the trunk WHOLESALE from its donor, so
# only the HEAD partitions must agree: same layer names, same shapes
# (both feed on [_, hidden_size] features, so hidden_size must match).
{trunk_layers, _} = trunk_raw |> Map.keys() |> Enum.sort() |> Enum.split_with(trunk_key?)
{_, head_layers} = heads_raw |> Map.keys() |> Enum.sort() |> Enum.split_with(trunk_key?)
{_, trunk_donor_heads} = trunk_raw |> Map.keys() |> Enum.sort() |> Enum.split_with(trunk_key?)

if head_layers != trunk_donor_heads do
  raise """
  head layer sets differ — heads are not interchangeable:
    only in trunk donor: #{inspect(trunk_donor_heads -- head_layers)}
    only in heads donor: #{inspect(head_layers -- trunk_donor_heads)}
  """
end

shapes = fn layer_params ->
  layer_params |> Enum.map(fn {k, v} -> {k, Nx.shape(v)} end) |> Map.new()
end

for k <- head_layers do
  sa = shapes.(trunk_raw[k])
  sb = shapes.(heads_raw[k])

  sa == sb or
    raise "head shape mismatch in #{k}: #{inspect(sa)} vs #{inspect(sb)} — hidden_size differs?"
end

keys_a = trunk_layers ++ head_layers

Output.puts("Trunk layers (#{length(trunk_layers)}): #{Enum.join(trunk_layers, ", ")}")
Output.puts("Head layers  (#{length(head_layers)}): #{Enum.join(head_layers, ", ")}")

if trunk_layers == [] or head_layers == [] do
  raise "degenerate split — check backbone prefix (#{prefix})"
end

to_bin = fn layer_params ->
  Map.new(layer_params, fn {k, v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)} end)
end

merged_raw =
  Map.new(keys_a, fn k ->
    src = if trunk_key?.(k), do: trunk_raw, else: heads_raw
    {k, to_bin.(src[k])}
  end)

merged_params =
  case trunk_export.params do
    %Axon.ModelState{} = ms -> %{ms | data: merged_raw}
    _ -> merged_raw
  end

config = trunk_export.config
# Always rebuild the spec: a loaded spec round-trips as a plain map (its
# struct atoms may not be interned), which Checkpoint.save rejects.
spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config), external: true)

Edifice.Checkpoint.save(merged_params, out_path, spec: spec, metadata: %{config: config})

Output.success("Hybrid written: #{out_path}")
Output.puts("  trunk <- #{Path.basename(trunk_path)}")
Output.puts("  heads <- #{Path.basename(heads_path)}")
