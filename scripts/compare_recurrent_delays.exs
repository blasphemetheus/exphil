# Run with scripts/run_recurrent_validation.exs; no training or default changes.
alias ExPhil.Training.Output
[out | rest] = System.argv()

{opts, [], []} =
  OptionParser.parse(rest, strict: [limit: :integer, shard: :integer, shards: :integer])

shard = Keyword.get(opts, :shard, 0)
shards = Keyword.get(opts, :shards, 1)
if shards < 1 or shard < 0 or shard >= shards, do: raise("invalid shard")
File.mkdir_p!(out)

min_gru = "checkpoints/coh_prev_q_ck4/model_best_policy.bin"
mamba = "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin"
gru = "checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin"

# Three one-factor contrasts from fd3_min_gru: delay, stage and opponent.
contexts = [
  {"fd3_min_gru", "final_destination", 3, min_gru},
  {"fd4_min_gru", "final_destination", 4, min_gru},
  {"bf3_min_gru", "battlefield", 3, min_gru},
  {"fd3_gru", "final_destination", 3, gru}
]

arms =
  for {arch, policy} <- [{"min_gru", min_gru}, {"mamba", mamba}],
      {context, stage, delay, opponent} <- contexts,
      seed <- 1101..1104,
      mode <- ["windowed", "carried", "resync30"] do
    label = "#{arch}_#{context}_#{mode}_s#{seed}"
    path = Path.join(out, label <> ".json")

    flags =
      case mode do
        "windowed" -> []
        "carried" -> ["--stateful-step"]
        "resync30" -> ["--stateful-step", "--stateful-resync", "30"]
      end

    {path,
     [
       "--policy",
       policy,
       "--label",
       label,
       "--opponent",
       opponent,
       "--character",
       "fox",
       "--opponent-character",
       "fox",
       "--envs",
       "8",
       "--frames",
       "1800",
       "--seed",
       to_string(seed),
       "--stage",
       stage,
       "--frame-delay",
       to_string(delay),
       "--opponent-frame-delay",
       to_string(delay),
       "--out",
       path
     ] ++ flags}
  end

pending =
  arms
  |> Enum.with_index()
  |> Enum.filter(fn {_, index} -> rem(div(index, 12), shards) == shard end)
  |> Enum.map(&elem(&1, 0))
  |> Enum.reject(fn {path, _} -> File.exists?(path) end)

pending = if opts[:limit], do: Enum.take(pending, opts[:limit]), else: pending

for {{path, args}, index} <- Enum.with_index(pending, 1) do
  Output.puts("Delay screen #{index}/#{length(pending)}: #{Path.basename(path)}")
  System.argv(args)
  Code.eval_file("scripts/sim_closed_loop.exs")
  :erlang.garbage_collect()
end
