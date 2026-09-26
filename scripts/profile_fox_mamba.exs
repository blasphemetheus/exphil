# Real Fox replay, full AR training step and sequential controller sampling.
# Run only with no other training active. Each configuration is a fresh VM.
# EXPHIL_EXLA_PRECISION=highest mix run scripts/profile_fox_mamba.exs \
#   --hidden 256 --batch 64 --window 80 --steps 20 --out eval_runs/0925_fox_mamba/h256_b64
alias ExPhil.Training.{Data, Imitation, Output, Utils}
alias ExPhil.Networks.Policy
alias ExPhil.Data.Peppi

defmodule FoxMambaProfile do
  def timed(fun) do
    {us, result} = :timer.tc(fun)
    {us / 1000, result}
  end

  def sync(%Nx.Tensor{} = value), do: (Nx.to_binary(value); value)
  def sync(value) when is_map(value), do: (Enum.each(value, fn {_, v} -> sync(v) end); value)
  def sync(value) when is_tuple(value), do: (value |> Tuple.to_list() |> Enum.each(&sync/1); value)
  def sync(value) when is_list(value), do: (Enum.each(value, &sync/1); value)
  def sync(value), do: value
  def count(%Nx.Tensor{} = t), do: Nx.size(t)
  def count(m) when is_map(m), do: m |> Map.values() |> Enum.map(&count/1) |> Enum.sum()
  def count(_), do: 0

  def stats(xs) do
    sorted = Enum.sort(xs)
    %{median: Enum.at(sorted, div(length(sorted), 2)),
      p95: Enum.at(sorted, min(length(sorted) - 1, ceil(length(sorted) * 0.95) - 1)),
      min: hd(sorted), max: List.last(sorted), count: length(sorted)}
  end
end

alias FoxMambaProfile, as: P
{opts, _, invalid} = OptionParser.parse(System.argv(), strict: [hidden: :integer,
  batch: :integer, window: :integer, steps: :integer, out: :string])
if invalid != [], do: raise("invalid options: #{inspect(invalid)}")
hidden = opts[:hidden] || 256
batch_size = opts[:batch] || 64
window = opts[:window] || 80
steps = opts[:steps] || 20
out = Keyword.fetch!(opts, :out)
File.mkdir_p!(out)
Output.banner("Fox Mamba full-policy profile")
Output.config([{"Hidden", hidden}, {"Batch", batch_size}, {"Window", window}, {"Steps", steps}])

paths = File.stream!("replays/erickfm_ranked/v2_filtered/manifest.jsonl")
  |> Stream.map(&Jason.decode!/1) |> Stream.filter(&(&1["verdict"] == "keep"))
  |> Stream.map(& &1["path"]) |> Enum.take(2)

{parse_ms, frames} = P.timed(fn ->
  Enum.flat_map(paths, fn path ->
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    if is_nil(own), do: raise("not a Fox replay: #{path}")
    opponent = Enum.find(meta.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(path)
    replay |> Peppi.to_training_frames(player_port: own.port, opponent_port: opponent.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
  end)
end)
{embedding_ms, dataset} = P.timed(fn ->
  frames |> Data.from_frames(embed_config: ExPhil.Embeddings.config_for_source([stage_internals: true], Peppi.provides()))
    |> Data.precompute_frame_embeddings()
end)
model_opts = [embed_config: dataset.embed_config,
  embed_size: ExPhil.Embeddings.embedding_size(dataset.embed_config), temporal: true,
  backbone: :mamba, head: :autoregressive, precision: :f32, window_size: window, seed: 905,
  hidden_size: hidden, num_layers: 2, state_size: 16, expand_factor: 2, conv_size: 4,
  learning_rate: 2.0e-4, max_grad_norm: 1.0, dropout: 0.0, label_smoothing: 0.0,
  button_weight: 1.0, button_pos_weight: Nx.broadcast(1.0, {8}), stick_edge_weight: 1.0,
  focal_loss: false, entropy_weight: 0.0, neutral_weight: 1.0, label_delay: 0]
trainer = Imitation.new(model_opts)
count = trainer.policy_params |> Utils.ensure_model_state() |> Map.fetch!(:data)
params = P.count(count)

batch_opts = [batch_size: batch_size, window_size: window, stride: 1,
  lazy: true, shuffle: true, drop_last: true, seed: 905, neutral_weight: 1.0]
# Time actual enumeration/materialization, unlike timing an already-built batch.
{batch_ms, batches} = P.timed(fn ->
  dataset |> Data.batched_sequences(batch_opts) |> Enum.take(steps + 3)
  |> Enum.map(&P.sync/1)
end)
if length(batches) < steps + 3, do: raise("too few batches")
if System.get_env("MAMBA_CAPTURE") == "1" do
  :ok = Imitation.save_checkpoint(trainer, Path.join(out, "before_step.axon"))
  host = fn
    %Nx.Tensor{} = t -> Nx.backend_copy(t, Nx.BinaryBackend)
    m -> Map.new(m, fn {k, t} -> {k, Nx.backend_copy(t, Nx.BinaryBackend)} end)
  end
  batch = Map.new(hd(batches), fn {k, v} -> {k, host.(v)} end)
  File.write!(Path.join(out, "batch.bin"), :erlang.term_to_binary(batch))
  Output.puts("Saved pre-step checkpoint and input batch before GPU training")
end
Edifice.CUDA.AutoTuneProfiler.start()
Output.puts("JIT compiling full training step; first step may take several minutes")
{compile_ms, {trainer, first}} = P.timed(fn ->
  {t, m} = Imitation.train_step(trainer, hd(batches), nil)
  P.sync(m)
  {t, m}
end)
{times, trainer, losses} = Enum.reduce(tl(batches), {[], trainer, []}, fn batch, {ts, tr, ls} ->
  {ms, {next, metrics}} = P.timed(fn ->
    {t, m} = Imitation.train_step(tr, batch, nil)
    P.sync(m)
    {t, m}
  end)
  loss = Nx.to_number(metrics.loss)
  unless is_number(loss), do: raise("nonfinite loss: #{inspect(loss)}")
  Output.puts("step #{length(ts) + 1}: #{Float.round(ms, 2)} ms loss=#{loss}")
  {ts ++ [ms], next, ls ++ [loss]}
end)
Edifice.CUDA.AutoTuneProfiler.report()
Edifice.CUDA.AutoTuneProfiler.stop()

trunk = Policy.build_temporal_trunk(model_opts)
{_, predict} = Utils.build_compiled(trunk)
input = Nx.slice_along_axis(hd(batches).states, 0, 1, axis: 0)
sample = fn ->
  Policy.sample_autoregressive(trainer.policy_params, predict, input,
    temperature: 1.0, deterministic: false) |> P.sync()
end
Output.puts("JIT compiling batch-one trunk + sequential autoregressive sampler")
{infer_compile_ms, _} = P.timed(sample)
Enum.each(1..3, fn _ -> sample.() end)
infer_times = Enum.map(1..50, fn _ -> elem(P.timed(sample), 0) end)
features = predict.(trainer.policy_params, input) |> Nx.to_flat_list()
Imitation.export_policy(trainer, Path.join(out, "smoke_policy.bin"))
row = %{hidden: hidden, batch: batch_size, window: window, params: params,
  fused_available: Edifice.CUDA.FusedScan.custom_call_available?(), seed: 905,
  exla_precision: System.get_env("EXPHIL_EXLA_PRECISION"),
  frames: length(frames), replay_paths: paths, embed_size: model_opts[:embed_size],
  parse_ms: parse_ms, embedding_ms: embedding_ms,
  batch_materialization_ms: batch_ms / length(batches),
  compile_ms: compile_ms, train_ms: P.stats(Enum.drop(times, 2)),
  inference_compile_ms: infer_compile_ms, inference_ms: P.stats(infer_times),
  first_loss: Nx.to_number(first.loss), last_loss: List.last(losses),
  features: features,
  note: "Windowed Mamba, f32, full AR controller sampler; excludes live game-state embedding/Agent mailbox. Two-replay development profile, no style conditioning."}
File.write!(Path.join(out, "profile.json"), Jason.encode!(row, pretty: true))
Output.success(inspect(Map.delete(row, :features), pretty: true))
