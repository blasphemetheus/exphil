# Replay selected ORIGINAL corpus chunks; no full-corpus scan or remapping drift.
alias ExPhil.Training.{Imitation, Data, Streaming, ChunkPipeline, Output}
alias ExPhil.Data.{Peppi, SubjectResolver}
[out, start_s, count_s, mode] = System.argv()
File.mkdir_p!(out)
start = String.to_integer(start_s)
count = String.to_integer(count_s)
files = "checkpoints/fox_mamba_v1_20260925/split.json" |> File.read!() |> Jason.decode!() |> Map.fetch!("train")
chunks = files |> Enum.chunk_every(64) |> Enum.slice(start - 1, count)
port_map = for path <- List.flatten(chunks), into: %{} do
  {:ok, meta} = Peppi.metadata(path)
  {:ok, %{subject_port: port}} = SubjectResolver.resolve(meta.players,
    subject_character: 2, ditto_tie_break: :port1)
  {path, port}
end
cfg = ExPhil.Embeddings.config_for_source([stage_internals: true], Peppi.provides())
chunk_opts = [port_map: port_map, player_port: 1, label_delay: 0, subject_character: "fox"]
dataset_opts = [temporal: true, window_size: 80, stride: 5, precompute: true, embed_config: cfg]
File.write!(Path.join(out, "chunks.json"), Jason.encode!(chunks))
trainer = Imitation.new(embed_config: cfg, embed_size: 264, temporal: true,
  backbone: :mamba, head: :autoregressive, precision: :f32, window_size: 80,
  hidden_size: 512, num_layers: 2, state_size: 16, expand_factor: 2, conv_size: 4,
  seed: 905, dropout: 0.0, learning_rate: 0.0002, max_grad_norm: 1.0,
  focal_loss: false, button_weight: 1.0, button_pos_weight: Nx.tensor(List.duplicate(1.0, 8), backend: Nx.BinaryBackend),
  neutral_weight: 1.0, stick_edge_weight: 1.0, label_delay: 0)
:ok = Imitation.save_checkpoint(trainer, Path.join(out, "initial.axon"))
stream = if mode == "parallel" do
  ChunkPipeline.stream_prepared_chunks(chunks, chunk_opts: chunk_opts, dataset_opts: dataset_opts)
else
  chunks |> Stream.with_index(1) |> Stream.map(fn {paths, i} ->
    Output.puts("Preparing serial chunk #{i}")
    {:ok, frames, errors} = Streaming.parse_chunk(paths, chunk_opts)
    {Streaming.create_dataset(frames, dataset_opts), i, errors}
  end)
end
{trainer, steps} = Enum.reduce(stream, {trainer, 0}, fn {dataset, chunk, errors}, {tr, step} ->
  if errors != [], do: raise("parse errors: #{inspect(errors)}")
  :ok = Imitation.save_checkpoint(tr, Path.join(out, "chunk_#{chunk}_start.axon"))
  Output.puts("Training chunk #{chunk}, #{dataset.size} frames, step #{step}")
  :rand.seed(:exsss, {905, 905, 905})
  dataset |> Data.batched_sequences(batch_size: 128, window_size: 80, stride: 5,
    lazy: true, shuffle: true, drop_last: true, neutral_weight: 1.0, seed: 905)
  |> Enum.with_index()
  |> Enum.reduce({tr, step}, fn {batch, idx}, {t, s} ->
    File.write!(Path.join(out, "last_step.json"), Jason.encode!(%{step: s, chunk: chunk,
      original_chunk: start + chunk - 1, batch_index: idx, seed: 905}))
    if rem(s, 100) == 0 do
      # Exact paired pre-step snapshot, bounded to two slots.
      slot = rem(div(s, 100), 2)
      :ok = Imitation.save_checkpoint(t, Path.join(out, "capture#{slot}.axon"))
      host = Map.new(batch, fn
        {k, %Nx.Tensor{} = v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)}
        {k, v} when is_map(v) -> {k, Map.new(v, fn {a, x} -> {a, Nx.backend_copy(x, Nx.BinaryBackend)} end)}
        pair -> pair
      end)
      File.write!(Path.join(out, "capture#{slot}_batch.bin"), :erlang.term_to_binary(host))
    end
    {next, m} = Imitation.train_step(t, batch, nil)
    loss = Nx.to_number(m.loss)
    unless is_number(loss), do: raise("nonfinite loss at #{s}")
    if rem(s, 100) == 0, do: Output.puts("step #{s}, chunk #{chunk}, batch #{idx}, loss #{loss}")
    if rem(s, 100) == 0, do: :erlang.garbage_collect()
    {next, s + 1}
  end)
end)
:ok = Imitation.save_checkpoint(trainer, Path.join(out, "completed.axon"))
File.write!(Path.join(out, "completed.json"), Jason.encode!(%{steps: steps, mode: mode, start: start, chunks: count}))
Output.success("Completed #{steps} steps across #{count} chunks")
