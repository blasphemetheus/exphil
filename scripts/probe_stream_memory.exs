# Probe: streaming chunk-prep memory + timing, via the REAL create_dataset path
# (0825 57GB-for-10-files hunt). Usage: mix run scripts/probe_stream_memory.exs N_FILES
alias ExPhil.Training.{Streaming, Data, Output}

n = case System.argv() do
  [n] -> String.to_integer(n)
  _ -> 2
end

rss = fn label ->
  {out, 0} = System.cmd("cat", ["/proc/self/status"])
  [kb] = Regex.run(~r/VmRSS:\s+(\d+) kB/, out, capture: :all_but_first)
  Output.puts("RSS @ #{label}: #{div(String.to_integer(kb), 1024)} MB")
end

files =
  Path.wildcard("replays/erickfm_ranked/FOX/extracted/*.slp")
  |> Enum.take(n)

Output.puts("Probing #{length(files)} files (real Streaming.create_dataset path)")
rss.("start")

t0 = System.monotonic_time(:millisecond)
{:ok, frames, errors} = Streaming.parse_chunk(files, show_progress: true)
t1 = System.monotonic_time(:millisecond)
Output.puts("parse_chunk: #{length(frames)} frames, #{length(errors)} errors, #{t1 - t0} ms")
rss.("after parse")

embed_config = ExPhil.Embeddings.config(stage_internals: true)

dataset =
  Streaming.create_dataset(frames,
    temporal: true,
    window_size: 60,
    stride: 5,
    embed_config: embed_config,
    show_progress: true
  )

t2 = System.monotonic_time(:millisecond)
Output.puts("create_dataset (lazy temporal): #{t2 - t1} ms, embedded shape #{inspect(Nx.shape(dataset.embedded_frames))}")
rss.("after create_dataset")

batches = Data.batched_sequences(dataset,
  batch_size: 256,
  shuffle: true,
  drop_last: true,
  lazy: true,
  window_size: 60,
  stride: 5
)

{batch, count} =
  batches
  |> Enum.reduce({nil, 0}, fn b, {first, c} -> {first || b, c + 1} end)

t3 = System.monotonic_time(:millisecond)
Output.puts("batched_sequences lazy: #{count} batches, first states shape #{inspect(Nx.shape(batch.states))}, #{t3 - t2} ms")
rss.("after batching")
Output.puts("TOTAL: #{t3 - t0} ms")
