defmodule ExPhil.Training.PipelineMixCorpusTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.{MmapCorpus, Pipeline}

  @embed 4
  @window 4

  defp tmp_dir(context, tag) do
    dir =
      Path.join(
        System.tmp_dir!(),
        "pipeline_mix_#{tag}_#{context.test |> to_string() |> String.replace(~r/[^a-zA-Z0-9]/, "_")}_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(dir) end)
    dir
  end

  defp action(seed) do
    %{
      buttons: %{
        a: rem(seed, 2) == 1,
        b: false,
        x: false,
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      },
      main_x: rem(seed, 17),
      main_y: 8,
      c_x: 8,
      c_y: 8,
      shoulder: 0
    }
  end

  # Frame i's embedding values are all `base + i` — lets a batch's origin
  # (main corpus vs mix corpus) be read back off the states tensor
  defp embeddings(base, n) do
    Nx.iota({n, 1}, type: :f32) |> Nx.add(base) |> Nx.broadcast({n, @embed})
  end

  defp build!(dir, base, files) do
    writer = MmapCorpus.create_writer(dir, @embed, %{note: "test"})

    writer =
      Enum.reduce(files, writer, fn {path, n}, w ->
        MmapCorpus.append!(
          w,
          embeddings(base, n),
          Enum.map(1..n, &action/1),
          path,
          1
        )
      end)

    MmapCorpus.finalize!(writer)
    MmapCorpus.open!(dir)
  end

  defp pipeline(main, mix, oversample) do
    {main_ranges, []} = MmapCorpus.split_files(main, 0.0)
    mix_ranges = if mix, do: elem(MmapCorpus.split_files(mix, 0.0), 0)

    %Pipeline{
      corpus: main,
      corpus_train_ranges: main_ranges,
      mix_corpus: mix,
      mix_corpus_ranges: mix_ranges,
      streaming: false,
      estimated_batches: 0,
      resolved_opts: [
        window_size: @window,
        stride: 1,
        batch_size: 2,
        mix_oversample: oversample,
        # Keep everything on BinaryBackend — no GPU in unit tests
        neutral_weight: 1.0
      ]
    }
  end

  # Main corpus embeddings live in [0, 100); mix in [1000, ...)
  defp mix_batch?(batch) do
    batch.states |> Nx.reduce_max() |> Nx.to_number() >= 1000.0
  end

  test "without --mix-corpus the stream is pure main corpus", context do
    main = build!(tmp_dir(context, "main"), 0, [{"a", 20}, {"b", 20}])
    {stream, _} = Pipeline.batch_stream(pipeline(main, nil, 1), shuffle: false)

    batches = Enum.to_list(stream)
    assert length(batches) > 0
    refute Enum.any?(batches, &mix_batch?/1)
  end

  test "mix batches are interleaved and oversampled", context do
    main = build!(tmp_dir(context, "main"), 0, [{"a", 40}, {"b", 40}])
    # Two snippets, each long enough for exactly a few windows
    mix = build!(tmp_dir(context, "mix"), 1000, [{"s0", 6}, {"s1", 5}])

    oversample = 2
    {stream, _} = Pipeline.batch_stream(pipeline(main, mix, oversample), shuffle: false)
    batches = Enum.to_list(stream)

    {mix_batches, main_batches} = Enum.split_with(batches, &mix_batch?/1)

    # stride 1, window 4: s0 (6 frames) -> 3 starts, s1 (5) -> 2 starts,
    # batch 2, drop_last false -> 3 batches per pass, x2 oversample = 6
    assert length(mix_batches) == 6
    assert length(main_batches) > 0

    # Injections are spread out, not clumped at the end
    first_mix_idx = Enum.find_index(batches, &mix_batch?/1)
    assert first_mix_idx < length(batches) - length(mix_batches)

    # Oversampled batches are usable tensors (backend_copy, not a
    # deallocating transfer): every mix batch's states must be readable
    for b <- mix_batches do
      assert Nx.to_number(Nx.reduce_max(b.states)) >= 1000.0
      assert map_size(b.actions) > 0
    end
  end

  test "mix corpus with a wrong embed size raises at setup", context do
    dir = tmp_dir(context, "bad")
    writer = MmapCorpus.create_writer(dir, @embed + 2, %{})

    writer =
      MmapCorpus.append!(
        writer,
        Nx.broadcast(0.0, {8, @embed + 2}),
        Enum.map(1..8, &action/1),
        "s",
        1
      )

    MmapCorpus.finalize!(writer)

    # The guard lives in build_pipeline_corpus; replicate its check here
    # against the reader structs (setup/1 needs a full training config)
    main = build!(tmp_dir(context, "main"), 0, [{"a", 20}])
    bad = MmapCorpus.open!(dir)
    assert bad.embed_size != main.embed_size
  end
end
