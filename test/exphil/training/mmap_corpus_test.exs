defmodule ExPhil.Training.MmapCorpusTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.MmapCorpus

  @embed 4

  defp tmp_dir(context) do
    dir =
      Path.join(
        System.tmp_dir!(),
        "mmap_corpus_test_#{context.test |> to_string() |> String.replace(~r/[^a-zA-Z0-9]/, "_")}_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(dir) end)
    dir
  end

  defp action(mask_seed) do
    %{
      buttons: %{
        a: rem(mask_seed, 2) == 1,
        b: rem(mask_seed, 3) == 1,
        x: false,
        y: rem(mask_seed, 5) == 1,
        z: false,
        l: false,
        r: rem(mask_seed, 7) == 1,
        d_up: false
      },
      main_x: rem(mask_seed, 17),
      main_y: rem(mask_seed + 3, 17),
      c_x: 8,
      c_y: 8,
      shoulder: rem(mask_seed, 4)
    }
  end

  # Embeddings where frame i's values are all i (easy to verify slices)
  defp embeddings(start, n) do
    Nx.iota({n, 1}, type: :f32)
    |> Nx.add(start)
    |> Nx.broadcast({n, @embed})
  end

  defp build!(dir, files) do
    writer = MmapCorpus.create_writer(dir, @embed, %{note: "test"})

    writer =
      Enum.reduce(files, writer, fn {path, start, n}, w ->
        MmapCorpus.append!(w, embeddings(start, n), Enum.map(start..(start + n - 1), &action/1), path, 1)
      end)

    MmapCorpus.finalize!(writer)
    writer
  end

  describe "label packing" do
    test "pack/decode round-trips" do
      for seed <- 0..40 do
        a = action(seed)
        assert MmapCorpus.decode_label(MmapCorpus.pack_label(a), 0) == a
      end
    end
  end

  describe "writer + reader round trip" do
    test "two files append and reopen consistently", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}, {"b.slp", 10, 7}])

      corpus = MmapCorpus.open!(dir)
      assert corpus.num_frames == 17
      assert corpus.embed_size == @embed
      assert Enum.map(corpus.files, & &1.path) == ["a.slp", "b.slp"]
      assert Enum.map(corpus.files, & &1.start) == [0, 10]

      # Frame 12 should embed to all-12s and carry action(12)
      frame = ExPhil.Training.MmapEmbeddings.read_frame(corpus.embd, 12)
      assert Nx.to_flat_list(frame) == List.duplicate(12.0, @embed)
      assert MmapCorpus.decode_label(corpus.labels, 12) == action(12)

      MmapCorpus.close(corpus)
    end

    test "sequence starts never cross file boundaries", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}, {"b.slp", 10, 7}, {"tiny.slp", 17, 3}])
      corpus = MmapCorpus.open!(dir)

      starts = MmapCorpus.sequence_starts(corpus.files, 5, 2)
      # file a: 0,2,4 (last valid 5) — wait: last = 10-5 = 5, so 0,2,4 (6 > 5)
      # file b: 10,12 (last valid 12)
      # tiny (3 < window): none
      assert starts == [0, 2, 4, 10, 12]

      MmapCorpus.close(corpus)
    end

    test "batched_sequences yields correct windows and labels", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}, {"b.slp", 10, 7}])
      corpus = MmapCorpus.open!(dir)

      {stream, n} =
        MmapCorpus.batched_sequences(corpus, corpus.files,
          window_size: 5,
          stride: 2,
          batch_size: 2,
          shuffle: false,
          drop_last: false,
          gpu: false
        )

      batches = Enum.to_list(stream)
      assert n == 3
      assert length(batches) == 3

      [b1, _b2, b3] = batches
      assert Nx.shape(b1.states) == {2, 5, @embed}

      # First window starts at frame 0: rows are 0..4
      assert b1.states[0] |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.to_flat_list() ==
               [0.0, 1.0, 2.0, 3.0, 4.0]

      # Its label is the LAST frame of the window (frame 4): action(4)
      a4 = action(4)
      assert Nx.to_number(b1.actions.main_x[0]) == a4.main_x
      assert Nx.to_number(b1.actions.shoulder[0]) == a4.shoulder

      # Last batch: single window starting at 12 (file b), label frame 16
      assert Nx.shape(b3.states) == {1, 5, @embed}
      a16 = action(16)
      assert Nx.to_number(b3.actions.main_x[0]) == a16.main_x

      # frame_weights present, one per sequence
      assert Nx.shape(b1.frame_weights) == {2}

      MmapCorpus.close(corpus)
    end

    test "split_files holds out whole trailing files", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}, {"b.slp", 10, 7}, {"c.slp", 17, 8}])
      corpus = MmapCorpus.open!(dir)

      {train, val} = MmapCorpus.split_files(corpus, 0.34)
      assert Enum.map(train, & &1.path) == ["a.slp", "b.slp"]
      assert Enum.map(val, & &1.path) == ["c.slp"]

      MmapCorpus.close(corpus)
    end
  end

  describe "resume" do
    test "reopened writer skips processed files and appends new ones", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}])

      writer = MmapCorpus.create_writer(dir, @embed)
      assert MmapCorpus.processed?(writer, "a.slp")
      refute MmapCorpus.processed?(writer, "b.slp")
      assert writer.num_frames == 10

      writer = MmapCorpus.append!(writer, embeddings(10, 7), Enum.map(10..16, &action/1), "b.slp", 2)
      MmapCorpus.finalize!(writer)

      corpus = MmapCorpus.open!(dir)
      assert corpus.num_frames == 17
      assert Enum.map(corpus.files, & &1.port) == [1, 2]
      assert MmapCorpus.decode_label(corpus.labels, 16) == action(16)
      MmapCorpus.close(corpus)
    end

    test "reopen truncates a torn tail (crash mid-append)", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 10}])

      # Simulate a crash: partial embedding/label bytes + torn manifest line
      File.write!(Path.join(dir, "embeddings.bin"), <<1, 2, 3, 4, 5>>, [:append])
      File.write!(Path.join(dir, "labels.bin"), <<9, 9, 9>>, [:append])
      File.write!(Path.join(dir, "manifest.jsonl"), ~s({"path": "torn.slp", "sta), [:append])

      writer = MmapCorpus.create_writer(dir, @embed)
      assert writer.num_frames == 10
      refute MmapCorpus.processed?(writer, "torn.slp")

      writer = MmapCorpus.append!(writer, embeddings(10, 4), Enum.map(10..13, &action/1), "c.slp", 1)
      MmapCorpus.finalize!(writer)

      corpus = MmapCorpus.open!(dir)
      assert corpus.num_frames == 14
      assert Enum.map(corpus.files, & &1.path) == ["a.slp", "c.slp"]

      frame = ExPhil.Training.MmapEmbeddings.read_frame(corpus.embd, 13)
      assert Nx.to_flat_list(frame) == List.duplicate(13.0, @embed)
      MmapCorpus.close(corpus)
    end

    test "embed size mismatch on reopen raises", context do
      dir = tmp_dir(context)
      build!(dir, [{"a.slp", 0, 4}])

      assert_raise ArgumentError, ~r/embed_size/, fn ->
        MmapCorpus.create_writer(dir, @embed + 1)
      end
    end
  end

  describe "button_rates" do
    test "computes press rates from packed labels", context do
      dir = tmp_dir(context)
      # actions 0..9: a pressed when odd → 5/10
      build!(dir, [{"a.slp", 0, 10}])
      corpus = MmapCorpus.open!(dir)

      rates = MmapCorpus.button_rates(corpus)
      assert_in_delta rates.a, 0.5, 1.0e-9
      assert rates.x == 0.0

      MmapCorpus.close(corpus)
    end
  end
end
