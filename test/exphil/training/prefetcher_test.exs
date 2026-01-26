defmodule ExPhil.Training.PrefetcherTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Prefetcher

  describe "each/2" do
    test "iterates over all batches" do
      results = Agent.start_link(fn -> [] end) |> elem(1)

      Prefetcher.each([1, 2, 3], fn batch ->
        Agent.update(results, &[batch | &1])
      end)

      assert Agent.get(results, & &1) |> Enum.reverse() == [1, 2, 3]
    end

    test "handles empty list" do
      assert Prefetcher.each([], fn _ -> :never_called end) == :ok
    end

    test "handles single element" do
      results = Agent.start_link(fn -> [] end) |> elem(1)

      Prefetcher.each([42], fn batch ->
        Agent.update(results, &[batch | &1])
      end)

      assert Agent.get(results, & &1) == [42]
    end
  end

  describe "reduce/3" do
    test "reduces over all batches" do
      result =
        Prefetcher.reduce([1, 2, 3], 0, fn batch, acc ->
          acc + batch
        end)

      assert result == 6
    end

    test "handles empty list" do
      result = Prefetcher.reduce([], :initial, fn _, _ -> :never_called end)
      assert result == :initial
    end

    test "handles single element" do
      result = Prefetcher.reduce([10], 5, fn batch, acc -> acc + batch end)
      assert result == 15
    end

    test "processes batches in order" do
      result =
        Prefetcher.reduce([1, 2, 3, 4], [], fn batch, acc ->
          [batch | acc]
        end)

      # Reversed because we prepend
      assert Enum.reverse(result) == [1, 2, 3, 4]
    end
  end

  describe "reduce_indexed/4" do
    test "provides correct indices" do
      result =
        Prefetcher.reduce_indexed([:a, :b, :c], [], fn batch, idx, acc ->
          [{batch, idx} | acc]
        end)

      assert Enum.reverse(result) == [{:a, 0}, {:b, 1}, {:c, 2}]
    end

    test "handles empty list" do
      result = Prefetcher.reduce_indexed([], :initial, fn _, _, _ -> :never_called end)
      assert result == :initial
    end

    test "handles single element" do
      result =
        Prefetcher.reduce_indexed([42], [], fn batch, idx, acc ->
          [{batch, idx} | acc]
        end)

      assert result == [{42, 0}]
    end

    test "accumulator works correctly" do
      result =
        Prefetcher.reduce_indexed([10, 20, 30], 0, fn batch, idx, acc ->
          acc + batch + idx
        end)

      # 0 + (10 + 0) = 10
      # 10 + (20 + 1) = 31
      # 31 + (30 + 2) = 63
      assert result == 63
    end
  end

  describe "wrap/2" do
    test "wraps enumerable with prefetching" do
      wrapped = Prefetcher.wrap([1, 2, 3, 4, 5])
      result = Enum.to_list(wrapped)

      assert result == [1, 2, 3, 4, 5]
    end

    test "respects buffer_size option" do
      # This primarily tests that the option is accepted without error
      wrapped = Prefetcher.wrap([1, 2, 3, 4, 5], buffer_size: 3)
      result = Enum.to_list(wrapped)

      assert result == [1, 2, 3, 4, 5]
    end

    test "handles empty enumerable" do
      wrapped = Prefetcher.wrap([])
      result = Enum.to_list(wrapped)

      assert result == []
    end
  end

  describe "stream/2" do
    test "creates stream from generator function" do
      # Generator that counts from 1 to 3
      counter = Agent.start_link(fn -> 0 end) |> elem(1)

      generator = fn ->
        current = Agent.get_and_update(counter, fn n -> {n + 1, n + 1} end)
        if current <= 3, do: {:ok, current}, else: :done
      end

      stream = Prefetcher.stream(generator)
      result = Enum.to_list(stream)

      assert result == [1, 2, 3]
    end

    test "handles immediate done" do
      generator = fn -> :done end
      stream = Prefetcher.stream(generator)
      result = Enum.to_list(stream)

      assert result == []
    end
  end

  describe "reduce_stream_indexed/4" do
    test "processes lazy stream with indices" do
      # Create a lazy stream (not materialized)
      stream = Stream.map(1..5, & &1)

      result =
        Prefetcher.reduce_stream_indexed(stream, [], fn batch, idx, acc ->
          [{batch, idx} | acc]
        end)

      assert Enum.reverse(result) == [{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}]
    end

    test "handles empty stream" do
      stream = Stream.map([], & &1)

      result =
        Prefetcher.reduce_stream_indexed(stream, :initial, fn _, _, _ ->
          :never_called
        end)

      assert result == :initial
    end

    test "respects buffer_size option" do
      stream = Stream.map(1..10, & &1)

      result =
        Prefetcher.reduce_stream_indexed(
          stream,
          0,
          fn batch, _idx, acc -> acc + batch end,
          buffer_size: 3
        )

      # 1 + 2 + ... + 10 = 55
      assert result == 55
    end

    test "maintains accumulator correctly" do
      stream = Stream.map([10, 20, 30], & &1)

      result =
        Prefetcher.reduce_stream_indexed(stream, 0, fn batch, idx, acc ->
          acc + batch + idx
        end)

      # 0 + (10 + 0) = 10
      # 10 + (20 + 1) = 31
      # 31 + (30 + 2) = 63
      assert result == 63
    end

    test "processes all batches" do
      stream = Stream.map([:a, :b, :c, :d], & &1)

      result =
        Prefetcher.reduce_stream_indexed(stream, [], fn batch, _idx, acc ->
          [batch | acc]
        end)

      # All batches should be processed (order may vary due to async prefetching)
      assert Enum.sort(result) == [:a, :b, :c, :d]
    end

    test "works with Stream.chunk_every" do
      # Simulate batched data
      data = 1..12
      batch_size = 4
      stream = Stream.chunk_every(data, batch_size)

      {sum, count} =
        Prefetcher.reduce_stream_indexed(stream, {0, 0}, fn batch, _idx, {sum, count} ->
          {sum + Enum.sum(batch), count + 1}
        end)

      # 1 + 2 + ... + 12
      assert sum == 78
      # 3 batches of 4
      assert count == 3
    end
  end

  describe "Nx tensor compatibility (regression test for EXLA deadlock bug)" do
    # Regression test for gotcha #32: Prefetcher deadlock with EXLA tensors in spawned processes
    # The bug: reduce_stream_indexed spawns a process that calls Stream.run(), and Nx.stack
    # operations inside that spawned process can deadlock with EXLA tensors.
    # The fix: use reduce_indexed (materializes in main process) for non-streaming mode.

    test "reduce_indexed works with Nx tensor batches" do
      # Simulate Data.batched_sequences: a stream that creates Nx tensors via Nx.stack
      # This is what happens with precomputed embeddings
      embeddings = [
        Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        Nx.tensor([[5.0, 6.0], [7.0, 8.0]]),
        Nx.tensor([[9.0, 10.0], [11.0, 12.0]])
      ]

      # Stream.map with Nx.stack simulates batch creation
      batch_stream =
        Enum.chunk_every(embeddings, 2)
        |> Stream.map(fn chunk ->
          %{states: Nx.stack(chunk)}
        end)

      # reduce_indexed should work (materializes to list in main process first)
      {results, count} =
        Prefetcher.reduce_indexed(batch_stream, {[], 0}, fn batch, idx, {acc, cnt} ->
          # Verify we can access the tensor
          shape = Nx.shape(batch.states)
          {[{idx, shape} | acc], cnt + 1}
        end)

      assert count == 2
      assert Enum.sort(results) == [{0, {2, 2, 2}}, {1, {1, 2, 2}}]
    end

    test "reduce_indexed processes all Nx batches without deadlock" do
      # Create a larger batch stream to stress test
      num_batches = 20
      batch_size = 4
      embed_size = 8

      # Pre-create tensors (simulating precomputed embeddings)
      key = Nx.Random.key(42)

      {tensors, _} =
        Enum.map_reduce(1..num_batches, key, fn _, k ->
          Nx.Random.uniform(k, shape: {batch_size, embed_size})
        end)

      batch_stream = Stream.map(tensors, fn t -> %{states: t, label: Nx.sum(t)} end)

      # This should complete without deadlock
      {processed_count, total_sum} =
        Prefetcher.reduce_indexed(
          batch_stream,
          {0, 0.0},
          fn batch, _idx, {count, sum} ->
            batch_sum = Nx.to_number(batch.label)
            {count + 1, sum + batch_sum}
          end
        )

      assert processed_count == num_batches
      # Sum should be positive (random values are positive)
      assert total_sum > 0
    end

    test "reduce_stream_indexed works with simple Nx tensors" do
      # Stream-based prefetcher should still work with Nx tensors
      # (it works with BinaryBackend, the issue was specifically with EXLA in spawned processes)
      stream =
        Stream.map(1..5, fn i ->
          %{tensor: Nx.tensor([i, i * 2, i * 3])}
        end)

      result =
        Prefetcher.reduce_stream_indexed(stream, [], fn batch, idx, acc ->
          sum = Nx.to_number(Nx.sum(batch.tensor))
          [{idx, sum} | acc]
        end)

      # i + 2i + 3i = 6i
      expected = [{0, 6}, {1, 12}, {2, 18}, {3, 24}, {4, 30}]
      assert Enum.sort(result) == Enum.sort(expected)
    end
  end

  describe "lazy iteration regression (gotcha #32)" do
    # Regression test for the fix that changed reduce_indexed from materializing
    # all batches upfront (Enum.to_list) to lazy iteration (Stream.with_index).
    # The old behavior caused 25+ minute delays when training on large datasets
    # because Nx.stack was called on ALL batches before training started.

    test "reduce_indexed iterates lazily, not materializing all batches upfront" do
      # Track when each batch is accessed
      access_log = Agent.start_link(fn -> [] end) |> elem(1)

      # Create a stream that logs when each element is accessed
      stream =
        Stream.map(1..5, fn i ->
          Agent.update(access_log, fn log -> [{:accessed, i} | log] end)
          i
        end)

      # Process with reduce_indexed, logging when each batch is processed
      _result =
        Prefetcher.reduce_indexed(stream, [], fn batch, _idx, acc ->
          Agent.update(access_log, fn log -> [{:processed, batch} | log] end)
          [batch | acc]
        end)

      log = Agent.get(access_log, & &1) |> Enum.reverse()

      # Key assertion: batches should be accessed and processed in an interleaved pattern
      # NOT all accessed first, then all processed (which would indicate upfront materialization)
      #
      # Lazy iteration pattern: [:accessed 1, :processed 1, :accessed 2, :processed 2, ...]
      # Upfront materialization: [:accessed 1, :accessed 2, ..., :accessed 5, :processed 1, ...]

      # Check that the first processed event comes early (lazy), not after all accesses (upfront)
      first_processed_idx = Enum.find_index(log, fn {type, _} -> type == :processed end)

      # If lazy: first_processed_idx should be 1 (after first access)
      # If upfront: first_processed_idx would be 5 (after all accesses)
      assert first_processed_idx < 5,
             "reduce_indexed should not materialize all batches before processing"

      # Verify interleaved pattern: each access should be followed by its processing
      assert log == [
               {:accessed, 1},
               {:processed, 1},
               {:accessed, 2},
               {:processed, 2},
               {:accessed, 3},
               {:processed, 3},
               {:accessed, 4},
               {:processed, 4},
               {:accessed, 5},
               {:processed, 5}
             ]
    end

    test "reduce_indexed with Nx.stack batches doesn't cause startup delay" do
      # Simulates the actual training scenario: stream of batches where each
      # batch creation involves Nx.stack (which was slow when done 3567 times upfront)

      batch_count = 10
      batch_creation_count = Agent.start_link(fn -> 0 end) |> elem(1)

      # Stream that creates Nx tensors (simulating batch creation with Nx.stack)
      batch_stream =
        Stream.map(1..batch_count, fn i ->
          Agent.update(batch_creation_count, &(&1 + 1))
          # Simulate batch creation with Nx.stack
          Nx.stack([Nx.tensor([i, i * 2]), Nx.tensor([i * 3, i * 4])])
        end)

      # Process batches
      {results, _} =
        Prefetcher.reduce_indexed(batch_stream, {[], 0}, fn batch, idx, {acc, count} ->
          sum = Nx.to_number(Nx.sum(batch))
          {[{idx, sum} | acc], count + 1}
        end)

      # Verify all batches were processed
      assert length(results) == batch_count

      # Verify batch creation happened (sanity check)
      assert Agent.get(batch_creation_count, & &1) == batch_count
    end
  end
end
