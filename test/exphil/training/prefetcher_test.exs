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
      result = Prefetcher.reduce([1, 2, 3], 0, fn batch, acc ->
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
      result = Prefetcher.reduce([1, 2, 3, 4], [], fn batch, acc ->
        [batch | acc]
      end)

      # Reversed because we prepend
      assert Enum.reverse(result) == [1, 2, 3, 4]
    end
  end

  describe "reduce_indexed/4" do
    test "provides correct indices" do
      result = Prefetcher.reduce_indexed([:a, :b, :c], [], fn batch, idx, acc ->
        [{batch, idx} | acc]
      end)

      assert Enum.reverse(result) == [{:a, 0}, {:b, 1}, {:c, 2}]
    end

    test "handles empty list" do
      result = Prefetcher.reduce_indexed([], :initial, fn _, _, _ -> :never_called end)
      assert result == :initial
    end

    test "handles single element" do
      result = Prefetcher.reduce_indexed([42], [], fn batch, idx, acc ->
        [{batch, idx} | acc]
      end)

      assert result == [{42, 0}]
    end

    test "accumulator works correctly" do
      result = Prefetcher.reduce_indexed([10, 20, 30], 0, fn batch, idx, acc ->
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
end
