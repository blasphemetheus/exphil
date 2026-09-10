defmodule ExPhil.Training.EmbeddingCacheBudgetTest do
  @moduledoc """
  INVARIANTS.md item 10: the cache refuses the write that would exceed
  its budget. Pins the structural fix for the 2026-09-05 disk fill.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.EmbeddingCache

  setup do
    dir = Path.join(System.tmp_dir!(), "exphil_cache_budget_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, dir: dir}
  end

  test "refuses a write that would exceed the budget", %{dir: dir} do
    tensor = Nx.iota({100, 8}, type: :f32)
    assert {:error, :over_budget} = EmbeddingCache.save("k1", tensor, cache_dir: dir, budget_bytes: 10)
    assert EmbeddingCache.dir_bytes(dir) == 0
  end

  test "accepts a write within budget, then refuses once the dir is full", %{dir: dir} do
    tensor = Nx.iota({100, 8}, type: :f32)
    bytes = Nx.byte_size(tensor)

    assert :ok = EmbeddingCache.save("k1", tensor, cache_dir: dir, budget_bytes: bytes * 2)
    assert EmbeddingCache.dir_bytes(dir) > 0

    # The second identical write would push held + incoming past 2x the
    # tensor (compressed files are smaller than the tensor, so allow the
    # budget to be just under held + incoming).
    held = EmbeddingCache.dir_bytes(dir)
    assert {:error, :over_budget} = EmbeddingCache.save("k2", tensor, cache_dir: dir, budget_bytes: held + bytes - 1)
  end

  test "budget resolves opt > env > default" do
    assert EmbeddingCache.budget_bytes(budget_bytes: 123) == 123
    assert EmbeddingCache.budget_bytes([]) == 50 * 1024 * 1024 * 1024
  end
end
