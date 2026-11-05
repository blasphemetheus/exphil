Code.require_file("../../../scripts/lib/learning_data_cache.exs", __DIR__)

defmodule ExPhil.LearningDataCacheTest do
  use ExUnit.Case, async: false
  alias ExPhil.LearningDataCache, as: Cache

  setup do
    path = Path.join(System.tmp_dir!(), "learning_cache_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(path) end)
    %{path: path}
  end

  test "reuses only the same data signature and keeps nested tensors on host", %{path: path} do
    value = %{
      batches: [
        {Nx.tensor([1, 2], backend: Nx.BinaryBackend),
         %{label: Nx.tensor(3, backend: Nx.BinaryBackend)}}
      ]
    }

    assert Cache.fetch(path, :first, fn -> value end) == value
    assert Cache.fetch(path, :first, fn -> flunk("should reuse prepared data") end) == value
    assert Cache.fetch(path, :second, fn -> :different_data end) == :different_data
    assert length(Path.wildcard(Path.join(path, "*.term"))) == 2
  end

  @tag :gpu
  test "copies device-backed validation labels before writing", %{path: path} do
    tensor = Nx.tensor([1.0, 2.0], backend: EXLA.Backend)

    %{labels: host} =
      Cache.fetch(path, :device, fn ->
        %{labels: tensor, dataset: %ExPhil.Training.Data{embedded_frames: tensor}}
      end)

    assert %Nx.BinaryBackend{} = host.data
    %{labels: restored} = Cache.fetch(path, :device, fn -> flunk("cache miss") end)
    assert Nx.to_flat_list(restored) == [1.0, 2.0]
  end

  test "rejects process-local data outside tensors", %{path: path} do
    assert_raise ArgumentError, fn ->
      Cache.fetch(path, :bad, fn -> %{reference: make_ref()} end)
    end

    assert Path.wildcard(Path.join(path, "*.term")) == []
  end
end
