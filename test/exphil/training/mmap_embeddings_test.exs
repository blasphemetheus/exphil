defmodule ExPhil.Training.MmapEmbeddingsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.MmapEmbeddings

  @test_dir "test/tmp/mmap_embeddings"

  setup do
    # Create test directory
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      # Clean up test files
      File.rm_rf!(@test_dir)
    end)

    :ok
  end

  describe "save/3 and open/1" do
    test "saves and loads embeddings correctly" do
      path = Path.join(@test_dir, "test_embeddings.bin")

      # Create test embeddings
      embeddings = Nx.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ], type: :f32)

      # Save
      assert :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)
      assert File.exists?(path)

      # Open
      assert {:ok, handle} = MmapEmbeddings.open(path)
      assert handle.num_frames == 3
      assert handle.embed_size == 3
      assert handle.dtype == :f32

      MmapEmbeddings.close(handle)
    end

    test "returns error for non-existent file" do
      assert {:error, {:file_open, :enoent}} = MmapEmbeddings.open("nonexistent.bin")
    end

    test "returns error for invalid shape" do
      path = Path.join(@test_dir, "invalid.bin")
      # 1D tensor is invalid
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      assert {:error, {:invalid_shape, {3}, _}} = MmapEmbeddings.save(tensor, path)
    end
  end

  describe "read_batch/2" do
    test "reads correct embeddings by indices" do
      path = Path.join(@test_dir, "batch_test.bin")

      embeddings = Nx.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
      ], type: :f32)

      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)
      {:ok, handle} = MmapEmbeddings.open(path)

      # Read batch [0, 2] - should get rows 0 and 2
      batch = MmapEmbeddings.read_batch(handle, [0, 2])

      assert Nx.shape(batch) == {2, 2}
      assert Nx.to_flat_list(batch) == [1.0, 2.0, 5.0, 6.0]

      # Read batch [3, 1] - out of order
      batch2 = MmapEmbeddings.read_batch(handle, [3, 1])
      assert Nx.to_flat_list(batch2) == [7.0, 8.0, 3.0, 4.0]

      MmapEmbeddings.close(handle)
    end

    test "reads single frame correctly" do
      path = Path.join(@test_dir, "single_test.bin")

      embeddings = Nx.tensor([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0]
      ], type: :f32)

      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)
      {:ok, handle} = MmapEmbeddings.open(path)

      frame = MmapEmbeddings.read_frame(handle, 1)
      assert Nx.shape(frame) == {3}
      assert Nx.to_flat_list(frame) == [40.0, 50.0, 60.0]

      MmapEmbeddings.close(handle)
    end
  end

  describe "info/1" do
    test "returns correct metadata" do
      path = Path.join(@test_dir, "info_test.bin")

      embeddings = Nx.iota({100, 64}, type: :f32)
      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)
      {:ok, handle} = MmapEmbeddings.open(path)

      info = MmapEmbeddings.info(handle)

      assert info.num_frames == 100
      assert info.embed_size == 64
      assert info.dtype == :f32
      assert info.path == path
      # 100 * 64 * 4 bytes + 32 header = 25632 bytes = ~0.025 MB
      assert_in_delta info.size_mb, 0.025, 0.001

      MmapEmbeddings.close(handle)
    end
  end

  describe "exists?/1" do
    test "returns true for valid file" do
      path = Path.join(@test_dir, "exists_test.bin")
      embeddings = Nx.iota({10, 5}, type: :f32)
      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)

      assert MmapEmbeddings.exists?(path)
    end

    test "returns false for non-existent file" do
      refute MmapEmbeddings.exists?("nonexistent.bin")
    end

    test "returns false for invalid file" do
      path = Path.join(@test_dir, "invalid.bin")
      File.write!(path, "not a valid embeddings file")

      refute MmapEmbeddings.exists?(path)
    end
  end

  describe "dtype handling" do
    test "preserves f32 type" do
      path = Path.join(@test_dir, "f32_test.bin")
      embeddings = Nx.tensor([[1.0, 2.0]], type: :f32)

      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)
      {:ok, handle} = MmapEmbeddings.open(path)

      assert handle.dtype == :f32
      frame = MmapEmbeddings.read_frame(handle, 0)
      assert Nx.type(frame) == {:f, 32}

      MmapEmbeddings.close(handle)
    end
  end

  describe "large dataset simulation" do
    test "handles many frames efficiently" do
      path = Path.join(@test_dir, "large_test.bin")

      # Simulate a larger dataset (1000 frames, 128 dims)
      embeddings = Nx.iota({1000, 128}, type: :f32)
      :ok = MmapEmbeddings.save(embeddings, path, show_progress: false)

      {:ok, handle} = MmapEmbeddings.open(path)

      # Read random batch
      batch_indices = Enum.take_random(0..999, 32)
      batch = MmapEmbeddings.read_batch(handle, batch_indices)

      assert Nx.shape(batch) == {32, 128}

      MmapEmbeddings.close(handle)
    end
  end
end
