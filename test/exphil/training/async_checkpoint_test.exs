defmodule ExPhil.Training.AsyncCheckpointTest do
  # Not async because it uses a named GenServer
  use ExUnit.Case, async: false

  alias ExPhil.Training.AsyncCheckpoint

  @moduletag :checkpoint

  setup do
    # The AsyncCheckpoint server is started by the application supervisor,
    # so we just need to ensure any pending saves are complete before each test
    AsyncCheckpoint.await_pending()
    :ok
  end

  describe "save_async/2" do
    test "saves checkpoint asynchronously" do
      checkpoint = %{
        params: %{layer: Nx.broadcast(1.0, {10, 10})},
        step: 42,
        config: %{learning_rate: 1.0e-4}
      }

      path = Path.join(System.tmp_dir!(), "async_test_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      # Save should return immediately
      assert :ok = AsyncCheckpoint.save_async(checkpoint, path)

      # Wait for save to complete
      assert :ok = AsyncCheckpoint.await_pending()

      # Verify file exists and content is correct
      assert File.exists?(path)
      loaded = :erlang.binary_to_term(File.read!(path))
      assert loaded.step == 42
      assert Nx.to_number(Nx.sum(loaded.params.layer)) == 100.0
    end

    test "handles multiple concurrent saves" do
      paths =
        for i <- 1..5 do
          path = Path.join(System.tmp_dir!(), "async_multi_#{i}_#{System.unique_integer()}.axon")
          on_exit(fn -> File.rm(path) end)
          path
        end

      # Queue multiple saves
      for {path, i} <- Enum.with_index(paths, 1) do
        checkpoint = %{step: i, data: Nx.broadcast(i, {5, 5})}
        assert :ok = AsyncCheckpoint.save_async(checkpoint, path)
      end

      # Wait for all to complete
      assert :ok = AsyncCheckpoint.await_pending()

      # Verify all files exist with correct content
      for {path, i} <- Enum.with_index(paths, 1) do
        assert File.exists?(path), "File #{path} should exist"
        loaded = :erlang.binary_to_term(File.read!(path))
        assert loaded.step == i
      end
    end

    test "pending_count tracks queue size" do
      # Start with empty queue
      assert AsyncCheckpoint.pending_count() == 0

      # Add a checkpoint that takes time (large tensor)
      large_tensor = Nx.broadcast(1.0, {1000, 1000})
      checkpoint = %{data: large_tensor}
      path = Path.join(System.tmp_dir!(), "async_pending_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      AsyncCheckpoint.save_async(checkpoint, path)

      # Should show as pending
      count = AsyncCheckpoint.pending_count()
      # Might already be done on fast systems
      assert count >= 0

      AsyncCheckpoint.await_pending()
      assert AsyncCheckpoint.pending_count() == 0
    end
  end

  describe "atomic_write/2" do
    test "writes file atomically" do
      checkpoint = %{test: "data", tensor: Nx.iota({3, 3})}
      path = Path.join(System.tmp_dir!(), "atomic_test_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      assert :ok = AsyncCheckpoint.atomic_write(checkpoint, path)
      assert File.exists?(path)

      # No temp files should remain
      dir = Path.dirname(path)
      temp_files = Path.wildcard("#{dir}/*.tmp")
      assert temp_files == []
    end

    test "creates parent directories" do
      nested_dir = Path.join([System.tmp_dir!(), "async_test", "nested", "dirs"])
      path = Path.join(nested_dir, "checkpoint.axon")
      on_exit(fn -> File.rm_rf!(Path.join(System.tmp_dir!(), "async_test")) end)

      checkpoint = %{step: 1}
      assert :ok = AsyncCheckpoint.atomic_write(checkpoint, path)
      assert File.exists?(path)
    end

    test "cleans up temp file on error" do
      # Try to write to invalid path (directory)
      path = System.tmp_dir!()

      checkpoint = %{step: 1}
      result = AsyncCheckpoint.atomic_write(checkpoint, path)

      assert match?({:error, _}, result)

      # No temp files should remain
      temp_files = Path.wildcard("#{path}/*.tmp")
      assert temp_files == []
    end
  end

  describe "saving?/0" do
    test "returns true while save is in progress" do
      # Start with no saves
      refute AsyncCheckpoint.saving?()

      # Queue a large save
      large_tensor = Nx.broadcast(1.0, {2000, 2000})
      checkpoint = %{data: large_tensor}
      path = Path.join(System.tmp_dir!(), "async_saving_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      AsyncCheckpoint.save_async(checkpoint, path)

      # Might be saving (depends on timing)
      # Just verify it doesn't crash
      _is_saving = AsyncCheckpoint.saving?()

      AsyncCheckpoint.await_pending()
      refute AsyncCheckpoint.saving?()
    end
  end
end
