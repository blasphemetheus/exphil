defmodule ExPhil.Training.DuplicateDetectorTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.DuplicateDetector

  @tmp_dir "test/tmp/duplicate_detector"

  setup do
    # Create temp directory
    File.rm_rf!(@tmp_dir)
    File.mkdir_p!(@tmp_dir)

    on_exit(fn ->
      File.rm_rf!(@tmp_dir)
    end)

    :ok
  end

  describe "hash_file/1" do
    test "computes consistent hash for same content" do
      path = Path.join(@tmp_dir, "test.txt")
      File.write!(path, "hello world")

      {:ok, hash1} = DuplicateDetector.hash_file(path)
      {:ok, hash2} = DuplicateDetector.hash_file(path)

      assert hash1 == hash2
      assert is_binary(hash1)
      # MD5 is 16 bytes
      assert byte_size(hash1) == 16
    end

    test "computes different hashes for different content" do
      path1 = Path.join(@tmp_dir, "file1.txt")
      path2 = Path.join(@tmp_dir, "file2.txt")
      File.write!(path1, "hello")
      File.write!(path2, "world")

      {:ok, hash1} = DuplicateDetector.hash_file(path1)
      {:ok, hash2} = DuplicateDetector.hash_file(path2)

      assert hash1 != hash2
    end

    test "returns error for non-existent file" do
      result = DuplicateDetector.hash_file("/nonexistent/file.txt")
      assert {:error, _} = result
    end
  end

  describe "filter_duplicates/2" do
    test "removes duplicate files" do
      # Create files with same content
      path1 = Path.join(@tmp_dir, "original.txt")
      path2 = Path.join(@tmp_dir, "copy.txt")
      path3 = Path.join(@tmp_dir, "another_copy.txt")
      content = "duplicate content"

      File.write!(path1, content)
      File.write!(path2, content)
      File.write!(path3, content)

      {unique, stats} = DuplicateDetector.filter_duplicates([path1, path2, path3])

      assert length(unique) == 1
      assert stats.total == 3
      assert stats.unique == 1
      assert stats.duplicates == 2
      assert map_size(stats.duplicate_groups) == 1
    end

    test "keeps all unique files" do
      path1 = Path.join(@tmp_dir, "file1.txt")
      path2 = Path.join(@tmp_dir, "file2.txt")
      path3 = Path.join(@tmp_dir, "file3.txt")

      File.write!(path1, "content 1")
      File.write!(path2, "content 2")
      File.write!(path3, "content 3")

      {unique, stats} = DuplicateDetector.filter_duplicates([path1, path2, path3])

      assert length(unique) == 3
      assert stats.duplicates == 0
      assert stats.duplicate_groups == %{}
    end

    test "handles empty list" do
      {unique, stats} = DuplicateDetector.filter_duplicates([])

      assert unique == []
      assert stats.total == 0
      assert stats.unique == 0
    end

    test "handles mixed duplicates and unique" do
      # 2 unique, 2 duplicates of another
      path1 = Path.join(@tmp_dir, "unique1.txt")
      path2 = Path.join(@tmp_dir, "unique2.txt")
      path3 = Path.join(@tmp_dir, "dup1.txt")
      path4 = Path.join(@tmp_dir, "dup2.txt")

      File.write!(path1, "unique 1")
      File.write!(path2, "unique 2")
      File.write!(path3, "duplicate")
      File.write!(path4, "duplicate")

      {unique, stats} = DuplicateDetector.filter_duplicates([path1, path2, path3, path4])

      assert length(unique) == 3
      assert stats.duplicates == 1
    end
  end

  describe "filter_duplicates_stream/1" do
    test "streams unique files" do
      path1 = Path.join(@tmp_dir, "a.txt")
      path2 = Path.join(@tmp_dir, "b.txt")
      path3 = Path.join(@tmp_dir, "c.txt")

      File.write!(path1, "content")
      # duplicate
      File.write!(path2, "content")
      File.write!(path3, "different")

      unique =
        [path1, path2, path3]
        |> DuplicateDetector.filter_duplicates_stream()
        |> Enum.to_list()

      assert length(unique) == 2
      # One of the duplicates
      assert path1 in unique or path2 in unique
      assert path3 in unique
    end
  end

  describe "print_summary/1" do
    import ExUnit.CaptureIO

    test "prints summary for duplicates found" do
      stats = %{
        total: 10,
        unique: 7,
        duplicates: 3,
        duplicate_groups: %{
          "hash1" => ["file1.txt", "file2.txt"],
          "hash2" => ["file3.txt", "file4.txt", "file5.txt"]
        }
      }

      output =
        capture_io(:stderr, fn ->
          DuplicateDetector.print_summary(stats)
        end)

      assert output =~ "removed 3/10"
      assert output =~ "30.0%"
    end

    test "prints summary for no duplicates" do
      stats = %{
        total: 5,
        unique: 5,
        duplicates: 0,
        duplicate_groups: %{}
      }

      output =
        capture_io(:stderr, fn ->
          DuplicateDetector.print_summary(stats)
        end)

      assert output =~ "no duplicates"
    end
  end
end
