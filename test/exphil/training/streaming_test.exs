defmodule ExPhil.Training.StreamingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Streaming

  describe "chunk_files/2" do
    test "splits files into chunks of specified size" do
      files = ["a.slp", "b.slp", "c.slp", "d.slp", "e.slp"]

      assert Streaming.chunk_files(files, 2) == [
               ["a.slp", "b.slp"],
               ["c.slp", "d.slp"],
               ["e.slp"]
             ]
    end

    test "handles exact divisor" do
      files = ["a.slp", "b.slp", "c.slp", "d.slp"]

      assert Streaming.chunk_files(files, 2) == [
               ["a.slp", "b.slp"],
               ["c.slp", "d.slp"]
             ]
    end

    test "handles single file" do
      assert Streaming.chunk_files(["a.slp"], 5) == [["a.slp"]]
    end

    test "handles empty list" do
      assert Streaming.chunk_files([], 5) == []
    end

    test "handles {path, port} tuples" do
      files = [{"a.slp", 1}, {"b.slp", 2}, {"c.slp", 1}]

      assert Streaming.chunk_files(files, 2) == [
               [{"a.slp", 1}, {"b.slp", 2}],
               [{"c.slp", 1}]
             ]
    end

    test "chunk size of 1 returns individual files" do
      files = ["a.slp", "b.slp", "c.slp"]

      assert Streaming.chunk_files(files, 1) == [
               ["a.slp"],
               ["b.slp"],
               ["c.slp"]
             ]
    end
  end

  describe "format_config/2" do
    test "formats streaming configuration" do
      result = Streaming.format_config(30, 100)
      assert result == "4 chunks of 30 files (100 total)"
    end

    test "handles exact divisor" do
      result = Streaming.format_config(25, 100)
      assert result == "4 chunks of 25 files (100 total)"
    end

    test "handles small datasets" do
      result = Streaming.format_config(50, 20)
      assert result == "1 chunks of 50 files (20 total)"
    end
  end
end
