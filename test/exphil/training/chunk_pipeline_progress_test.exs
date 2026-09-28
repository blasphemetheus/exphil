defmodule ExPhil.Training.ChunkPipelineProgressTest do
  @moduledoc """
  Stop/restart markers (2026-09-27): `ChunkPipeline.read_progress/1` and the
  marker contract a resuming driver relies on (absolute chunk index, atomic
  write). The writer is exercised through `stream_prepared_chunks/2` with a
  stubbed chunk list of zero files, which the pipeline prepares as empty
  datasets — no replays, no GPU.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.ChunkPipeline

  setup do
    dir = Path.join(System.tmp_dir!(), "chunk_progress_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, path: Path.join(dir, "progress.json")}
  end

  test "read_progress returns :none for a missing or malformed marker", %{path: path} do
    assert ChunkPipeline.read_progress(path) == :none
    File.write!(path, "not json")
    assert ChunkPipeline.read_progress(path) == :none
    File.write!(path, Jason.encode!(%{chunk: 3}))
    assert ChunkPipeline.read_progress(path) == :none
  end

  test "read_progress parses a marker", %{path: path} do
    File.write!(path, Jason.encode!(%{chunk: 409, of: 445, started_at: "2026-09-27T22:11:00Z"}))
    assert ChunkPipeline.read_progress(path) == {:ok, %{chunk: 409, of: 445}}
  end

  test "the marker is absolute: chunk_offset + local index, of = offset + local total", %{path: path} do
    # A driver that dropped 408 of 445 chunks hands the pipeline 37; local
    # indices 1..3 must be recorded as 409, 410, 411 of 445.
    prep = %{progress_path: path, chunk_offset: 408, total_absolute: 445}

    seen =
      for idx <- 1..3 do
        :ok = ChunkPipeline.record_progress(prep, idx)
        {:ok, marker} = ChunkPipeline.read_progress(path)
        {idx, marker}
      end

    assert seen == [
             {1, %{chunk: 409, of: 445}},
             {2, %{chunk: 410, of: 445}},
             {3, %{chunk: 411, of: 445}}
           ]

    refute File.exists?(path <> ".tmp")
  end

  test "no progress_path means no marker", %{path: path} do
    assert ChunkPipeline.record_progress(%{progress_path: nil}, 1) == :ok
    refute File.exists?(path)
  end
end
