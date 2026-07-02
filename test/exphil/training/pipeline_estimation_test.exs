defmodule ExPhil.Training.PipelineEstimationTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Pipeline

  describe "estimated batch count" do
    # Test the math behind batch estimation without needing actual data
    # The Pipeline uses: div(num_sequences, batch_size)
    # where num_sequences = div(num_frames - window_size, stride) + 1

    test "basic sequence count calculation" do
      # 1000 frames, window 60, stride 5
      num_frames = 1000
      window = 60
      stride = 5
      num_sequences = div(num_frames - window, stride) + 1
      assert num_sequences == 189
    end

    test "batch count from sequence count" do
      num_sequences = 189
      batch_size = 32
      batches = div(num_sequences, batch_size)
      assert batches == 5
    end

    test "large dataset estimation" do
      # 200 files, ~7000 frames each, 90% train split
      num_frames = 200 * 7000 * 0.9 |> trunc()
      window = 60
      stride = 5
      batch_size = 32

      num_sequences = div(num_frames - window, stride) + 1
      batches = div(num_sequences, batch_size)

      # Should be roughly 7880 (matching what we saw in real runs)
      assert batches > 7000
      assert batches < 9000
    end

    test "stride 1 vs stride 5 comparison" do
      num_frames = 10000
      window = 60

      seqs_s1 = div(num_frames - window, 1) + 1
      seqs_s5 = div(num_frames - window, 5) + 1

      # Stride 5 should be ~5x fewer sequences
      ratio = seqs_s1 / seqs_s5
      assert_in_delta ratio, 5.0, 0.1
    end
  end

  describe "Pipeline struct" do
    test "has_validation? returns false for nil val_batches" do
      pipeline = %Pipeline{val_batches: nil}
      refute Pipeline.has_validation?(pipeline)
    end

    test "has_validation? returns false for empty val_batches" do
      pipeline = %Pipeline{val_batches: []}
      refute Pipeline.has_validation?(pipeline)
    end

    test "has_validation? returns true with val_batches" do
      pipeline = %Pipeline{val_batches: [%{states: :mock}]}
      assert Pipeline.has_validation?(pipeline)
    end

    test "val_batch_count returns 0 for nil" do
      assert Pipeline.val_batch_count(%Pipeline{val_batches: nil}) == 0
    end

    test "val_batch_count returns length" do
      assert Pipeline.val_batch_count(%Pipeline{val_batches: [1, 2, 3]}) == 3
    end
  end
end
