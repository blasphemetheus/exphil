defmodule ExPhil.Data.SituationIndexTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.SituationIndex

  # Query-side unit tests on hand-built shards (build/3 needs real .slp —
  # covered by the live smoke in the flywheel test plan).

  @dim 8

  defp write_shard!(dir, seq, source, vectors, frames) do
    n = length(vectors)

    emb_bin =
      vectors
      |> Nx.tensor(type: :f32)
      |> Nx.as_type(:f16)
      |> Nx.to_binary()

    file = "shard_#{String.pad_leading(to_string(seq), 5, "0")}.bin"

    File.write!(
      Path.join(dir, file),
      :erlang.term_to_binary(%{emb_bin: emb_bin, n: n, dim: @dim, frames: frames})
    )

    %{"file" => file, "source_slp" => source, "port" => 1, "frames" => n}
  end

  defp with_index(shard_specs, fun) do
    dir =
      Path.join(
        System.tmp_dir!(),
        "sit_idx_test_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(dir)

    shards =
      shard_specs
      |> Enum.with_index()
      |> Enum.map(fn {{source, vectors, frames}, seq} ->
        write_shard!(dir, seq, source, vectors, frames)
      end)

    manifest = %{"dim" => @dim, "dtype" => "f16", "shards" => shards}
    File.write!(Path.join(dir, "manifest.json"), Jason.encode!(manifest))

    try do
      fun.(dir)
    after
      File.rm_rf!(dir)
    end
  end

  defp unit(i), do: List.duplicate(0.0, i) ++ [1.0] ++ List.duplicate(0.0, @dim - i - 1)

  describe "query/3" do
    test "returns the exact match first with score ~1.0" do
      with_index(
        [{"a.slp", [unit(0), unit(1), unit(2)], [300, 600, 900]}],
        fn dir ->
          q = Nx.tensor(unit(1), type: :f32)
          [best | _] = SituationIndex.query(dir, q, k: 2)
          assert best.slp == "a.slp"
          assert best.frame == 600
          assert_in_delta best.score, 1.0, 0.01
        end
      )
    end

    test "merges top-k across shards, best-first" do
      # b.slp holds the closest vector to the query direction.
      close = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      with_index(
        [
          {"a.slp", [unit(1), unit(2)], [300, 600]},
          {"b.slp", [close, unit(3)], [1200, 1500]}
        ],
        fn dir ->
          q = Nx.tensor(unit(0), type: :f32)
          [best, second | _] = SituationIndex.query(dir, q, k: 3)
          assert best.slp == "b.slp"
          assert best.frame == 1200
          assert best.score > second.score
        end
      )
    end

    test "exclude_slp drops self-retrieval" do
      with_index(
        [
          {"self.slp", [unit(0)], [300]},
          {"other.slp", [unit(0)], [900]}
        ],
        fn dir ->
          q = Nx.tensor(unit(0), type: :f32)
          results = SituationIndex.query(dir, q, k: 5, exclude_slp: "self.slp")
          assert Enum.all?(results, &(&1.slp == "other.slp"))
        end
      )
    end

    test "temporal spacing keeps only the best of a consecutive run" do
      # Five near-identical consecutive frames of one held situation, plus a
      # distinct later moment in the same replay.
      held = for _ <- 1..5, do: unit(0)

      with_index(
        [{"a.slp", held ++ [unit(0)], [1000, 1001, 1002, 1003, 1004, 2000]}],
        fn dir ->
          q = Nx.tensor(unit(0), type: :f32)
          results = SituationIndex.query(dir, q, k: 5, spacing: 240)
          frames = Enum.map(results, & &1.frame)
          # One survivor from the 1000-1004 run, plus 2000.
          assert length(frames) == 2
          assert 2000 in frames
          assert Enum.count(frames, &(&1 < 1240)) == 1
        end
      )
    end

    test "spacing 0 disables the filter" do
      held = for _ <- 1..3, do: unit(0)

      with_index([{"a.slp", held, [1000, 1001, 1002]}], fn dir ->
        q = Nx.tensor(unit(0), type: :f32)
        assert length(SituationIndex.query(dir, q, k: 5, spacing: 0)) == 3
      end)
    end

    test "dim mismatch raises" do
      with_index([{"a.slp", [unit(0)], [300]}], fn dir ->
        assert_raise ArgumentError, ~r/query dim/, fn ->
          SituationIndex.query(dir, Nx.tensor([1.0, 0.0]), k: 1)
        end
      end)
    end
  end

  describe "manifest/stats" do
    test "empty dir yields an empty manifest and stats" do
      dir = Path.join(System.tmp_dir!(), "sit_idx_empty_#{System.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      assert SituationIndex.stats(dir) == %{files: 0, frames: 0, dim: nil}
      File.rm_rf!(dir)
    end
  end
end
