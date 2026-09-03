defmodule ExPhil.Training.TrajectoryCursorsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.{Data, TrajectoryCursors}

  # Synthetic frame: the Slippi frame counter drives segmentation; the
  # :action key short-circuits Data.frame_action (no controller needed).
  defp frame(counter, action_id) do
    %{
      game_state: %{frame: counter},
      action: action(action_id)
    }
  end

  defp action(id) do
    %{
      buttons: %{a: rem(id, 2) == 1, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
      main_x: rem(id, 17),
      main_y: 8,
      c_x: 8,
      c_y: 8,
      shoulder: 0
    }
  end

  # games: list of lengths; frame counters restart at 0 per game.
  # Embedding row i = [i, i] so contiguity is checkable from values.
  defp dataset(game_lengths) do
    frames =
      game_lengths
      |> Enum.flat_map(fn len -> Enum.map(0..(len - 1), &frame(&1, &1)) end)

    n = length(frames)
    embedded = Nx.iota({n, 2}, axis: 0) |> Nx.as_type(:f32)

    %Data{frames: frames, embedded_frames: embedded, size: n}
  end

  describe "segments/1" do
    test "splits on frame-counter reset" do
      ds = dataset([100, 50, 200])
      assert TrajectoryCursors.segments(ds.frames) == [{0, 100}, {100, 50}, {150, 200}]
    end

    test "single game" do
      ds = dataset([42])
      assert TrajectoryCursors.segments(ds.frames) == [{0, 42}]
    end

    test "empty" do
      assert TrajectoryCursors.segments([]) == []
    end
  end

  describe "batch_stream/2" do
    test "raises when too few segments for batch_size" do
      ds = dataset([100, 100])

      assert_raise ArgumentError, ~r/segments/, fn ->
        TrajectoryCursors.batch_stream(ds, batch_size: 3, unroll: 20, gpu: false)
        |> Enum.take(1)
      end
    end

    test "skips segments shorter than unroll" do
      # 5-frame game can't fit unroll 20; only 2 usable segments remain
      ds = dataset([100, 5, 100])

      assert_raise ArgumentError, ~r/only 2 segments/, fn ->
        TrajectoryCursors.batch_stream(ds, batch_size: 3, unroll: 20, gpu: false)
        |> Enum.take(1)
      end
    end

    test "rows are contiguous across batches with overlap" do
      ds = dataset([100, 100])
      unroll = 20
      overlap = 1

      batches =
        TrajectoryCursors.batch_stream(ds,
          batch_size: 2,
          unroll: unroll,
          overlap: overlap,
          gpu: false
        )
        |> Enum.to_list()

      assert length(batches) >= 2
      [b1, b2 | _] = batches

      # Batch shapes
      assert Nx.shape(b1.states) == {2, unroll, 2}
      assert Nx.shape(b1.frame_weights) == {2, unroll}
      assert Nx.shape(b1.actions.buttons) == {2, unroll, 8}
      assert Nx.shape(b1.actions.main_x) == {2, unroll}

      # First batch: both rows fresh
      assert Nx.to_flat_list(b1.is_resetting) == [1, 1]
      # Second batch: both rows continue the same segments
      assert Nx.to_flat_list(b2.is_resetting) == [0, 0]

      # Contiguity: row r's batch-2 start == batch-1 start + unroll - overlap
      # (embedding value == global frame index by construction)
      for r <- 0..1 do
        first1 = b1.states[r][0][0] |> Nx.to_number()
        first2 = b2.states[r][0][0] |> Nx.to_number()
        assert first2 == first1 + unroll - overlap
      end
    end

    test "no chunk crosses a segment boundary; resets only at real boundaries" do
      # 2 games of 100 with unroll 30/overlap 1: each segment yields
      # chunks at offsets 0, 29, 58 (87+30 > 100 -> new segment).
      ds = dataset([100, 100])

      batches =
        TrajectoryCursors.batch_stream(ds,
          batch_size: 2,
          unroll: 30,
          overlap: 1,
          gpu: false
        )
        |> Enum.to_list()

      # Every chunk must lie inside one segment. Embedding col 0 is the
      # GLOBAL frame index by construction and segments are [0,100) and
      # [100,200), so a chunk's first and last frame must land in the
      # same 100-block and be exactly unroll-1 apart (contiguous slice).
      for b <- batches, r <- 0..1 do
        first = b.states[r][0][0] |> Nx.to_number() |> round()
        last = b.states[r][29][0] |> Nx.to_number() |> round()
        assert last - first == 29
        assert div(first, 100) == div(last, 100)
      end

      # Total resets == segments consumed (2), all in the first batch
      total_resets =
        batches |> Enum.map(&Nx.to_number(Nx.sum(&1.is_resetting))) |> Enum.sum()

      assert total_resets == 2
    end

    test "halts when queue is exhausted" do
      ds = dataset([50, 50])

      batches =
        TrajectoryCursors.batch_stream(ds,
          batch_size: 2,
          unroll: 40,
          overlap: 1,
          gpu: false
        )
        |> Enum.to_list()

      # Each 50-frame segment fits exactly one 40-frame chunk (offset 39
      # + 40 > 50), and there are exactly 2 segments for 2 rows.
      assert length(batches) == 1
    end

    test "per-timestep targets match the frames" do
      ds = dataset([100, 100])

      [b1 | _] =
        TrajectoryCursors.batch_stream(ds,
          batch_size: 2,
          unroll: 10,
          overlap: 1,
          gpu: false
        )
        |> Enum.take(1)

      # main_x of frame i is rem(i_within_game, 17); both rows start at
      # offset 0 of their (shuffled) segments, so t-th target is rem(t, 17).
      for r <- 0..1, t <- 0..9 do
        assert Nx.to_number(b1.actions.main_x[r][t]) == rem(t, 17)
      end
    end
  end
end
