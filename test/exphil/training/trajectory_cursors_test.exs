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
      buttons: %{
        a: rem(id, 2) == 1,
        b: false,
        x: false,
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      },
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
    test "splits on forward gaps as well as resets" do
      frames = Enum.map([0, 1, 5, 6, 0, 1], &frame(&1, 0))
      assert TrajectoryCursors.segments(frames) == [{0, 2}, {2, 2}, {4, 2}]
    end

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
    test "coverage and reset laws hold across boundary lengths, batch widths and seeds" do
      for unroll <- [1, 3, 10], batch_size <- [1, 2, 7], seed <- [1, 905, 906] do
        lengths = [1, unroll, unroll + 1, max(1, unroll - 1), 2 * unroll + 1]
        ds = dataset(lengths)
        segments = TrajectoryCursors.segments(ds.frames)

        opts = [
          unroll: unroll,
          batch_size: batch_size,
          seed: seed,
          gpu: false,
          neutral_weight: 1.0
        ]

        batches = Enum.to_list(TrajectoryCursors.batch_stream(ds, opts))

        {seen, _} =
          Enum.reduce(batches, {[], %{}}, fn b, {seen, previous} ->
            Enum.reduce(0..(batch_size - 1), {seen, previous}, fn row, {ids, prev} ->
              count = Nx.to_number(Nx.sum(b.valid_mask[row])) |> round()

              if count == 0 do
                assert Nx.to_number(b.is_resetting[row]) == 1
                {ids, prev}
              else
                row_ids =
                  Nx.slice_along_axis(b.states[row], 0, count, axis: 0)
                  |> Nx.slice_along_axis(0, 1, axis: 1)
                  |> Nx.to_flat_list()
                  |> Enum.map(&round/1)

                first = hd(row_ids)
                {start, len} = Enum.find(segments, fn {s, n} -> first >= s and first < s + n end)
                assert List.last(row_ids) < start + len

                if Nx.to_number(b.is_resetting[row]) == 1,
                  do: assert(first == start),
                  else: assert(first == prev[row] + 1)

                {row_ids ++ ids, Map.put(prev, row, List.last(row_ids))}
              end
            end)
          end)

        assert Enum.sort(seen) == Enum.to_list(0..(ds.size - 1))
        repeated = Enum.to_list(TrajectoryCursors.batch_stream(ds, opts))

        assert Enum.map(batches, &Nx.serialize(&1.states)) ==
                 Enum.map(repeated, &Nx.serialize(&1.states))
      end
    end

    test "scores every frame exactly once across short, long and partial segments" do
      for lengths <- [[80, 8000], [3, 100, 1, 81, 159], List.duplicate(160, 40)],
          batch_size <- [2, 128] do
        ds = dataset(lengths)

        batches =
          TrajectoryCursors.batch_stream(ds,
            batch_size: batch_size,
            unroll: 80,
            overlap: 0,
            gpu: false,
            neutral_weight: 1.0
          )
          |> Enum.to_list()

        seen =
          Enum.flat_map(batches, fn b ->
            assert Nx.shape(b.states) == {batch_size, 80, 2}
            ids = Nx.slice_along_axis(b.states, 0, 1, axis: 2) |> Nx.to_flat_list()

            Enum.zip(ids, Nx.to_flat_list(b.valid_mask))
            |> Enum.filter(fn {_, valid} -> valid == 1 end)
            |> Enum.map(fn {id, _} -> round(id) end)
          end)

        assert Enum.sort(seen) == Enum.to_list(0..(ds.size - 1))
        assert Enum.sum(Enum.map(batches, &Nx.to_number(Nx.sum(&1.frame_weights)))) == ds.size
      end
    end

    test "rows advance without overlap and reset before a different segment" do
      ds = dataset([90, 11, 210, 5])
      segments = TrajectoryCursors.segments(ds.frames)

      batches =
        TrajectoryCursors.batch_stream(ds, batch_size: 2, unroll: 80, gpu: false)
        |> Enum.to_list()

      Enum.reduce(batches, %{}, fn b, previous ->
        Enum.reduce(0..1, previous, fn row, prev ->
          valid = Nx.to_number(Nx.sum(b.valid_mask[row])) |> round()

          if valid > 0 do
            first = Nx.to_number(b.states[row][0][0]) |> round()
            last = Nx.to_number(b.states[row][valid - 1][0]) |> round()
            {start, length} = Enum.find(segments, fn {s, n} -> first >= s and first < s + n end)
            assert last == first + valid - 1
            assert last < start + length

            if Nx.to_number(b.is_resetting[row]) == 0,
              do: assert(first == prev[row] + 1),
              else: assert(first == start)

            Map.put(prev, row, last)
          else
            assert Nx.to_number(b.is_resetting[row]) == 1
            prev
          end
        end)
      end)
    end

    test "positive overlap is rejected rather than replaying input into advanced carry" do
      assert_raise ArgumentError, ~r/overlap must be 0/, fn ->
        TrajectoryCursors.batch_stream(dataset([100]), batch_size: 1, overlap: 1)
      end
    end

    test "empty datasets emit no batches" do
      assert Enum.to_list(
               TrajectoryCursors.batch_stream(%Data{frames: [], size: 0},
                 batch_size: 128,
                 gpu: false
               )
             ) == []
    end

    test "padding contributes zero weight and valid targets remain aligned" do
      [b] =
        TrajectoryCursors.batch_stream(dataset([5]),
          batch_size: 2,
          unroll: 80,
          gpu: false,
          neutral_weight: 1.0
        )
        |> Enum.to_list()

      assert Nx.to_number(Nx.sum(b.valid_mask)) == 5

      for row <- 0..1, t <- 0..79 do
        if Nx.to_number(b.valid_mask[row][t]) == 1 do
          assert Nx.to_number(b.actions.main_x[row][t]) == rem(t, 17)
        else
          assert Nx.to_number(b.frame_weights[row][t]) == 0
        end
      end
    end
  end
end
