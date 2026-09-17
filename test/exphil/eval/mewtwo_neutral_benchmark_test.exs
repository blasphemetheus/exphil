defmodule ExPhil.Eval.MewtwoNeutralBenchmarkTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.MewtwoNeutralBenchmark, as: Benchmark

  defp player do
    %{
      action: 14,
      on_ground: true,
      in_hitstun: false,
      percent: 0.0,
      stock: 4,
      x: 0.0,
      y: 0.0,
      controller: %{button_l: false, button_r: false, main_stick_x: 0.5, main_stick_y: 0.5}
    }
  end

  defp replay(rows),
    do: %{metadata: %{stage: 32, players: [%{port: 1}, %{port: 2}]}, frames: rows}

  defp frames,
    do: for(f <- 0..39, do: %{frame_number: f, players: %{1 => player(), 2 => player()}})

  test "first observation initializes without inventing an opening" do
    score = Benchmark.score(replay(frames()), 39)
    assert score.frames == 40
    assert score.neutral.outcomes == %{censored: 1}
    assert score.wavedashes == 0
  end

  test "finalization frames do not affect scored stocks or openings" do
    rows =
      frames()
      |> Enum.map(fn g ->
        if g.frame_number > 35, do: put_in(g.players[1].stock, 0), else: g
      end)

    score = Benchmark.score(replay(rows), 35)
    assert score.deaths == 0
    assert score.frames == 36
  end

  test "instant landing requires diagonal dodge input and actual slide" do
    rows =
      frames()
      |> Enum.map(fn g ->
        p =
          cond do
            g.frame_number == 1 ->
              %{player() | action: 24}

            g.frame_number == 2 ->
              %{player() | action: 25, on_ground: false, y: 1.4}

            g.frame_number == 3 ->
              %{
                player()
                | action: 43,
                  controller: %{
                    button_l: true,
                    button_r: false,
                    main_stick_x: 0.9,
                    main_stick_y: 0.2
                  }
              }

            g.frame_number in 4..13 ->
              %{player() | action: 43, x: (g.frame_number - 3) * 2.0}

            true ->
              player()
          end

        put_in(g.players[1], p)
      end)

    assert Benchmark.score(replay(rows), 39).wavedashes == 1
    still = Enum.map(rows, &put_in(&1.players[1].x, 0.0))
    assert Benchmark.score(replay(still), 39).wavedashes == 0
  end
end
