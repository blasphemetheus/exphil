defmodule ExPhil.Embeddings.RepresentationalLimitsTest do
  @moduledoc """
  Hunt #5: can the REPRESENTATION carry key Melee behaviors? The overfit gates
  certified the pipeline reproduces what it can express — these tests probe
  what it can express at all.

  - Stick discretization (17 uniform buckets, 0..1 space): do the critical
    control inputs (shine-down, wavedash diagonals, walk-vs-dash deflection)
    survive discretize → undiscretize with their SEMANTIC CLASS intact?
  - Single-frame state: is a dash-dance's state sequence pairwise
    distinguishable in the embedding (the precondition for any single-frame
    model to express it)?
  """
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Bridge.{GameState, Player}

  @buckets 16

  defp round_trip(v) do
    v
    |> ControllerEmbed.discretize_axis(@buckets)
    |> ControllerEmbed.undiscretize_axis(@buckets)
  end

  describe "stick discretization semantic classes (17 buckets)" do
    test "critical inputs round-trip within half a bucket" do
      # 0..1 stick space, 0.5 = neutral. Values from libmelee conventions.
      criticals = %{
        neutral: 0.5,
        full_down: 0.0,
        full_up: 1.0,
        full_left: 0.0,
        full_right: 1.0,
        # Perfect-wavedash x deflection (~0.95 of full range → 0.5 ± 0.475)
        wd_left_x: 0.025,
        wd_right_x: 0.975,
        # Wavedash y (slightly below horizontal, ~0.28 down)
        wd_y: 0.36,
        # Walk (sub-dash) deflection ~0.35 of range vs dash > 0.8
        walk_right: 0.675,
        dash_right: 0.95
      }

      half_bucket = 0.5 / @buckets

      for {name, v} <- criticals do
        rt = round_trip(v)

        assert abs(rt - v) <= half_bucket + 1.0e-9,
               "#{name}: #{v} round-trips to #{rt} (> half-bucket error)"
      end
    end

    test "walk and dash deflections stay in DISTINCT buckets" do
      # If walk tilt and full dash collapse into one bucket, the policy
      # cannot express the walk/dash distinction at all.
      walk = ControllerEmbed.discretize_axis(0.675, @buckets)
      dash = ControllerEmbed.discretize_axis(0.95, @buckets)
      neutral = ControllerEmbed.discretize_axis(0.5, @buckets)

      assert walk != dash, "walk (0.675) and dash (0.95) collapse to bucket #{walk}"
      assert walk != neutral, "walk (0.675) indistinguishable from neutral"
    end

    test "wavedash diagonal survives with airdodge-angle semantics" do
      # A wavedash needs: strong horizontal deflection + below-horizontal y.
      # After round-trip, the decoded stick must still satisfy both.
      for {x, y} <- [{0.025, 0.36}, {0.975, 0.36}] do
        rx = round_trip(x)
        ry = round_trip(y)

        horizontal_deflection = abs(rx - 0.5) * 2
        assert horizontal_deflection > 0.8,
               "wavedash x=#{x} decodes to #{rx} — deflection #{horizontal_deflection} too weak"

        assert ry < 0.5, "wavedash y=#{y} decodes to #{ry} — no longer below horizontal"
      end
    end

    test "shine input class is preserved (down + B detectability)" do
      # ReplicationCheck and the fixtures define shine as y < 0.5 — the
      # discretized/decoded down input must stay in that class.
      for y <- [0.0, 0.1, 0.2] do
        assert round_trip(y) < 0.5
      end
    end
  end

  describe "single-frame distinguishability of dash-dance states" do
    # Dash-dance: alternating left/right dashes. Period-8: 4 frames dashing
    # right, 4 dashing left, with action_frame advancing within each dash.
    defp dash_dance_states do
      for i <- 0..7 do
        phase = rem(i, 8)
        facing = if phase < 4, do: 1, else: -1
        action_frame = rem(phase, 4) + 1

        player = %Player{
          # Fox
          character: 2,
          x: 0.0 + facing * action_frame * 1.5,
          y: 0.0,
          percent: 0.0,
          stock: 4,
          facing: facing,
          # DASH
          action: 20,
          action_frame: action_frame,
          on_ground: true,
          jumps_left: 2,
          shield_strength: 60.0,
          invulnerable: false,
          hitstun_frames_left: 0,
          speed_air_x_self: 0.0,
          speed_ground_x_self: facing * 1.8,
          speed_y_self: 0.0,
          speed_x_attack: 0.0,
          speed_y_attack: 0.0,
          nana: nil
        }

        opponent = %Player{player | character: 10, x: 40.0, facing: -1, action: 14, action_frame: 1, speed_ground_x_self: 0.0}

        %GameState{
          frame: i,
          stage: 32,
          menu_state: 2,
          players: %{1 => player, 2 => opponent},
          projectiles: []
        }
      end
    end

    test "all 8 dash-dance phase states embed to pairwise-distinct vectors" do
      config = Embeddings.config()

      embedded =
        dash_dance_states()
        |> Enum.map(&Embeddings.Game.embed(&1, nil, 1, config: config))

      pairs = for i <- 0..7, j <- (i + 1)..7//1, do: {i, j}

      for {i, j} <- pairs do
        diff =
          Enum.at(embedded, i)
          |> Nx.subtract(Enum.at(embedded, j))
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff > 1.0e-6,
               "dash-dance phases #{i} and #{j} embed identically — a single-frame " <>
                 "model cannot express dash-dance (needs temporal context or richer state)"
      end
    end
  end
end
