defmodule ExPhil.Embeddings.StageInternalsTest do
  # Stage internals (W4 2026-08-24): FoD platform heights + PS
  # transformation as embedding features, zero-gated by stage.
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Game, as: GameEmbed
  alias ExPhil.Embeddings.Game.Stage
  alias ExPhil.Bridge.{GameState, Player}

  defp player(x \\ 0.0, y \\ 0.0) do
    %Player{
      character: 2,
      x: x,
      y: y,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 14,
      action_frame: 1.0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_y_self: 0.0
    }
  end

  defp gs(stage, extra \\ []) do
    struct!(
      %GameState{
        frame: 100,
        stage: stage,
        menu_state: 2,
        players: %{1 => player(), 2 => player(10.0, 0.0)},
        projectiles: [],
        distance: 10.0
      },
      extra
    )
  end

  describe "Stage.internals_values/1" do
    test "FoD (external 2): normalized heights, zero PS block" do
      vals = Stage.internals_values(%{stage: 2, fod_platform_left: 21.0, fod_platform_right: 28.0})
      assert vals == [21.0 / 35.0, 28.0 / 35.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "FoD nil heights fall back to the 20/28 start values" do
      assert [l, r | _] = Stage.internals_values(%{stage: 2})
      assert_in_delta l, 20.0 / 35.0, 1.0e-6
      assert_in_delta r, 28.0 / 35.0, 1.0e-6
    end

    test "PS (external 3): transformation one-hot, zero heights" do
      # water (Slippi type 9) -> class 4
      assert Stage.internals_values(%{stage: 3, stadium_type: 9}) ==
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

      # nil -> normal (class 0)
      assert Stage.internals_values(%{stage: 3}) ==
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "other stages: all zeros" do
      assert Stage.internals_values(%{stage: 32, fod_platform_left: 25.0, stadium_type: 3}) ==
               List.duplicate(0.0, 7)
    end
  end

  describe "embedding integration" do
    test "flag off: size unchanged; flag on: +7 raw dims" do
      base = ExPhil.Embeddings.config([])
      with_flag = ExPhil.Embeddings.config(stage_internals: true)

      assert GameEmbed.raw_embedding_size(with_flag) ==
               GameEmbed.raw_embedding_size(base) + Stage.internals_size()
    end

    test "live and batched paths agree with the flag on (parity)" do
      config = ExPhil.Embeddings.config(stage_internals: true)
      state = gs(2, fod_platform_left: 24.5, fod_platform_right: 16.0)

      live = GameEmbed.embed(state, nil, 1, config: config)
      [batched] = GameEmbed.embed_states_fast([state], 1, config: config) |> Nx.to_batched(1) |> Enum.to_list()
      batched = Nx.squeeze(batched, axes: [0])

      assert Nx.shape(live) == Nx.shape(batched)
      assert Nx.all_close(live, batched, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "embedding responds to the transformation iff the flag is on" do
      on = ExPhil.Embeddings.config(stage_internals: true)
      off = ExPhil.Embeddings.config([])

      rock = gs(3, stadium_type: 6)
      water = gs(3, stadium_type: 9)

      diff = fn config ->
        a = GameEmbed.embed(rock, nil, 1, config: config)
        b = GameEmbed.embed(water, nil, 1, config: config)
        Nx.abs(Nx.subtract(a, b)) |> Nx.sum() |> Nx.to_number()
      end

      assert diff.(on) > 0.5, "flag on: transform change must move the embedding"
      assert diff.(off) == 0.0, "flag off: transform must be invisible"
    end
  end
end
