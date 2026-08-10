defmodule ExPhil.Embeddings.PerStageLedgeTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.GameState
  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Embeddings
  alias ExPhil.Embeddings.Player, as: PlayerEmbed

  @yoshis 8
  @ys_edge 56.0

  defp player(x) do
    %PlayerState{x: x, y: 0.0, percent: 0.0, stock: 4, action: 14, on_ground: true}
  end

  defp game_state(x, stage) do
    %GameState{
      frame: 100,
      stage: stage,
      players: %{1 => player(x), 2 => player(-20.0)}
    }
  end

  test "embed_ledge_distance defaults to the 85 constant" do
    # x=60: constant says (85-60)/85 ≈ 0.294 ("safe")
    [d] = PlayerEmbed.embed_ledge_distance(player(60.0)) |> Nx.to_flat_list()
    assert_in_delta d, (85.0 - 60.0) / 85.0, 1.0e-6
  end

  test "embed_ledge_distance with a real YS edge flags x=60 as offstage" do
    [d] = PlayerEmbed.embed_ledge_distance(player(60.0), @ys_edge) |> Nx.to_flat_list()
    assert d < 0.0, "x=60 on YS (edge #{@ys_edge}) must read offstage, got #{d}"
  end

  test "embed_batch threads per-player stage edges" do
    players = [player(60.0), player(60.0)]
    base = PlayerEmbed.embed_batch(players)
    per_stage = PlayerEmbed.embed_batch(players, PlayerEmbed.default_config(), [@ys_edge, nil])

    # Same shape; only the ledge dim of player 1 differs
    assert Nx.shape(base) == Nx.shape(per_stage)

    diff = Nx.subtract(base, per_stage) |> Nx.abs() |> Nx.sum(axes: [1]) |> Nx.to_flat_list()
    [p1_diff, p2_diff] = diff
    assert p1_diff > 0.1
    assert_in_delta p2_diff, 0.0, 1.0e-6
  end

  test "game embed: per_stage_ledge off is bit-identical to the old path" do
    gs = game_state(60.0, @yoshis)
    off = Embeddings.config()
    emb = ExPhil.Embeddings.Game.embed(gs, nil, 1, config: off)
    # The flag defaults off — same call through a config that names it
    off2 = Embeddings.config(per_stage_ledge: false)
    emb2 = ExPhil.Embeddings.Game.embed(gs, nil, 1, config: off2)
    assert Nx.to_flat_list(emb) == Nx.to_flat_list(emb2)
  end

  test "game embed: per_stage_ledge on changes the embedding on YS but keeps size" do
    gs = game_state(60.0, @yoshis)
    off = Embeddings.config()
    on = Embeddings.config(per_stage_ledge: true)

    emb_off = ExPhil.Embeddings.Game.embed(gs, nil, 1, config: off)
    emb_on = ExPhil.Embeddings.Game.embed(gs, nil, 1, config: on)

    assert Nx.shape(emb_off) == Nx.shape(emb_on)
    refute Nx.to_flat_list(emb_off) == Nx.to_flat_list(emb_on)

    # The REAL YS edge (56) must appear, not internal-id-collision FoD's
    # 63.35. Both players' ledge dims change (own x=60, opp x=-20); own
    # is embedded first. At x=60 the real YS value is (56-60)/56 < 0.
    off_l = Nx.to_flat_list(emb_off)
    on_l = Nx.to_flat_list(emb_on)
    changed = Enum.zip(off_l, on_l) |> Enum.filter(fn {a, b} -> a != b end)
    assert [{own_off, own_on}, {_opp_off, opp_on}] = changed
    assert_in_delta own_off, (85.0 - 60.0) / 85.0, 1.0e-4
    assert_in_delta own_on, (56.0 - 60.0) / 56.0, 1.0e-4
    assert_in_delta opp_on, (56.0 - 20.0) / 56.0, 1.0e-4
  end

  test "batch path (embed_states_fast) matches the live path with the flag on" do
    gs = game_state(60.0, @yoshis)
    on = Embeddings.config(per_stage_ledge: true)

    live = ExPhil.Embeddings.Game.embed(gs, nil, 1, config: on)
    [batch] = ExPhil.Embeddings.Game.embed_states_fast([gs], 1, config: on) |> Nx.to_batched(1) |> Enum.to_list()

    assert_in_delta(
      Nx.subtract(live, Nx.squeeze(batch, axes: [0])) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number(),
      0.0,
      1.0e-5
    )
  end

  test "unknown stage falls back to the 85 constant with the flag on" do
    gs_unknown = game_state(60.0, 0)
    on = Embeddings.config(per_stage_ledge: true)
    off = Embeddings.config()

    emb_on = ExPhil.Embeddings.Game.embed(gs_unknown, nil, 1, config: on)
    emb_off = ExPhil.Embeddings.Game.embed(gs_unknown, nil, 1, config: off)

    assert Nx.to_flat_list(emb_on) == Nx.to_flat_list(emb_off)
  end
end
