defmodule ExPhil.Embeddings.ActionFrameBucketsTest do
  @moduledoc """
  `--action-frame-buckets N` (2026-09-09, the jab-chain lever): N one-hot
  dims per player over the parsed-space action frame. Pins the layout
  (size), the bucket semantics (own bucket below N-1, overflow at N-1),
  live/batched parity, and that the checkpoint <-> Agent canary carries
  the key.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Embeddings
  alias ExPhil.Embeddings.Canary
  alias ExPhil.Embeddings.Game, as: GameEmbed
  alias ExPhil.Embeddings.Player, as: PlayerEmbed

  @n 24

  defp player(af) do
    %Player{
      x: 10.0,
      y: 0.0,
      percent: 30.0,
      stock: 3,
      facing: 1,
      action: 44,
      action_frame: af,
      on_ground: true,
      character: 2,
      jumps_left: 2,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      invulnerable: false
    }
  end

  defp gs(af) do
    %GameState{frame: 100, stage: 32, players: %{1 => player(af), 2 => player(0)}, own_port: 1}
  end

  test "flag off: size unchanged; flag on: +N dims per player" do
    off = Embeddings.config([])
    on = Embeddings.config(action_frame_buckets: @n)
    assert on.player.action_frame_buckets == @n
    assert PlayerEmbed.embedding_size(on.player) == PlayerEmbed.embedding_size(off.player) + @n
    # two players in the game embedding
    assert GameEmbed.raw_embedding_size(on) == GameEmbed.raw_embedding_size(off) + 2 * @n
  end

  test "bucket semantics: own bucket below N-1, overflow bucket at or beyond N-1" do
    cfg = Embeddings.config(action_frame_buckets: @n).player

    for af <- [0, 6, 11, @n - 2] do
      hot = PlayerEmbed.embed_action_frame_buckets(player(af * 1.0), cfg) |> Nx.to_flat_list()
      assert Enum.at(hot, af) == 1.0, "frame #{af} not in its own bucket"
      assert Enum.sum(hot) == 1.0
    end

    for af <- [@n - 1, @n, 90] do
      hot = PlayerEmbed.embed_action_frame_buckets(player(af * 1.0), cfg) |> Nx.to_flat_list()
      assert Enum.at(hot, @n - 1) == 1.0, "frame #{af} not in the overflow bucket"
    end
  end

  test "live and batched paths agree with buckets on (parity)" do
    config = Embeddings.config(action_frame_buckets: @n)
    state = gs(7.0)

    live = GameEmbed.embed(state, nil, 1, config: config)

    [batched] =
      GameEmbed.embed_states_fast([state], 1, config: config) |> Nx.to_batched(1) |> Enum.to_list()

    batched = Nx.squeeze(batched, axes: [0])

    assert Nx.shape(live) == Nx.shape(batched)
    assert Nx.all_close(live, batched, atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "the canary fingerprint carries the key (a bucket mismatch is DETECTED)" do
    default = ExPhil.Embeddings.Game.Config.default()
    on = %{default | player: %{default.player | action_frame_buckets: @n}}

    assert Canary.compare(Canary.fingerprint_batched(on), Canary.fingerprint_live(on)) == :ok

    assert {:error, {:size_mismatch, _, _}} =
             Canary.compare(Canary.fingerprint_batched(default), Canary.fingerprint_live(on))
  end

  test "the training flag parses and defaults to off" do
    assert ExPhil.Training.Config.defaults()[:action_frame_buckets] == 0
    assert "--action-frame-buckets" in ExPhil.Training.Config.Parser.flags()
  end
end
