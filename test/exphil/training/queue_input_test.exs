defmodule ExPhil.Training.QueueInputTest do
  @moduledoc """
  Queue-as-input alignment pins (2026-07-31, slippi-ai delayed_actions
  parity). Slot k of the committed-action queue at frame i must be the
  action committed k decision-steps ago:

    * plain replays:  frames[i-k].controller
    * DAgger frames:  frames[i-k+1].prev_controller (the POLICY's press,
      never the expert correction)
    * boundaries (frame-number discontinuity) nil the slot (zeros)

  Off-by-ones here are invisible to val_loss and cost live behavior —
  same class as the embed-path parity bug. Everything below is checked
  at the EMBEDDING level through the public precompute API.
  """
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Data
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Embeddings.Game.Config
  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  defp controller(x) do
    %ControllerState{
      main_stick: %{x: x, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0,
      button_a: false,
      button_b: x > 0.5,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  defp frame(n, ctrl, extra \\ []) do
    player = %Player{
      character: 2,
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 14,
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }

    base = %{
      game_state: %GameState{
        frame: n,
        stage: 32,
        menu_state: 2,
        players: %{1 => player, 2 => player},
        projectiles: [],
        distance: 50.0
      },
      controller: ctrl
    }

    Enum.into(extra, base)
  end

  defp config(depth, delay_id?) do
    %{Config.default() | queue_depth: depth, with_delay_id: delay_id?}
  end

  # Offset of the queue block inside the embedding (matches both embed
  # paths: own + opp + stage precede it).
  defp queue_offset(cfg) do
    2 * PlayerEmbed.embedding_size(cfg.player) +
      ExPhil.Embeddings.Game.Stage.embedding_size(cfg.stage_mode)
  end

  defp slot(emb, cfg, i, k) do
    s = ControllerEmbed.continuous_embedding_size()
    emb[i][(queue_offset(cfg) + (k - 1) * s)..(queue_offset(cfg) + k * s - 1)]
  end

  defp expected(ctrl_or_nil) do
    ControllerEmbed.embed_continuous(ctrl_or_nil) |> Nx.to_flat_list()
  end

  defp precompute(frames, cfg, opts \\ []) do
    frames
    |> Data.from_frames()
    |> Map.put(:embed_config, cfg)
    |> Data.precompute_frame_embeddings(
      Keyword.merge([use_prev_action: true, show_progress: false], opts)
    )
    |> Map.fetch!(:embedded_frames)
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  test "queue_depth 3: slot k is the controller from k frames back" do
    ctrls = Enum.map(0..4, fn i -> controller(i / 10) end)
    frames = Enum.with_index(ctrls) |> Enum.map(fn {c, i} -> frame(i, c) end)
    cfg = config(3, false)
    emb = precompute(frames, cfg)

    # frame 4: slot1 = ctrl 3, slot2 = ctrl 2, slot3 = ctrl 1
    for {k, src} <- [{1, 3}, {2, 2}, {3, 1}] do
      assert Nx.to_flat_list(slot(emb, cfg, 4, k)) == expected(Enum.at(ctrls, src)),
             "slot #{k} at frame 4 should be controller #{src}"
    end

    # frame 1: slot1 = ctrl 0, slots 2-3 out of range -> zeros
    assert Nx.to_flat_list(slot(emb, cfg, 1, 1)) == expected(Enum.at(ctrls, 0))
    assert Nx.to_flat_list(slot(emb, cfg, 1, 2)) == expected(nil)
    assert Nx.to_flat_list(slot(emb, cfg, 1, 3)) == expected(nil)
  end

  test "queue_depth 3: frame discontinuity nils exactly the crossing slots" do
    ctrls = Enum.map(0..4, fn i -> controller(i / 10) end)
    # frames 0,1,2 then a GAP then 10,11
    numbers = [0, 1, 2, 10, 11]

    frames =
      Enum.zip(numbers, ctrls) |> Enum.map(fn {n, c} -> frame(n, c) end)

    cfg = config(3, false)
    emb = precompute(frames, cfg)

    # index 4 (frame 11): slot1 = index 3 (frame 10, consecutive) — valid;
    # slots 2-3 cross the gap -> zeros
    assert Nx.to_flat_list(slot(emb, cfg, 4, 1)) == expected(Enum.at(ctrls, 3))
    assert Nx.to_flat_list(slot(emb, cfg, 4, 2)) == expected(nil)
    assert Nx.to_flat_list(slot(emb, cfg, 4, 3)) == expected(nil)
  end

  test "queue_depth 3 DAgger frames: slots come from prev_controller, never :controller" do
    expert = Enum.map(0..4, fn _ -> controller(0.9) end)
    policy = Enum.map(0..4, fn i -> controller(i / 10) end)

    frames =
      Enum.zip([0..4 |> Enum.to_list(), expert, policy])
      |> Enum.map(fn {n, e, p} -> frame(n, e, prev_controller: p) end)

    cfg = config(3, false)
    emb = precompute(frames, cfg)

    # slot k at i = frames[i-k+1].prev_controller = policy press at i-k.
    # frame 4: slot1 = policy[4] (own prev_controller), slot2 = policy[3],
    # slot3 = policy[2]. The expert 0.9 controller must appear in NO slot.
    for {k, src} <- [{1, 4}, {2, 3}, {3, 2}] do
      got = Nx.to_flat_list(slot(emb, cfg, 4, k))
      assert got == expected(Enum.at(policy, src)), "slot #{k} at frame 4"
      refute got == expected(Enum.at(expert, 0))
    end
  end

  test "delay_id one-hot lands after the queue block and encodes the id" do
    ctrls = Enum.map(0..2, fn i -> controller(i / 10) end)
    frames = Enum.with_index(ctrls) |> Enum.map(fn {c, i} -> frame(i, c) end)
    cfg = config(2, true)
    emb = precompute(frames, cfg, delay_id: 3)

    s = ControllerEmbed.continuous_embedding_size()
    off = queue_offset(cfg) + 2 * s
    one_hot = emb[2][off..(off + Config.delay_id_size() - 1)] |> Nx.to_flat_list()
    assert one_hot == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  end

  test "queue_depth 1 stays byte-identical to the classic prev-action channel" do
    ctrls = Enum.map(0..3, fn i -> controller(i / 10) end)
    frames = Enum.with_index(ctrls) |> Enum.map(fn {c, i} -> frame(i, c) end)

    emb_default = precompute(frames, Config.default())
    emb_depth1 = precompute(frames, config(1, false))

    assert Nx.to_flat_list(emb_default) == Nx.to_flat_list(emb_depth1)
  end
end
