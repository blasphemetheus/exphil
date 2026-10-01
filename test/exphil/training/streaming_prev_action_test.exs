defmodule ExPhil.Training.StreamingPrevActionTest do
  @moduledoc """
  `--prev-action` through the STREAMING path (2026-09-30). Until this date
  `Streaming.create_dataset/2` never forwarded `use_prev_action`, so streamed
  chunks embedded an all-zero previous-action slot while the flag looked on
  (fox-mamba-v2-prevact's first launch). Contract: with the flag, frame i's
  embedding carries frame i-1's controller; without it (or at a game start)
  the slot is zeros; the width never changes.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.Streaming
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  defp frame(opts) do
    player = %Player{
      character: 2, x: Keyword.get(opts, :x, 0.0), y: 0.0, percent: 0.0, stock: 4, facing: 1,
      action: 14, action_frame: 0, invulnerable: false, jumps_left: 2, on_ground: true,
      shield_strength: 60.0, hitstun_frames_left: 0, speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0, speed_y_self: 0.0, speed_x_attack: 0.0, speed_y_attack: 0.0,
      nana: nil, controller_state: nil
    }

    controller = %ControllerState{
      main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
      button_a: Keyword.get(opts, :button_a, false), button_b: false, button_x: false,
      button_y: false, button_z: false, button_l: false, button_r: false, button_d_up: false
    }

    %{
      game_state: %GameState{
        frame: Keyword.fetch!(opts, :frame), stage: 32, menu_state: 2,
        players: %{1 => player, 2 => player}, projectiles: [], distance: 50.0
      },
      controller: controller
    }
  end

  defp embed(frames, opts) do
    Streaming.create_dataset(frames, [temporal: false, precompute: true] ++ opts)
    |> Map.fetch!(:embedded_frames)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_list()
  end

  test "streamed chunks carry the previous frame's controller when use_prev_action is on" do
    # frame 0: A pressed; frame 1: nothing pressed. With the channel on,
    # frame 1's embedding must differ from the flag-off embedding (it now
    # holds frame 0's A press); frame 0 (game start, no predecessor) must not.
    frames = [frame(frame: 0, button_a: true), frame(frame: 1)]
    off = embed(frames, use_prev_action: false)
    on = embed(frames, use_prev_action: true)

    assert length(hd(on)) == length(hd(off)), "prev-action must not change the embedding width"
    assert hd(on) == hd(off), "game-start frame has no predecessor: slot stays zeros"
    refute Enum.at(on, 1) == Enum.at(off, 1), "frame 1 must see frame 0's A press"

    changed = Enum.zip(Enum.at(on, 1), Enum.at(off, 1)) |> Enum.count(fn {a, b} -> a != b end)
    assert changed >= 1 and changed <= 13, "only the 13-dim controller slot may differ, got #{changed}"
  end

  test "block dropout masks whole runs of frames, not scattered ones" do
    # Every frame holds A, so an unmasked frame i >= 1 always differs from the
    # channel-off embedding. With block 10 each block is all-masked or
    # all-visible; per-frame masking would mix them inside a block.
    frames = for i <- 0..199, do: frame(frame: i, button_a: true)
    off = embed(frames, use_prev_action: false)

    on =
      embed(frames, use_prev_action: true, prev_action_dropout: 0.5, prev_action_dropout_block: 10)

    masked = Enum.zip_with(on, off, &(&1 == &2))

    blocks =
      masked |> Enum.drop(1) |> Enum.with_index(1) |> Enum.group_by(fn {_, i} -> div(i, 10) end, &elem(&1, 0))

    assert Enum.all?(blocks, fn {_b, ms} -> length(Enum.uniq(ms)) == 1 end)
    kinds = blocks |> Map.values() |> Enum.map(&hd/1) |> Enum.uniq() |> Enum.sort()
    assert kinds == [false, true], "20 blocks at p=0.5 should contain both masked and visible blocks"
  end

  test "the flag defaults to off in streaming, matching the pre-wire behaviour" do
    frames = [frame(frame: 0, button_a: true), frame(frame: 1)]
    assert embed(frames, []) == embed(frames, use_prev_action: false)
  end
end
