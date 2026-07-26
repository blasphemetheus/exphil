defmodule ExPhil.Embeddings.AfConventionTest do
  @moduledoc """
  The embedding-boundary half of task #8 phase 2 option 1 (GOTCHAS #81).

  Live regression against Dolphin is home-only; these pin the behavior that
  can be verified offline.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Embeddings.Player, as: PlayerEmbed

  # action 24 (jumpsquat) has delta 1: live af 1 is parsed af 0.
  defp player(action, af) do
    %PlayerState{
      character: 2,
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: true,
      action: action,
      action_frame: af,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0
    }
  end

  defp parsed_config, do: %PlayerEmbed{}
  defp live_config, do: %PlayerEmbed{af_convention: :live}

  describe "default is an exact no-op" do
    test "af_convention defaults to :parsed" do
      assert PlayerEmbed.default_config().af_convention == :parsed
    end

    test "parsed config embeds action_frame unchanged" do
      # Guards every existing checkpoint: turning this feature on must be
      # opt-in, so the default path has to be byte-identical to before.
      a = PlayerEmbed.embed_frame_info(player(24, 2), parsed_config())
      b = PlayerEmbed.embed_frame_info(player(24, 2))
      assert Nx.to_flat_list(a) == Nx.to_flat_list(b)
    end

    test "embedding size is unaffected by the convention" do
      assert PlayerEmbed.embedding_size(parsed_config()) ==
               PlayerEmbed.embedding_size(live_config())
    end
  end

  describe ":live converts into parsed space" do
    test "live af 1 embeds identically to parsed af 0 for jumpsquat" do
      live = PlayerEmbed.embed_frame_info(player(24, 1), live_config())
      parsed = PlayerEmbed.embed_frame_info(player(24, 0), parsed_config())

      assert Nx.to_flat_list(live) == Nx.to_flat_list(parsed)
    end

    test "and differs from treating the live value as parsed" do
      # If this ever passes, the normalization is not actually happening.
      live = PlayerEmbed.embed_frame_info(player(24, 1), live_config())
      naive = PlayerEmbed.embed_frame_info(player(24, 1), parsed_config())

      refute Nx.to_flat_list(live) == Nx.to_flat_list(naive)
    end

    test "agreeing actions are untouched even under :live" do
      # 365 has delta 0 — a blanket "subtract 1" would corrupt it.
      live = PlayerEmbed.embed_frame_info(player(365, 2), live_config())
      parsed = PlayerEmbed.embed_frame_info(player(365, 2), parsed_config())

      assert Nx.to_flat_list(live) == Nx.to_flat_list(parsed)
    end

    test "unmeasured actions are untouched under :live" do
      # 252 = cliff catch, which no recorded pair covers yet.
      live = PlayerEmbed.embed_frame_info(player(252, 5), live_config())
      parsed = PlayerEmbed.embed_frame_info(player(252, 5), parsed_config())

      assert Nx.to_flat_list(live) == Nx.to_flat_list(parsed)
    end
  end

  describe "batch path matches the single path" do
    test "embed_batch applies the same conversion" do
      # The two code paths read action_frame separately; they must not drift.
      single = PlayerEmbed.embed(player(24, 1), live_config())
      batched = PlayerEmbed.embed_batch([player(24, 1)], live_config())

      assert Nx.to_flat_list(batched) == Nx.to_flat_list(single)
    end

    test "batch :live equals batch :parsed shifted to parsed af" do
      live = PlayerEmbed.embed_batch([player(24, 1)], live_config())
      parsed = PlayerEmbed.embed_batch([player(24, 0)], parsed_config())

      assert Nx.to_flat_list(live) == Nx.to_flat_list(parsed)
    end

    test "batch tolerates nil players under :live" do
      out = PlayerEmbed.embed_batch([player(24, 1), nil], live_config())
      assert Nx.shape(out) |> elem(0) == 2
    end
  end
end
