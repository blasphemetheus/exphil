defmodule ExPhil.Data.LibmeleeActionFrameTest do
  use ExUnit.Case, async: true
  alias ExPhil.Data.ActionFrameConvention, as: AFC

  test "inverts the producer indexing rule across characters and actions" do
    for character <- 0..32, action <- 0..398, raw <- [-1, 0, 1, 11] do
      emitted = if Melee.FrameData.zero_indexed?(character, action), do: raw + 1, else: raw
      assert AFC.libmelee_to_parsed(character, action, emitted) == raw
    end
  end

  test "Fox reflector hold is covered without widening the invalid historical table" do
    assert AFC.libmelee_to_parsed(1, 363, 12) == 11
    assert AFC.libmelee_to_parsed(1, 365, 2) == 2
    refute AFC.known?(363)
    assert AFC.libmelee_to_parsed(nil, 363, 12) == 12
  end

  test "the scenario producer contract is explicit and legacy behavior is opt-in" do
    assert AFC.scenario_convention([]) == :libmelee
    assert AFC.scenario_convention(live_af: true) == :libmelee
    assert AFC.scenario_convention(live_af: false) == :parsed
  end

  test "all 1440 pinned same-frame live/replay observations obey the producer contract" do
    report =
      File.read!("test/fixtures/statestream/libmelee_action_frame_pairs.json") |> Jason.decode!()

    assert report["observations"] == 1440
    assert Enum.sum(Enum.map(report["samples"], & &1["count"])) == 1440

    for sample <- report["samples"] do
      assert AFC.libmelee_to_parsed(sample["character"], sample["action"], sample["libmelee"]) ==
               sample["parsed"]
    end
  end

  test "scalar and batched embeddings apply the same character-aware conversion" do
    alias ExPhil.Embeddings.Player, as: Embed

    player = %ExPhil.Bridge.Player{
      character: 1,
      action: 363,
      action_frame: 12,
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      on_ground: true,
      jumps_left: 2,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      invulnerable: false
    }

    parsed = %{player | action_frame: 11}
    config = ExPhil.Embeddings.config(af_convention: :libmelee, action_frame_buckets: 24).player
    baseline = %{config | af_convention: :parsed}

    assert Nx.all_close(Embed.embed(player, config), Embed.embed(parsed, baseline))
           |> Nx.to_number() == 1

    assert Nx.all_close(
             Embed.embed_batch([player], config),
             Embed.embed_batch([parsed], baseline)
           )
           |> Nx.to_number() == 1
  end
end
