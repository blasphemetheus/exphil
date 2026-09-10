defmodule ExPhil.Embeddings.SourceChannelsTest do
  @moduledoc "INVARIANTS.md item 4: train-time-absent input channels are absent live."
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent
  alias ExPhil.Data.Peppi
  alias ExPhil.Embeddings
  alias ExPhil.Training.Config

  test "Peppi does not provide projectiles or items" do
    refute Peppi.provides?(:projectiles)
    refute Peppi.provides?(:items)
    assert Peppi.provides?(:players)
  end

  test "config_for_source drops the projectile block for a source that lacks it (smaller embedding)" do
    with_block = Embeddings.config(with_projectiles: true)
    without = Embeddings.config_for_source([with_projectiles: true], Peppi.provides())
    assert without.with_projectiles == false
    assert Embeddings.embedding_size(without) < Embeddings.embedding_size(with_block)

    # a source that DOES provide them keeps the block
    keeps = Embeddings.config_for_source([with_projectiles: true], [:players, :projectiles])
    assert keeps.with_projectiles == true
  end

  test "the checkpoint config records provided_channels and the resolved with_projectiles" do
    json = Config.build_config_json(Config.defaults())
    assert json.provided_channels == Peppi.provides()
    assert json.with_projectiles == false
  end

  test "zero_projectiles?/2: old checkpoints zero, stamped-provided populate, no-block never zeroes, env overrides" do
    # old replay-trained checkpoint: block present, no stamp -> zero
    assert Agent.zero_projectiles?(%{with_projectiles: true}, nil)
    # new Peppi-trained checkpoint: no block -> nothing to zero
    refute Agent.zero_projectiles?(%{with_projectiles: false, provided_channels: [:players]}, nil)
    # a source that provided projectiles (json round-trip strings) -> populate
    refute Agent.zero_projectiles?(%{with_projectiles: "true", provided_channels: ["players", "projectiles"]}, nil)
    # env overrides both ways
    assert Agent.zero_projectiles?(%{with_projectiles: false}, "1")
    refute Agent.zero_projectiles?(%{with_projectiles: true}, "0")
  end
end
