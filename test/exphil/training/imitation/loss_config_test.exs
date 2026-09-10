defmodule ExPhil.Training.Imitation.LossConfigTest do
  @moduledoc "INVARIANTS.md item 8: loss types, not loss knobs."
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config
  alias ExPhil.Training.Imitation.LossConfig

  test "absent keys fall back to Config.defaults (one source); present nil/false are honored" do
    lc = LossConfig.from_config(%{})
    assert lc.precision == Config.defaults()[:precision]
    assert lc.head == Config.defaults()[:head]

    lc2 = LossConfig.from_config(%{stick_edge_weight: nil, focal_loss: false, head: :autoregressive, precision: :bf16})
    assert lc2.stick_edge_weight == nil
    assert lc2.button.focal == nil
    assert lc2.head == :autoregressive and lc2.precision == :bf16
  end

  test "button loss has NO smoothing field (the July 2026 poison is unrepresentable)" do
    lc = LossConfig.from_config(%{label_smoothing: 0.1, button_pos_weight: [1, 1, 1, 1, 1, 1, 1, 1], focal_loss: true, focal_gamma: 3.0})
    refute Map.has_key?(lc.button, :smoothing)
    assert lc.label_smoothing == 0.1
    assert %Nx.Tensor{} = lc.button.pos_weight
    assert lc.button.focal == %{gamma: 3.0}
  end

  test "to_loss_opts is the only exit and round-trips every knob" do
    cfg = %{label_smoothing: 0.0, focal_loss: true, focal_gamma: 3.0, button_weight: 2.0, button_pos_weight: :auto,
            stick_edge_weight: 1.0, entropy_weight: 0.0, head_normalize: false}
    opts = cfg |> LossConfig.from_config() |> LossConfig.to_loss_opts()
    assert opts[:focal_loss] == true and opts[:focal_gamma] == 3.0
    assert opts[:button_weight] == 2.0 and opts[:button_pos_weight] == nil
    assert Keyword.keys(opts) |> Enum.sort() ==
             Enum.sort([:label_smoothing, :focal_loss, :focal_gamma, :button_weight, :button_pos_weight, :stick_edge_weight, :entropy_weight, :head_normalize])
  end

  test "no builder in loss.ex extracts knobs inline any more (structural tripwire)" do
    src = File.read!("lib/exphil/training/imitation/loss.ex")
    refute src =~ ~r/config\[:(label_smoothing|focal_loss|focal_gamma|button_weight|button_pos_weight|stick_edge_weight|entropy_weight|head_normalize|precision|head)\]/
    refute src =~ ~r/Keyword\.get\(opts, :(label_smoothing|focal_loss|focal_gamma|button_weight)/
  end
end
