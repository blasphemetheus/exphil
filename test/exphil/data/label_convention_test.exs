defmodule ExPhil.Data.LabelConventionTest do
  @moduledoc """
  INVARIANTS.md item 1 (structural form): the deploy-mapping law.

  One module owns how a checkpoint's delay numbers relate to the live
  `--frame-delay` flag. These cases are the deploy cards:

    * ms_g19 (legacy, trained {2,3}) plays local d3 with id 3;
    * fox_gen_v16e (legacy, `--frame-delay 1`) deploys at `--frame-delay 1`;
    * fox_gen_v1/v2 (legacy delay 0) are the leak;
    * a causal checkpoint at reaction delay 0 is the SAME target as v16e
      and deploys at `--frame-delay 1` with id 0.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Data.LabelConvention, as: LC
  alias ExPhil.Training.Comparability

  @v16e %{frame_delay: 1, action_delay: 0, train_delays: [1]}
  @v2 %{frame_delay: 0, action_delay: 0, train_delays: [0]}
  @ms_g19 %{frame_delay: 0, action_delay: 3, train_delays: [0, 2, 3]}
  @causal0 %{frame_delay: 0, action_delay: 0, train_delays: [0], label_convention: :causal}
  @causal_json %{"frame_delay" => 2, "train_delays" => [1, 2], "label_convention" => "causal"}

  test "unstamped configs are legacy; stamped (atom or JSON string) are causal" do
    assert LC.of(@v16e) == :producing
    assert LC.of(nil) == :producing
    assert LC.of(@causal0) == :causal
    assert LC.of(@causal_json) == :causal
    assert LC.current() == :causal
  end

  test "reaction delay: legacy d - 1, causal d; the legacy leak is -1" do
    assert LC.reaction_delay(@v16e) == 0
    assert LC.reaction_delay(@ms_g19) == 2
    assert LC.reaction_delay(@v2) == -1
    assert LC.leaky?(@v2)
    refute LC.leaky?(@v16e)
    assert LC.reaction_delay(@causal0) == 0
    assert LC.reaction_delay(@causal_json) == 2
    refute LC.leaky?(@causal0)
  end

  test "train_delays are normalized to reaction numbering" do
    assert LC.train_reaction_delays(@ms_g19) == [-1, 1, 2]
    assert LC.train_reaction_delays(@causal_json) == [1, 2]
    assert LC.train_reaction_delays(%{}) == nil
  end

  test "deploy law: a checkpoint trained at reaction k deploys at --frame-delay k+1" do
    assert LC.live_frame_delay(LC.reaction_delay(@v16e)) == 1
    assert LC.live_frame_delay(LC.reaction_delay(@causal0)) == 1
    assert LC.live_frame_delay(2) == 3
    assert LC.live_reaction_delay(3) == 2
    assert LC.live_reaction_delay(0) == -1
  end

  test "delay_id is in the checkpoint's own numbering" do
    # ms_g19 local d3 -> id 3 (the card); netplay d4 -> id 4 (untrained, hence the override)
    assert LC.delay_id(3, @ms_g19) == 3
    assert LC.delay_id(4, @ms_g19) == 4
    # v16e at its deploy rung -> id 1 == its train_delays
    assert LC.delay_id(1, @v16e) == 1
    # causal checkpoint at the same physical rung -> id 0 == its train_delays
    assert LC.delay_id(1, @causal0) == 0
    assert LC.delay_id(3, @causal_json) == 2
    # never negative
    assert LC.delay_id(0, @causal0) == 0
  end

  test "a legacy delay-1 checkpoint and a causal delay-0 checkpoint are the SAME target" do
    assert Comparability.key(@v16e).label_delay == 0
    assert Comparability.key(@causal0).label_delay == 0
    assert Comparability.key(@v16e).train_delays == Comparability.key(@causal0).train_delays
    assert Comparability.key(@v2).label_delay == -1
  end
end
