defmodule ExPhil.Eval.HarnessRungTest do
  @moduledoc """
  INVARIANTS.md item 12: every harness's decision->application latency is
  declared in ONE table, and delay-ids are derived from it. The cases are
  the measured calibrations (each cited in the module doc); a change to the
  table must re-measure, not re-type.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Eval.HarnessRung, as: HR

  # ms_g23a_ep57 as stamped (drill line, causal, --pipeline-offset 2 NOT stamped)
  @ep57 %{train_delays: [0, 1, 2, 3], label_convention: :causal, queue_depth: 4, with_delay_id: true}
  # the same checkpoint as the drill stamps it from 2026-09-12
  @ep57_stamped Map.put(@ep57, :delay_id_reaction_offset, 2)
  # ms_g19 (legacy numbering, trained {2,3} == reaction {1,2} nominal)
  @ms_g19 %{frame_delay: 0, action_delay: 3, train_delays: [0, 2, 3]}
  # fox_gen_v16e (legacy delay 1 == reaction 0, not delay-conditioned)
  @v16e %{frame_delay: 1, action_delay: 0, train_delays: [1]}
  # a v3-style causal checkpoint at reaction 0
  @causal0 %{frame_delay: 0, action_delay: 0, train_delays: [0], label_convention: :causal}

  test "latency = knob + the harness's pipeline; training reaction k is latency k+1" do
    assert HR.latency(:training, 0) == 1
    assert HR.latency(:sync_runner, 3) == 5
    assert HR.latency(:async_runner, 3) == 5
    assert HR.latency(:scenario_suite, 2) == 3
  end

  test "09-12 sync pin: sync and async share the 2-frame pipeline (fd3/id2 427/436, fd2/id2 2, fd4/id2 3/106)" do
    assert HR.latency(:sync_runner, 3) == HR.latency(:async_runner, 3)
    assert HR.latency(:scenario_suite, 4) == HR.latency(:sync_runner, 3)
    assert HR.delay_id(:sync_runner, 3, @ep57) == 2
    assert HR.delay_id(:sync_runner, 2, @ep57) == 1
    assert HR.aligned_knob(:sync_runner, 2, @ep57) == {:ok, 3}
  end

  test "09-12 suite grid: response-delay 2 <-> id 0, 3 <-> id 1 (6/6 == teacher)" do
    assert HR.delay_id(:scenario_suite, 2, @ep57) == 0
    assert HR.delay_id(:scenario_suite, 3, @ep57) == 1
    assert HR.aligned_knob(:scenario_suite, 0, @ep57) == {:ok, 2}
    assert HR.aligned_knob(:scenario_suite, 2, @ep57) == {:ok, 4}
    # the suite's native latency (rd 0) is BELOW every trained rung
    assert HR.delay_id(:scenario_suite, 0, @ep57) == 0
    assert HR.reaction_delay(:scenario_suite, 0) == 0
  end

  test "rung law: async --frame-delay 3 <-> drill id 2 (== ms_g19's legacy id 3)" do
    assert HR.delay_id(:async_runner, 3, @ep57) == 2
    assert HR.delay_id(:async_runner, 3, @ms_g19) == 3
    # netplay d4: legacy id 4 (untrained -> the --delay-id-override 3 card)
    assert HR.delay_id(:async_runner, 4, @ms_g19) == 4
    assert HR.aligned_knob(:async_runner, 2, @ep57) == {:ok, 3}
  end

  test "the stamp wins; unstamped delay-conditioned checkpoints assume the drill's 2; others 0" do
    assert HR.delay_id_reaction_offset(@ep57_stamped) == {2, :stamped}
    assert HR.delay_id_reaction_offset(@ep57) == {2, :assumed_drill}
    assert HR.delay_id_reaction_offset(@v16e) == {0, :none}
    assert HR.delay_id_reaction_offset(%{"delay_id_reaction_offset" => "1", "train_delays" => [0, 1]}) == {1, :stamped}
    # a future drill run at --pipeline-offset 0 maps differently, by its stamp
    assert HR.delay_id(:async_runner, 1, Map.put(@ep57, :delay_id_reaction_offset, 0)) == 2
  end

  test "non-conditioned checkpoints: the exact deploy knob per harness, unreachable is loud" do
    # reaction 0 (v16e / causal0): NEITHER runner can go that fast (floor latency 2 = reaction 1);
    # --frame-delay 0 is the nearest (one slower), the 09-09 deploy card (--frame-delay 1) is TWO slower
    assert HR.deploy_knob(:sync_runner, @v16e) == {:error, {:unreachable, 2}}
    assert HR.deploy_knob(:sync_runner, @causal0) == {:error, {:unreachable, 2}}
    assert HR.deploy_knob(:async_runner, @causal0) == {:error, {:unreachable, 2}}
    assert HR.deploy_knob(:scenario_suite, @causal0) == {:ok, 0}
    # reaction 2 -> --frame-delay 1 on either runner; reaction 1 -> 0
    assert HR.deploy_knob(:async_runner, %{label_delay: 2, label_convention: :causal}) == {:ok, 1}
    assert HR.deploy_knob(:sync_runner, %{label_delay: 1, label_convention: :causal}) == {:ok, 0}
  end

  test "delay_id never goes negative" do
    assert HR.delay_id(:scenario_suite, 0, @ep57) == 0
    assert HR.delay_id(:sync_runner, 0, @ms_g19) == 0
  end

  test "describe/3 is one readable line" do
    line = HR.describe(:async_runner, 3, @ep57)
    assert line =~ "latency 5"
    assert line =~ "delay-id 2"
    assert line =~ "assumed_drill"
  end
end
