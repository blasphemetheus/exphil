defmodule ExPhil.Agents.AgentStatefulStepTest do
  @moduledoc """
  GenServer-level tests for the Agent's Edifice.Stateful step path (task #16):
  activation via `stateful_step: true`, the snapshot/restore rollback API
  (for #9 Yeti), and reset semantics. No Dolphin needed — synthetic
  GameStates drive `get_action/3` directly.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Utils

  @hidden_size 16
  @num_layers 1
  @window 8

  defp embed_size do
    ExPhil.Embeddings.Game.embedding_size(ExPhil.Embeddings.Game.default_config())
  end

  defp build_policy do
    # Mirror the model the Agent rebuilds in load_policy_internal (same
    # trunk_opts semantics) so init params load cleanly.
    model =
      Policy.build_temporal(
        embed_size: embed_size(),
        backbone: :gru,
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        window_size: @window,
        dropout: 0.0,
        axis_buckets: 16,
        shoulder_buckets: 4
      )

    {init_fn, _predict_fn} = Utils.build_compiled(model)

    params =
      init_fn.(
        Nx.template({1, @window, embed_size()}, :f32),
        Axon.ModelState.empty()
      )

    %{
      params: params,
      config: %{
        temporal: true,
        backbone: :gru,
        window_size: @window,
        embed_size: embed_size(),
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        axis_buckets: 16,
        shoulder_buckets: 4,
        dropout: 0.0
      }
    }
  end

  defp game_state(x) do
    player = %Player{
      character: 25,
      x: x,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 0,
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

    %GameState{
      frame: 0,
      stage: 32,
      menu_state: 2,
      players: %{1 => player, 2 => %{player | x: -x}},
      projectiles: [],
      distance: abs(2 * x)
    }
  end

  defp start_agent(policy, opts) do
    {:ok, agent} =
      Agent.start_link(
        [policy: policy, deterministic: true] ++ opts
      )

    agent
  end

  # Comparable, backend-independent fingerprint of an action
  defp action_sig(action) do
    %{
      buttons: Nx.to_flat_list(action.buttons),
      main_x: action.main_x |> Nx.to_flat_list(),
      main_y: action.main_y |> Nx.to_flat_list(),
      c_x: action.c_x |> Nx.to_flat_list(),
      c_y: action.c_y |> Nx.to_flat_list(),
      shoulder: action.shoulder |> Nx.to_flat_list()
    }
  end

  defp play(agent, xs) do
    for x <- xs do
      {:ok, action} = Agent.get_action(agent, game_state(x))
      action_sig(action)
    end
  end

  setup_all do
    {:ok, policy: build_policy()}
  end

  test "stateful_step activates for a temporal GRU policy", %{policy: policy} do
    agent = start_agent(policy, stateful_step: true)
    config = Agent.get_config(agent)

    assert config.temporal
    assert config.backbone == :gru
    assert config.stateful_step
    assert config.stateful_step_active

    # And it actually produces well-formed actions
    [sig] = play(agent, [10.0])
    assert length(sig.buttons) == 8
    assert length(sig.main_x) == 1

    GenServer.stop(agent)
  end

  test "windowed mode leaves the step path inactive", %{policy: policy} do
    agent = start_agent(policy, [])
    config = Agent.get_config(agent)

    refute config.stateful_step
    refute config.stateful_step_active
    assert {:error, :not_stateful} = Agent.snapshot_state(agent)
    assert {:error, :not_stateful} = Agent.restore_state(agent, <<131, 106>>)

    GenServer.stop(agent)
  end

  test "snapshot/restore round-trip replays identically (rollback pin)", %{policy: policy} do
    agent = start_agent(policy, stateful_step: true)

    # Advance a few frames, then snapshot at frame k
    _warm = play(agent, [0.0, 5.0, -3.0, 12.0])
    {:ok, blob} = Agent.snapshot_state(agent)
    assert is_binary(blob)

    # Timeline A: keep playing from the snapshot point
    tail_xs = [20.0, -8.0, 1.5]
    timeline_a = play(agent, tail_xs)

    # Rollback: restore frame-k state, replay the SAME frames
    :ok = Agent.restore_state(agent, blob)
    timeline_b = play(agent, tail_xs)

    assert timeline_a == timeline_b

    GenServer.stop(agent)
  end

  test "restore_state rejects garbage blobs", %{policy: policy} do
    agent = start_agent(policy, stateful_step: true)
    _warm = play(agent, [1.0])

    assert {:error, {:invalid_snapshot, _}} =
             Agent.restore_state(agent, :erlang.term_to_binary(%{version: 1}))

    # Agent still works after the failed restore
    assert [_] = play(agent, [2.0])

    GenServer.stop(agent)
  end

  test "reset_buffer restores fresh-game behavior", %{policy: policy} do
    xs = [4.0, -7.0, 9.0]

    fresh = start_agent(policy, stateful_step: true)
    fresh_actions = play(fresh, xs)
    GenServer.stop(fresh)

    agent = start_agent(policy, stateful_step: true)
    _pollute = play(agent, [30.0, -30.0, 15.0, 2.0])
    :ok = Agent.reset_buffer(agent)
    reset_actions = play(agent, xs)
    GenServer.stop(agent)

    assert fresh_actions == reset_actions
  end

  test "first frame matches the windowed path (cold-start warmup pin)", %{policy: policy} do
    # The windowed path pads its buffer with copies of the first frame; the
    # step path replicates that by stepping the first frame window_size-1
    # extra times when cold. Frame-1 outputs must therefore agree. Compare
    # only the categorical heads — random-init button logits sit at ~0
    # (sigmoid ~0.5), where fp noise between the two compilation paths could
    # legitimately flip the 0.5 threshold.
    gs = game_state(17.0)

    windowed = start_agent(policy, [])
    {:ok, windowed_action} = Agent.get_action(windowed, gs)
    GenServer.stop(windowed)

    stateful = start_agent(policy, stateful_step: true)
    {:ok, stateful_action} = Agent.get_action(stateful, gs)
    GenServer.stop(stateful)

    for head <- [:main_x, :main_y, :c_x, :c_y, :shoulder] do
      assert Nx.to_flat_list(windowed_action[head]) == Nx.to_flat_list(stateful_action[head]),
             "#{head} diverges between windowed and stateful first frame"
    end
  end

  test "stateful_step falls back to windowed for non-recurrent backbones" do
    # MLP temporal policy: step path must warn + stay inactive, not crash
    model =
      Policy.build_temporal(
        embed_size: embed_size(),
        backbone: :mlp,
        hidden_size: @hidden_size,
        window_size: @window,
        dropout: 0.0
      )

    {init_fn, _} = Utils.build_compiled(model)

    params =
      init_fn.(Nx.template({1, @window, embed_size()}, :f32), Axon.ModelState.empty())

    policy = %{
      params: params,
      config: %{
        temporal: true,
        backbone: :mlp,
        window_size: @window,
        embed_size: embed_size(),
        hidden_size: @hidden_size,
        axis_buckets: 16,
        shoulder_buckets: 4
      }
    }

    agent = start_agent(policy, stateful_step: true)
    config = Agent.get_config(agent)

    assert config.stateful_step
    refute config.stateful_step_active
    assert [_] = play(agent, [3.0])

    GenServer.stop(agent)
  end
end
