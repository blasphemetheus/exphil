defmodule ExPhil.Agents.AgentStatefulMambaTest do
  @moduledoc """
  Carried-state Mamba inference (2026-10-01): `stateful_step: true` routes a
  temporal Mamba policy through `Edifice.SSM.Mamba.step/3` (SSM state + conv
  ring buffer per layer) instead of re-running the whole window each frame.
  Synthetic GameStates, no Dolphin. The batched path (autoregressive head only)
  is covered end to end by scripts/sim_closed_loop.exs --stateful-step.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Utils

  @hidden_size 16
  @num_layers 2
  @window 8

  defp embed_size do
    ExPhil.Embeddings.Game.embedding_size(ExPhil.Embeddings.Game.default_config())
  end

  defp build_policy do
    model =
      Policy.build_temporal(
        embed_size: embed_size(),
        backbone: :mamba,
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        window_size: @window,
        dropout: 0.0,
        axis_buckets: 16,
        shoulder_buckets: 4
      )

    {init_fn, _predict_fn} = Utils.build_compiled(model)
    params = init_fn.(Nx.template({1, @window, embed_size()}, :f32), Axon.ModelState.empty())

    %{
      params: params,
      config: %{
        temporal: true,
        backbone: :mamba,
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
    {:ok, agent} = Agent.start_link([policy: policy, deterministic: true] ++ opts)
    agent
  end

  # Categorical heads only: random-init button logits sit at ~0, where fp
  # noise between the two compilation paths can flip the 0.5 threshold.
  defp sig(action) do
    for head <- [:main_x, :main_y, :c_x, :c_y, :shoulder], do: Nx.to_flat_list(action[head])
  end

  defp play(agent, xs) do
    for x <- xs do
      {:ok, action} = Agent.get_action(agent, game_state(x))
      sig(action)
    end
  end

  setup_all do
    {:ok, policy: build_policy()}
  end

  test "stateful_step activates for a temporal Mamba policy", %{policy: policy} do
    agent = start_agent(policy, stateful_step: true)
    config = Agent.get_config(agent)

    assert config.backbone == :mamba
    assert config.stateful_step_active
    assert [[_, _, _, _, _]] = play(agent, [10.0])

    # one SSM hidden state + one conv buffer per layer, batch 1
    trunk_state = :sys.get_state(agent).trunk_state
    assert map_size(trunk_state) > 0
    assert Enum.all?(trunk_state, fn {_k, v} -> elem(Nx.shape(v), 0) == 1 end)

    GenServer.stop(agent)
  end

  test "first frame matches the windowed path (cold-start warmup pin)", %{policy: policy} do
    windowed = start_agent(policy, [])
    stateful = start_agent(policy, stateful_step: true)

    assert play(windowed, [17.0]) == play(stateful, [17.0])

    GenServer.stop(windowed)
    GenServer.stop(stateful)
  end

  test "reset_buffer restores fresh-game behavior", %{policy: policy} do
    xs = [4.0, -7.0, 9.0]

    fresh = start_agent(policy, stateful_step: true)
    fresh_actions = play(fresh, xs)
    GenServer.stop(fresh)

    agent = start_agent(policy, stateful_step: true)
    _pollute = play(agent, [30.0, -30.0, 15.0, 2.0])
    :ok = Agent.reset_buffer(agent)
    assert play(agent, xs) == fresh_actions

    GenServer.stop(agent)
  end
end
