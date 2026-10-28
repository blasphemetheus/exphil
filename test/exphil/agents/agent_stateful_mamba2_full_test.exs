defmodule ExPhil.Agents.AgentStatefulMamba2FullTest do
  use ExUnit.Case, async: false
  import Nx.Testing
  alias ExPhil.Agents.Agent
  alias ExPhil.Networks.Policy

  test "canonical checkpoint reload, stepping, replay, rollback and row reset" do
    embed = ExPhil.Embeddings.Game.embedding_size(ExPhil.Embeddings.Game.default_config())

    config = %{
      temporal: true,
      backbone: :mamba2_full,
      head: :independent,
      embed_size: embed,
      hidden_size: 16,
      num_layers: 2,
      head_dim: 4,
      state_size: 3,
      num_groups: 2,
      conv_size: 3,
      expand_factor: 2,
      window_size: 5,
      dropout: 0.0,
      axis_buckets: 16,
      shoulder_buckets: 4
    }

    model = Policy.build_temporal(Map.to_list(config))
    {init, _} = Axon.build(model)
    params = init.(Nx.broadcast(0.0, {1, 5, embed}), Axon.ModelState.empty())
    policy = %{params: params, config: config}
    # Exercise the serialized parameter/config boundary used by exported policies.
    policy = policy |> :erlang.term_to_binary() |> :erlang.binary_to_term()

    {:ok, agent} =
      Agent.start_link(
        policy: policy,
        deterministic: true,
        stateful_step: true,
        stateful_resync: 2,
        stateful_resync_stagger: true
      )

    on_exit(fn -> if Process.alive?(agent), do: GenServer.stop(agent) end)
    s = :sys.get_state(agent)
    assert Nx.shape(s.trunk_state.h) == {1, 2, 8, 4, 3}
    assert Nx.shape(s.trunk_state.conv) == {1, 2, 2, 44}
    assert Map.has_key?(s.trunk_step_params, "final_norm")
    assert Map.has_key?(s.trunk_step_params, "mamba2_2_norm")
    x = Nx.sin(Nx.iota({1, 5, embed}))

    backbone =
      ExPhil.Networks.Policy.Backbone.build_temporal_backbone(
        embed,
        :mamba2_full,
        Map.to_list(config)
      )

    {_, predict} = Axon.build(backbone)
    expected = predict.(%{params | data: s.trunk_step_params}, x)

    {actual, _} =
      Enum.reduce(0..4, {nil, s.trunk_state}, fn t, {_, state} ->
        Edifice.SSM.Mamba2.step(s.trunk_step_params, state, Nx.take(x, t, axis: 1))
      end)

    assert_all_close(actual, expected, atol: 1.0e-5, rtol: 1.0e-4)
    for i <- 0..6, do: assert({:ok, _} = Agent.get_action(agent, frame(i)))
    {:ok, rollback} = Agent.snapshot_state(agent)
    actions = for i <- 7..10, do: action(agent, i)
    :ok = Agent.restore_state(agent, rollback)
    assert for(i <- 7..10, do: action(agent, i)) == actions
    :ok = Agent.batch_init(agent, 2)
    assert :sys.get_state(agent).batch.mode == :stateful

    for i <- 0..8,
        do: assert({:ok, _} = Agent.batch_get_controllers(agent, [frame(i), frame(-i)]))

    before = :sys.get_state(agent).batch.trunk_state
    {:ok, snapshot} = Agent.snapshot_state(agent)
    :ok = Agent.batch_reset_rows(agent, [0])

    for key <- [:h, :conv] do
      reset = :sys.get_state(agent).batch.trunk_state[key]
      assert_all_close(Nx.take(reset, 0), Nx.broadcast(0.0, Nx.shape(Nx.take(reset, 0))))
      assert_all_close(Nx.take(reset, 1), Nx.take(before[key], 1))
    end

    :ok = Agent.restore_state(agent, snapshot)
    assert Agent.snapshot_state(agent) == {:ok, snapshot}
  end

  defp action(agent, i) do
    {:ok, result} = Agent.get_action(agent, frame(i))

    Map.new([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder], fn key ->
      {key, Nx.to_flat_list(result[key])}
    end)
  end

  defp frame(i) do
    player = %ExPhil.Bridge.Player{
      character: 25,
      x: i * 1.0,
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

    %ExPhil.Bridge.GameState{
      frame: abs(i),
      stage: 32,
      menu_state: 2,
      players: %{1 => player, 2 => %{player | x: -player.x}},
      projectiles: [],
      distance: abs(2 * player.x)
    }
  end

  test "training config keeps canonical and legacy architectures distinct" do
    alias ExPhil.Training.Config
    assert :mamba2_full in Config.valid_backbones()
    assert {Edifice.SSM.Mamba2, :embed_dim, _} = Config.backbone_recipe(:mamba2_full)
    assert Config.backbone_defaults(:mamba2_full)[:state_size] == 128
    assert Config.backbone_defaults(:mamba_2)[:state_size] == 16
    opts = Config.parse_args(["--backbone", "mamba2_full", "--num-groups", "2"])
    assert opts[:num_groups] == 2
  end

  test "imitation training updates canonical weights and exports the group count" do
    alias ExPhil.Training.Imitation

    trainer =
      Imitation.new(
        temporal: true,
        backbone: :mamba2_full,
        hidden_size: 8,
        hidden_sizes: [8],
        num_layers: 1,
        state_size: 3,
        head_dim: 2,
        num_groups: 2,
        conv_size: 3,
        window_size: 4,
        dropout: 0.0,
        head: :independent,
        precision: :f32,
        event_context: false,
        stick_release: false,
        learning_rate: 0.01,
        lr_schedule: :constant,
        warmup_steps: 0
      )

    batch = %{
      states: Nx.sin(Nx.iota({2, 4, trainer.config.embed_size})),
      actions: %{
        buttons: Nx.broadcast(0, {2, 8}),
        main_x: Nx.broadcast(8, {2}),
        main_y: Nx.broadcast(8, {2}),
        c_x: Nx.broadcast(8, {2}),
        c_y: Nx.broadcast(8, {2}),
        shoulder: Nx.broadcast(2, {2})
      }
    }

    {trained, _} = Imitation.train_step(trainer, batch, nil)
    assert trained.step == 1

    refute Nx.to_binary(trained.policy_params.data["mamba2_1"]["A_log"]) ==
             Nx.to_binary(trainer.policy_params.data["mamba2_1"]["A_log"])

    path = Path.join(System.tmp_dir!(), "mamba2_full_#{System.unique_integer([:positive])}.bin")
    on_exit(fn -> File.rm(path) end)
    assert :ok = Imitation.export_policy(trained, path)
    {:ok, exported} = ExPhil.Training.Checkpoint.load_policy(path)
    assert exported.config.backbone == :mamba2_full
    assert exported.config.num_groups == 2
    assert exported.config.state_size == 3
    {:ok, agent} = Agent.start_link(policy: exported, deterministic: true, stateful_step: true)
    assert Nx.shape(:sys.get_state(agent).trunk_state.h) == {1, 1, 8, 2, 3}
    GenServer.stop(agent)
  end
end
