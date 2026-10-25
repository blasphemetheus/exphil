defmodule ExPhil.Agents.AgentStatefulMinGRUTest do
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

  defp build_policy(head \\ :independent, backbone \\ :min_gru, window \\ @window) do
    # Mirror the model the Agent rebuilds in load_policy_internal (same
    # trunk_opts semantics) so init params load cleanly.
    model =
      Policy.build_temporal(
        embed_size: embed_size(),
        backbone: backbone,
        head: head,
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        window_size: window,
        dropout: 0.0,
        axis_buckets: 16,
        shoulder_buckets: 4
      )

    {init_fn, _predict_fn} = Utils.build_compiled(model, seed: 42)

    inputs = Nx.template({1, window, embed_size()}, :f32)

    inputs =
      if head == :autoregressive,
        do: Map.put(Policy.Heads.tf_templates(1), "state_sequence", inputs),
        else: inputs

    params = init_fn.(inputs, Axon.ModelState.empty())

    %{
      params: params,
      config: %{
        temporal: true,
        backbone: backbone,
        head: head,
        window_size: window,
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
      Agent.start_link([policy: policy, deterministic: true] ++ opts)

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
      # Same-frame polling returns a cached action; advance the game clock so
      # recurrence, resync and rollback tests actually execute new steps.
      frame = (:sys.get_state(agent).last_debounce_frame || -1) + 1
      {:ok, action} = Agent.get_action(agent, %{game_state(x) | frame: frame})
      action_sig(action)
    end
  end

  setup_all do
    {:ok, policy: build_policy()}
  end

  test "explicit sampling keys reproduce stochastic batched rollouts after reset" do
    policy = build_policy(:autoregressive)
    {:ok, agent} = Agent.start_link(policy: policy, deterministic: false, stateful_step: true)
    :ok = Agent.batch_init(agent, 2)

    rollout = fn ->
      for tick <- 1..5 do
        {:ok, controllers} =
          Agent.batch_get_controllers(
            agent,
            [game_state(tick * 1.0), game_state(tick * -2.0)],
            key: Nx.Random.key(tick)
          )

        controllers
      end
    end

    expected = rollout.()
    :ok = Agent.batch_reset_rows(agent, [0, 1])
    assert rollout.() == expected
    GenServer.stop(agent)
  end

  for backbone <- [:min_gru, :gru, :lstm, :mamba, :mamba_2], window <- [1, @window] do
    test "#{backbone}, window #{window}: batched resync matches independent agents across staggered resets and observations" do
      policy = build_policy(:autoregressive, unquote(backbone), unquote(window))
      batch = start_agent(policy, stateful_step: true, stateful_resync: 2)
      singles = for _ <- 1..2, do: start_agent(policy, stateful_step: true, stateful_resync: 2)
      :ok = Agent.batch_init(batch, 2)

      for tick <- 0..23 do
        if tick in [18, 20] do
          interval = if tick == 18, do: nil, else: 2

          for agent <- [batch | singles],
              do: :ok = Agent.reconfigure(agent, stateful_resync: interval)
        end

        if tick in [10, 15] do
          row = if tick == 10, do: 0, else: 1
          :ok = Agent.batch_reset_rows(batch, [row])
          :ok = Agent.reset_buffer(Enum.at(singles, row))
        end

        frames = for row <- 0..1, do: %{game_state((tick * (row + 1) - 13) * 1.0) | frame: tick}

        if tick in [3, 4, 8, 17] do
          assert :ok = Agent.batch_observe(batch, frames)

          for {single, frame} <- Enum.zip(singles, frames),
              do: assert(:ok = Agent.observe(single, frame, nil))
        else
          assert {:ok, _} = Agent.batch_get_controllers(batch, frames)

          for {single, frame} <- Enum.zip(singles, frames),
              do: assert({:ok, _} = Agent.get_action(single, frame))
        end

        b = :sys.get_state(batch).batch

        for {single, row} <- Enum.with_index(singles) do
          s = :sys.get_state(single)
          assert Enum.at(b.resync_steps, row) == s.steps_since_resync

          for {key, value} <- s.trunk_state do
            assert Nx.to_number(
                     Nx.reduce_max(
                       Nx.abs(
                         Nx.subtract(
                           Nx.slice_along_axis(b.trunk_state[key], row, 1, axis: 0),
                           value
                         )
                       )
                     )
                   ) < 1.0e-4,
                   "carry diverged at tick #{tick}, row #{row}, key #{key}"

            Nx.Testing.assert_all_close(
              Nx.slice_along_axis(b.trunk_state[key], row, 1, axis: 0),
              value,
              atol: 1.0e-5,
              rtol: 1.0e-5
            )
          end
        end
      end

      for agent <- [batch | singles], do: GenServer.stop(agent)
    end
  end

  for backbone <- [:min_gru, :gru, :mamba, :mamba_2, :mamba_ssd] do
    test "#{backbone}: a reset batch row gets cold padding without advancing other rows" do
      policy = build_policy(:autoregressive, unquote(backbone))
      agents = for _ <- 1..3, do: start_agent(policy, stateful_step: true)
      [actual, continuing, _fresh] = agents
      for agent <- agents, do: :ok = Agent.batch_init(agent, 2)

      for x <- [9.0, -3.0, 25.0], agent <- [actual, continuing] do
        assert {:ok, _} = Agent.batch_get_controllers(agent, [game_state(x), game_state(-x)])
      end

      :ok = Agent.batch_reset_rows(actual, [0])
      inputs = [game_state(4.0), game_state(-17.0)]

      outputs =
        for agent <- agents do
          assert {:ok, _, features} =
                   Agent.batch_get_controllers(agent, inputs, return_features: true)

          features
        end

      [a, c, f] = outputs
      Nx.Testing.assert_all_close(Nx.take(a, 0), Nx.take(f, 0), atol: 1.0e-5, rtol: 1.0e-5)
      Nx.Testing.assert_all_close(Nx.take(a, 1), Nx.take(c, 1), atol: 1.0e-5, rtol: 1.0e-5)
      assert MapSet.size(:sys.get_state(actual).batch.cold_rows) == 0
      for agent <- agents, do: GenServer.stop(agent)
    end
  end

  for backbone <- [:min_gru, :mamba, :mamba_2, :mamba_ssd] do
    test "#{backbone}: compiled cold start and resync preserve window state and rollback" do
      backbone = unquote(backbone)
      policy = build_policy(:autoregressive, backbone)
      agent = start_agent(policy, stateful_step: true, stateful_resync: 2)
      windowed = start_agent(policy, [])
      assert Agent.get_config(agent).stateful_step_active
      {:ok, before_warmup} = Agent.snapshot_state(agent)
      {:ok, _milliseconds} = Agent.warmup(agent)
      assert Agent.snapshot_state(agent) == {:ok, before_warmup}
      assert play(agent, [12.0]) == play(windowed, [12.0])

      for x <- [5.0, -7.0, 9.0, 20.0, -1.0, 3.0] do
        play(agent, [x])
        play(windowed, [x])
      end

      before = :sys.get_state(agent)
      assert before.steps_since_resync >= 2
      assert play(agent, [18.0]) == play(windowed, [18.0])
      after_resync = :sys.get_state(agent)
      assert after_resync.steps_since_resync == 0

      module =
        case backbone do
          :min_gru -> Edifice.Recurrent.MinGRU
          :mamba -> Edifice.SSM.Mamba
          _ -> Edifice.SSM.MambaSSD
        end

      initial =
        module.init_state(after_resync.trunk_step_params,
          batch_size: 1,
          hidden_size: @hidden_size,
          num_layers: @num_layers
        )

      frames = Enum.reverse(before.step_frame_buffer) ++ [hd(after_resync.step_frame_buffer)]
      step = Edifice.Stateful.jit_step(module, EXLA)

      expected =
        Enum.reduce(frames, initial, fn frame, s ->
          elem(step.(after_resync.trunk_step_params, s, frame), 1)
        end)

      for {key, tensor} <- expected do
        Nx.Testing.assert_all_close(after_resync.trunk_state[key], tensor,
          atol: 1.0e-5,
          rtol: 1.0e-5
        )
      end

      {:ok, snapshot} = Agent.snapshot_state(agent)
      actions = play(agent, [1.0, 8.0, -4.0, 2.0])
      {:ok, final_snapshot} = Agent.snapshot_state(agent)
      :ok = Agent.restore_state(agent, snapshot)
      assert play(agent, [1.0, 8.0, -4.0, 2.0]) == actions
      assert Agent.snapshot_state(agent) == {:ok, final_snapshot}
      :ok = Agent.reset_buffer(agent)
      :ok = Agent.reset_buffer(windowed)
      assert play(agent, [6.0]) == play(windowed, [6.0])
      GenServer.stop(agent)
      GenServer.stop(windowed)
    end
  end

  test "autoregressive head, batched state, row reset and rollback use the MinGRU adapter" do
    policy = build_policy(:autoregressive)
    agent = start_agent(policy, stateful_step: true)
    windowed = start_agent(policy, [])
    assert play(agent, [17.0]) == play(windowed, [17.0])
    {:ok, snapshot} = Agent.snapshot_state(agent)
    first = play(agent, [3.0, -4.0])
    :ok = Agent.restore_state(agent, snapshot)
    assert play(agent, [3.0, -4.0]) == first

    :ok = Agent.batch_init(agent, 2)
    assert :sys.get_state(agent).batch.mode == :stateful
    assert {:ok, _} = Agent.batch_get_controllers(agent, [game_state(5.0), game_state(-7.0)])
    before = :sys.get_state(agent).batch.trunk_state.h
    assert Nx.shape(before) == {2, @num_layers, @hidden_size}
    :ok = Agent.batch_reset_rows(agent, [0])
    after_reset = :sys.get_state(agent).batch.trunk_state.h
    assert Nx.to_number(Nx.all(Nx.equal(Nx.take(after_reset, 0), 0))) == 1
    assert Nx.to_binary(Nx.take(after_reset, 1)) == Nx.to_binary(Nx.take(before, 1))
    GenServer.stop(agent)
    GenServer.stop(windowed)
  end

  test "legacy Mamba-2 convolution history follows exported weights" do
    policy = build_policy(:autoregressive, :mamba_2)
    policy = %{policy | config: Map.put(policy.config, :conv_size, 1)}
    agent = start_agent(policy, stateful_step: true)
    windowed = start_agent(policy, [])
    assert Nx.axis_size(:sys.get_state(agent).trunk_state.conv, 2) == 3
    assert play(agent, [12.0]) == play(windowed, [12.0])
    GenServer.stop(agent)
    GenServer.stop(windowed)
  end

  test "stateful_step activates for a temporal MinGRU policy", %{policy: policy} do
    agent = start_agent(policy, stateful_step: true)
    config = Agent.get_config(agent)

    assert config.temporal
    assert config.backbone == :min_gru
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
