defmodule ExPhil.Agents.AgentObserveTest do
  @moduledoc """
  Observe-only history warm-up (`Agent.observe/4`, closed-loop correction
  validation 2026-09-12). The pin: a decision after OBSERVING a history must
  equal the decision after PLAYING that history — on both the windowed path
  and the stateful-step path — because the agent's history state is what
  decides, not how it got there. Synthetic GRU policy, no Dolphin.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.{ControllerState, GameState, Player}
  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Utils

  @hidden_size 16
  @num_layers 1
  @window 8

  defp embed_size do
    ExPhil.Embeddings.Game.embedding_size(ExPhil.Embeddings.Game.default_config())
  end

  defp build_policy do
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
      init_fn.(Nx.template({1, @window, embed_size()}, :f32), Axon.ModelState.empty())

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

  defp game_state(frame, x) do
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
      frame: frame,
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

  defp action_sig(action) do
    %{
      buttons: Nx.to_flat_list(action.buttons),
      main_x: Nx.to_flat_list(action.main_x),
      main_y: Nx.to_flat_list(action.main_y),
      c_x: Nx.to_flat_list(action.c_x),
      c_y: Nx.to_flat_list(action.c_y),
      shoulder: Nx.to_flat_list(action.shoulder)
    }
  end

  # A 12-frame history (longer than the window) plus the decision frame
  @history Enum.map(0..11, fn i -> {i, 3.0 * i - 10.0} end)
  @decision {12, 7.5}

  defp decide_after_playing(agent) do
    for {f, x} <- @history, do: {:ok, _} = Agent.get_action(agent, game_state(f, x))
    {f, x} = @decision
    {:ok, action} = Agent.get_action(agent, game_state(f, x))
    action_sig(action)
  end

  defp decide_after_observing(agent) do
    for {f, x} <- @history do
      :ok = Agent.observe(agent, game_state(f, x), ControllerState.neutral(), player_port: 1)
    end

    {f, x} = @decision
    {:ok, action} = Agent.get_action(agent, game_state(f, x))
    action_sig(action)
  end

  setup_all do
    {:ok, policy: build_policy()}
  end

  test "windowed path: observe-then-decide == play-then-decide", %{policy: policy} do
    played = decide_after_playing(start_agent(policy, []))
    observed = decide_after_observing(start_agent(policy, []))
    assert observed == played
  end

  test "stateful-step path: observe-then-decide == play-then-decide", %{policy: policy} do
    played = decide_after_playing(start_agent(policy, stateful_step: true))
    observed = decide_after_observing(start_agent(policy, stateful_step: true))
    assert observed == played
  end

  test "observe fills the window and reset_buffer empties it again", %{policy: policy} do
    agent = start_agent(policy, [])

    for {f, x} <- @history do
      :ok = Agent.observe(agent, game_state(f, x), nil, player_port: 1)
    end

    state = :sys.get_state(agent)
    assert :queue.len(state.frame_buffer) == @window
    assert state.last_debounce_frame == 11
    assert state.last_action == nil

    :ok = Agent.reset_buffer(agent)
    assert :queue.len(:sys.get_state(agent).frame_buffer) == 0
  end

  test "observing a history changes the decision distribution vs a cold start", %{policy: policy} do
    # Sanity that the pins above are not vacuous: history must reach the
    # heads. Argmax of a random-init trunk can coincide, so compare the
    # head confidences (the probabilities), not the sampled action.
    {f, x} = @decision
    # probe: true reads the stick heads' softmax (17 buckets) — far less likely
    # to saturate to identical values under a random init than 8 sigmoids
    {:ok, cold} = Agent.observe(start_agent(policy, []), game_state(f, x), nil, player_port: 1, probe: true)

    warm_agent = start_agent(policy, [])

    for {hf, hx} <- @history do
      :ok = Agent.observe(warm_agent, game_state(hf, hx), ControllerState.neutral(), player_port: 1)
    end

    {:ok, warm} = Agent.observe(warm_agent, game_state(f, x), nil, player_port: 1, probe: true)
    assert cold.main_x != warm.main_x or cold.main_y != warm.main_y
  end

  test "ControllerState.from_input/1 inverts to_input/1" do
    cs = %{
      ControllerState.neutral()
      | main_stick: %{x: 0.25, y: 0.75},
        c_stick: %{x: 1.0, y: 0.5},
        l_shoulder: 0.6,
        button_b: true,
        button_y: true
    }

    assert ControllerState.from_input(ControllerState.to_input(cs)) == cs
    assert ControllerState.from_input(%{}) == ControllerState.neutral()
  end
end
