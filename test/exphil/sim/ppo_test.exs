defmodule ExPhil.Sim.PPOTest do
  use ExUnit.Case, async: false
  alias ExPhil.Sim.PPO

  test "Polaris updates do not send its default nil state through LazyContainer" do
    p = %{"layer" => %{"kernel" => Nx.tensor([1.0, 2.0])}}
    u = %{"layer" => %{"kernel" => Nx.tensor([-0.1, 0.2])}}
    result = PPO.apply_updates(p, u)
    [a, b] = Nx.to_flat_list(result["layer"]["kernel"])
    assert_in_delta a, 0.9, 1.0e-6
    assert_in_delta b, 2.2, 1.0e-6
  end

  test "GAE bootstraps truncations but cuts credit at a terminal" do
    rewards = Nx.tensor([[1.0, 2.0], [1.0, 2.0]])
    values = Nx.tensor([[0.5, 0.7], [0.5, 0.7]])
    dones = Nx.tensor([[0, 0], [0, 1]])
    {adv, returns} = PPO.gae(rewards, values, dones, 0.9, 1.0, Nx.tensor([3.0, 3.0]))
    Enum.zip(Nx.to_flat_list(returns), [5.23, 4.7, 2.8, 2.0])
    |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-5 end)
    Enum.zip(Nx.to_flat_list(adv), [4.73, 4.0, 2.3, 1.3])
    |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-5 end)
  end

  test "head PPO update changes parameters and keeps finite metrics" do
    model = PPO.head_model(4, residual_size: 8, component_hidden: 8)
    {init, predict} = Axon.build(model, mode: :inference)
    features = Nx.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    action = %{buttons: Nx.broadcast(0.0, {2, 8}), main_x: Nx.tensor([1, 2]),
      main_y: Nx.tensor([3, 4]), c_x: Nx.tensor([5, 6]), c_y: Nx.tensor([7, 8]), shoulder: Nx.tensor([0, 1])}
    params = init.(PPO.tf_inputs(features, action), Axon.ModelState.empty()).data
    logits = PPO.logits_of(predict, params, features, action)
    batch = %{features: features, action: action, advantages: Nx.tensor([1.0, -1.0]),
      logp_old: PPO.logp(logits, action), prior_logits: logits}
    {opt_init, update} = Polaris.Optimizers.adam(learning_rate: 1.0e-3)
    {next, _, metrics} = PPO.update_step(predict, params, opt_init.(params), update, batch, [])
    assert Enum.all?(metrics, fn {_, v} -> is_number(v) end)
    assert_in_delta metrics.kl, 0.0, 1.0e-5
    delta = Nx.subtract(next["ar_buttons_logits"]["kernel"], params["ar_buttons_logits"]["kernel"])
    assert Nx.to_number(Nx.sum(Nx.abs(delta))) > 0.0
  end
end
