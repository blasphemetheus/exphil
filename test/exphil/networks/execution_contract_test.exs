defmodule ExPhil.Networks.ExecutionContractTest do
  use ExUnit.Case, async: false
  alias ExPhil.Networks.Policy.ExecutionContract, as: Contract

  test "unstamped policies retain legacy behavior and unknown precision is explicit" do
    assert Contract.load(%{}).recurrent_state == :legacy_random
    assert Contract.load(%{}).training_precision == :unknown
    assert Contract.load(%{}).inference_precision == :f32
    assert_raise ArgumentError, fn -> Contract.load(%{recurrent_state: :zeros}) end
    assert_raise ArgumentError, fn -> Contract.load(%{execution_contract: :future}) end
  end

  test "zero-state v1 requires end-to-end F32 windowed GRU" do
    config = %{temporal: true, backbone: :gru, recurrent_state: :zeros, precision: :f32}
    fields = Contract.training(config)
    assert Contract.load(Map.merge(config, fields)) == fields

    for override <- [
          %{precision: :bf16},
          %{bptt: true},
          %{mixed_precision: true},
          %{backbone: :lstm}
        ] do
      assert_raise ArgumentError, fn -> Contract.training(Map.merge(config, override)) end
    end

    assert_raise ArgumentError, fn ->
      Contract.load(Map.merge(config, %{fields | inference_precision: :bf16}))
    end
  end

  test "explicit state rejects fused and non-GRU builders" do
    for extra <- [
          [cell_type: :lstm],
          [cell_type: :gru, fused_block: true],
          [recurrent_state: :invalid]
        ] do
      assert_raise ArgumentError, fn ->
        Edifice.Recurrent.build(Keyword.merge([embed_dim: 3, recurrent_state: :zeros], extra))
      end
    end
  end

  test "new-contract readiness requires matching report provenance" do
    config = %{temporal: true, backbone: :gru, recurrent_state: :zeros, precision: :f32}
    fields = Contract.training(config)
    config = Map.merge(config, fields)
    report = %{"execution_contract" => fields |> Jason.encode!() |> Jason.decode!()}
    assert :ok == Contract.verify_report!(config, report)
    assert_raise ArgumentError, fn -> Contract.verify_report!(config, %{}) end
    assert_raise ArgumentError, fn -> Contract.verify_report!(%{}, report) end
    assert :ok == Contract.verify_report!(%{}, %{})
  end

  test "Edifice zero-state GRU is row, permutation, and batch-size invariant" do
    model =
      Edifice.Recurrent.build(
        embed_dim: 3,
        hidden_size: 8,
        num_layers: 2,
        window_size: 4,
        cell_type: :gru,
        dropout: 0.0,
        recurrent_state: :zeros
      )

    refute Enum.any?(Map.values(model.nodes), &(&1.op_name == :recurrent_state))
    {init, predict} = Axon.build(model, compiler: Nx.Defn.Evaluator)
    single = Nx.tensor([[[0.1, 0.2, 0.3], [0.2, 0.4, 0.1], [0.3, 0.1, 0.5], [0.4, 0.2, 0.0]]])
    params = init.(single, Axon.ModelState.empty())
    expected = predict.(params, single)
    copies = Nx.broadcast(single, {4, 4, 3})

    assert Nx.all_close(predict.(params, copies), Nx.broadcast(expected, {4, 8}), atol: 1.0e-5)
           |> Nx.to_number() == 1

    mixed = Nx.concatenate([single, Nx.multiply(single, -1), Nx.multiply(single, 2)], axis: 0)
    order = Nx.tensor([2, 0, 1])

    assert Nx.all_close(
             predict.(params, Nx.take(mixed, order)),
             Nx.take(predict.(params, mixed), order),
             atol: 1.0e-5
           )
           |> Nx.to_number() == 1

    legacy =
      Edifice.Recurrent.build(embed_dim: 3, hidden_size: 8, window_size: 4, cell_type: :gru)

    assert Enum.any?(Map.values(legacy.nodes), &(&1.op_name == :recurrent_state))
  end
end
