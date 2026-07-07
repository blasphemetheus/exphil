defmodule ExPhil.Networks.BackendParityTest do
  @moduledoc """
  Hunt #6: the same model, params, and batch must produce (near-)identical
  results through EXLA and the pure-Elixir evaluator. Catches compiler/kernel
  bugs — the class the nx fuzz work hunts upstream — before they corrupt
  training or live inference.

  Tagged :slow (evaluator forward passes are expensive).
  Run with: mix test test/exphil/networks/backend_parity_test.exs --include slow
  """
  use ExUnit.Case, async: false

  @moduletag :slow
  @moduletag timeout: 300_000

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Utils

  @embed 64
  @batch 4

  defp forward(predict_fn, params, x), do: predict_fn.(Utils.ensure_model_state(params), x)

  defp max_head_diff(a_tuple, b_tuple) do
    [Tuple.to_list(a_tuple), Tuple.to_list(b_tuple)]
    |> Enum.zip()
    |> Enum.map(fn {a, b} ->
      a
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.subtract(Nx.backend_copy(b, Nx.BinaryBackend))
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()
    end)
    |> Enum.max()
  end

  test "MLP policy forward pass: EXLA == evaluator within 1e-4" do
    model = Policy.build(embed_size: @embed, hidden_sizes: [32, 32], dropout: 0.0)

    {init_fn, _} = Axon.build(model, mode: :inference)
    key = Nx.Random.key(1234)
    {x, _} = Nx.Random.uniform(key, shape: {@batch, @embed}, type: :f32)
    x = Nx.backend_copy(x, Nx.BinaryBackend)

    params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())

    {_, exla_fn} = Axon.build(model, mode: :inference, compiler: EXLA)
    {_, eval_fn} = Axon.build(model, mode: :inference)

    exla_out = forward(exla_fn, params, x)
    eval_out = forward(eval_fn, params, x)

    diff = max_head_diff(exla_out, eval_out)

    assert diff < 1.0e-4,
           "EXLA and evaluator disagree by #{diff} on identical MLP forward pass — " <>
             "compiler/kernel bug territory (see the nx fuzz findings)"
  end

  test "temporal GRU policy forward pass: EXLA == evaluator within 1e-4" do
    window = 8

    model =
      Policy.build_temporal(
        embed_size: @embed,
        backbone: :gru,
        window_size: window,
        hidden_size: 32,
        num_layers: 1,
        dropout: 0.0
      )

    {init_fn, _} = Axon.build(model, mode: :inference)
    key = Nx.Random.key(4321)
    {x, _} = Nx.Random.uniform(key, shape: {@batch, window, @embed}, type: :f32)
    x = Nx.backend_copy(x, Nx.BinaryBackend)

    params = init_fn.(Nx.template({@batch, window, @embed}, :f32), Axon.ModelState.empty())

    {_, exla_fn} = Axon.build(model, mode: :inference, compiler: EXLA)
    {_, eval_fn} = Axon.build(model, mode: :inference)

    exla_out = forward(exla_fn, params, x)
    eval_out = forward(eval_fn, params, x)

    diff = max_head_diff(exla_out, eval_out)

    assert diff < 1.0e-4,
           "EXLA and evaluator disagree by #{diff} on identical GRU forward pass"
  end
end
