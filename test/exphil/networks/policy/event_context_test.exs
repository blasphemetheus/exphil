defmodule ExPhil.Networks.Policy.EventContextTest do
  @moduledoc """
  Event context (2026-10-05): the previous input as a FEATURE of the AR
  heads on top of the event heads (`event_context: true`). Zero-initialised,
  so it starts as the plain event head; the sampler mirror reproduces the
  Axon head's logits once the context kernels are non-zero.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.{Heads, Sampling}

  @embed 24
  @hidden 8
  @buckets 4

  defp model(opts) do
    Policy.build_temporal(
      Keyword.merge(
        [embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1, window_size: 6,
         head: :autoregressive, axis_buckets: @buckets, shoulder_buckets: 2, dropout: 0.0,
         button_events: true, stick_events: true],
        opts
      )
    )
  end

  defp inputs(prev_buttons, prev_sticks) do
    %{
      "state_sequence" => Nx.iota({2, 6, @embed}, type: :f32) |> Nx.divide(40.0),
      "tf_buttons" => Nx.broadcast(0.0, {2, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "prev_buttons" => prev_buttons,
      "prev_sticks" => prev_sticks
    }
  end

  test "context params exist, start at zero, and the head equals the plain event head at init" do
    {init, predict} = Axon.build(model(event_context: true), mode: :inference)
    {init0, predict0} = Axon.build(model([]), mode: :inference)
    in0 = inputs(Nx.broadcast(1.0, {2, 8}), Nx.broadcast(Nx.tensor(3, type: :s64), {2, 4}))

    params = init.(in0, Axon.ModelState.empty())
    assert Nx.shape(params.data["ar_prev_buttons_embed"]["kernel"]) == {8, 128} or
             elem(Nx.shape(params.data["ar_prev_buttons_embed"]["kernel"]), 0) == 8
    assert Nx.to_number(Nx.sum(Nx.abs(params.data["ar_prev_stick_1_embed"]["kernel"]))) == 0.0

    # same trunk/head weights without the context layers
    plain = init0.(in0, Axon.ModelState.empty())
    shared = Map.take(params.data, Map.keys(plain.data))
    out = predict.(%{params | data: Map.merge(params.data, shared)}, in0)
    out0 = predict0.(%{plain | data: shared}, in0)
    assert Nx.all_close(elem(out, 0), elem(out0, 0), atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "a non-zero context moves the button logits and the sampler mirror agrees" do
    {init, predict} = Axon.build(model(event_context: true), mode: :inference)
    prev_b = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    prev_s = Nx.tensor([[1, 1, 1, 1], [1, 3, 1, 1]], type: :s64)
    in1 = inputs(prev_b, prev_s)
    params = init.(in1, Axon.ModelState.empty())

    key = Nx.Random.key(3)
    {kb, key} = Nx.Random.normal(key, shape: Nx.shape(params.data["ar_prev_buttons_embed"]["kernel"]))
    {ks, _} = Nx.Random.normal(key, shape: Nx.shape(params.data["ar_prev_stick_1_embed"]["kernel"]))
    data =
      params.data
      |> put_in(["ar_prev_buttons_embed", "kernel"], kb)
      |> put_in(["ar_prev_stick_1_embed", "kernel"], ks)
    params = %{params | data: data}

    {b, mx, _my, _cx, _cy, _sh} = predict.(params, in1)
    # row 1 differs from row 0 only through the context (same state rows would
    # be identical otherwise) — assert the context reaches the button head
    in_same = inputs(prev_b, prev_s) |> Map.put("state_sequence", Nx.broadcast(0.1, {2, 6, @embed}))
    {b_same, _, _, _, _, _} = predict.(params, in_same)
    refute Nx.all_close(b_same[0], b_same[1], atol: 1.0e-6) |> Nx.to_number() == 1

    # sampler mirror: trunk features from the trunk-only build, then the defn head
    trunk = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1,
      window_size: 6, dropout: 0.0)
    {_, trunk_predict} = Axon.build(trunk, mode: :inference)
    features = trunk_predict.(params, %{"state_sequence" => in1["state_sequence"]})

    out =
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: prev_b, event_prev_sticks: prev_s)

    assert Nx.all_close(out.logits.buttons, b, atol: 1.0e-4) |> Nx.to_number() == 1
    assert Nx.all_close(out.logits.main_x, mx, atol: 1.0e-4) |> Nx.to_number() == 1
    _ = Heads
  end
end
