defmodule ExPhil.Networks.Policy.DangerContextTest do
  @moduledoc """
  Danger context (2026-10-08): the current frame's own-player danger features
  (y, jumps left, on_ground, speed_y, ledge distance), sliced from the state
  input, as a FEATURE of the AR heads through a ReLU readout with a
  zero-initialised output. Starts as the plain event head; the sampler mirror
  reproduces the Axon head once the readout is non-zero; the column indices
  match a perturbation of the real player embedding.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Player
  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.Sampling

  @embed 24
  @hidden 8
  @buckets 4
  @cols [3, 7, 9, 12]

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

  defp inputs(state_sequence) do
    %{
      "state_sequence" => state_sequence,
      "tf_buttons" => Nx.broadcast(0.0, {2, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "prev_buttons" => Nx.broadcast(0.0, {2, 8}),
      "prev_sticks" => Nx.broadcast(Nx.tensor(2, type: :s64), {2, 4})
    }
  end

  test "readout params exist, output starts at zero, head equals the plain head at init; no new input" do
    m = model(danger_columns: @cols)
    assert Map.keys(Axon.get_inputs(m)) |> Enum.sort() == Map.keys(Axon.get_inputs(model([]))) |> Enum.sort()

    {init, predict} = Axon.build(m, mode: :inference)
    {init0, predict0} = Axon.build(model([]), mode: :inference)
    in0 = inputs(Nx.iota({2, 6, @embed}, type: :f32) |> Nx.divide(40.0))

    params = init.(in0, Axon.ModelState.empty())
    assert Nx.shape(params.data["ar_danger_hidden"]["kernel"]) == {length(@cols), 32}
    assert Nx.to_number(Nx.sum(Nx.abs(params.data["ar_danger_embed"]["kernel"]))) == 0.0

    plain = init0.(in0, Axon.ModelState.empty())
    shared = Map.take(params.data, Map.keys(plain.data))
    out = predict.(%{params | data: Map.merge(params.data, shared)}, in0)
    out0 = predict0.(%{plain | data: shared}, in0)
    assert Nx.all_close(elem(out, 0), elem(out0, 0), atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "a non-zero readout reads the LAST frame's danger columns and the sampler mirror agrees" do
    {init, predict} = Axon.build(model(danger_columns: @cols), mode: :inference)
    seq = Nx.broadcast(0.1, {2, 6, @embed})
    params = init.(inputs(seq), Axon.ModelState.empty())
    key = Nx.Random.key(5)
    {k, _} = Nx.Random.normal(key, shape: Nx.shape(params.data["ar_danger_embed"]["kernel"]))
    params = %{params | data: put_in(params.data, ["ar_danger_embed", "kernel"], k)}

    # row 1 differs from row 0 only in the last frame's y column (index 3)
    bump = Nx.indexed_put(seq, Nx.tensor([[1, 5, 3]]), Nx.tensor([2.0]))
    {b, mx, _, _, _, _} = predict.(params, inputs(bump))
    refute Nx.all_close(b[0], b[1], atol: 1.0e-6) |> Nx.to_number() == 1

    # a change in an EARLIER frame's danger column, or in a non-danger
    # column of the last frame, does not reach the head through the readout
    # (the trunk may still see it, so compare the readout contribution only)
    danger_rows = bump[[.., -1]] |> Nx.take(Nx.tensor(@cols), axis: -1)
    assert Nx.shape(danger_rows) == {2, length(@cols)}

    trunk = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1,
      window_size: 6, dropout: 0.0)
    {_, trunk_predict} = Axon.build(trunk, mode: :inference)
    features = trunk_predict.(params, %{"state_sequence" => bump})

    out =
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: Nx.broadcast(0.0, {2, 8}),
        event_prev_sticks: Nx.broadcast(Nx.tensor(2, type: :s64), {2, 4}), danger: danger_rows)

    assert Nx.all_close(out.logits.buttons, b, atol: 1.0e-4) |> Nx.to_number() == 1
    assert Nx.all_close(out.logits.main_x, mx, atol: 1.0e-4) |> Nx.to_number() == 1

    assert_raise ArgumentError, ~r/danger-context head/, fn ->
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: Nx.broadcast(0.0, {2, 8}),
        event_prev_sticks: Nx.broadcast(Nx.tensor(2, type: :s64), {2, 4}))
    end
  end

  test "Player.danger_columns matches a perturbation of the real embedding" do
    config = Player.default_config()
    base = %ExPhil.Bridge.Player{x: 10.0, y: -30.0, percent: 0.0, stock: 4, facing: 1, action: 29, action_frame: 1,
      invulnerable: false, jumps_left: 1, on_ground: false, shield_strength: 60.0, hitstun_frames_left: 0,
      speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: -1.0, speed_x_attack: 0.0, speed_y_attack: 0.0}
    changed = fn over ->
      a = Player.embed(base, config, 85.0) |> Nx.backend_transfer(Nx.BinaryBackend)
      b = Player.embed(Map.merge(base, Map.new(over)), config, 85.0) |> Nx.backend_transfer(Nx.BinaryBackend)
      Nx.not_equal(a, b) |> Nx.to_flat_list() |> Enum.with_index() |> Enum.filter(&(elem(&1, 0) == 1)) |> Enum.map(&elem(&1, 1))
    end

    cols = Player.danger_columns(config)
    [y, jumps, on_ground, speed_y, ledge] = cols
    assert changed.(y: -50.0) == [y]
    assert changed.(jumps_left: 0) == [jumps]
    assert changed.(on_ground: true) == [on_ground]
    assert changed.(speed_y_self: 2.0) == [speed_y]
    # x moves x itself and the ledge distance
    x_cols = changed.(x: 40.0)
    assert ledge in x_cols and length(x_cols) == 2
  end
end
