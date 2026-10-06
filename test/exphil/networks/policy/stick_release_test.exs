defmodule ExPhil.Networks.Policy.StickReleaseTest do
  @moduledoc """
  `stick_release: true` (2026-10-05): each stick axis gets a release logit
  beside the hold logit ("<prefix>_logits_hr", K + 2 wide), collapsed by
  `Heads.collapse_hold_release_change/2`; the sampler mirror picks the same
  params and collapse by name.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.Sampling

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

  defp inputs(prev_sticks) do
    %{
      "state_sequence" => Nx.iota({2, 6, @embed}, type: :f32) |> Nx.divide(40.0),
      "tf_buttons" => Nx.broadcast(0.0, {2, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "prev_buttons" => Nx.broadcast(0.0, {2, 8}),
      "prev_sticks" => prev_sticks
    }
  end

  test "release params exist (K + 2 wide, _hr name), outputs are log-probs over K, sampler mirror agrees" do
    {init, predict} = Axon.build(model(stick_release: true), mode: :inference)
    prev_s = Nx.tensor([[1, 1, 1, 1], [0, 3, 2, 1]], type: :s64)
    in1 = inputs(prev_s)
    params = init.(in1, Axon.ModelState.empty())

    refute Map.has_key?(params.data, "ar_main_x_logits")
    {_, width} = Nx.shape(params.data["ar_main_x_logits_hr"]["kernel"])
    assert width == @buckets + 1 + 2

    {b, mx, my, _cx, _cy, _sh} = predict.(params, in1)
    assert Nx.shape(mx) == {2, @buckets + 1}
    for row <- Nx.to_list(Nx.exp(mx)), do: assert_in_delta(Enum.sum(row), 1.0, 1.0e-4)
    for row <- Nx.to_list(Nx.exp(my)), do: assert_in_delta(Enum.sum(row), 1.0, 1.0e-4)

    trunk = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1,
      window_size: 6, dropout: 0.0)
    {_, trunk_predict} = Axon.build(trunk, mode: :inference)
    features = trunk_predict.(params, %{"state_sequence" => in1["state_sequence"]})

    out =
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: in1["prev_buttons"], event_prev_sticks: prev_s)

    assert Nx.all_close(out.logits.buttons, b, atol: 1.0e-4) |> Nx.to_number() == 1
    assert Nx.all_close(out.logits.main_x, mx, atol: 1.0e-4) |> Nx.to_number() == 1
  end

  test "a strongly positive release logit puts the centre bucket on top whatever prev was" do
    {init, predict} = Axon.build(model(stick_release: true), mode: :inference)
    prev_s = Nx.tensor([[0, 0, 0, 0], [4, 4, 4, 4]], type: :s64)
    in1 = inputs(prev_s)
    params = init.(in1, Axon.ModelState.empty())
    k = @buckets + 1
    bias = params.data["ar_main_x_logits_hr"]["bias"] |> Nx.put_slice([k + 1], Nx.tensor([30.0])) |> Nx.put_slice([k], Nx.tensor([-30.0]))
    params = %{params | data: put_in(params.data, ["ar_main_x_logits_hr", "bias"], bias)}
    {_b, mx, _my, _cx, _cy, _sh} = predict.(params, in1)
    assert Nx.to_list(Nx.argmax(mx, axis: -1)) == [div(k, 2), div(k, 2)]
  end
end
