defmodule ExPhil.Networks.Policy.StickDurationTest do
  @moduledoc """
  `stick_duration: C` (2026-10-06): the semi-Markov main stick. The head grows
  a duration component ("ar_duration_*", C classes) and a hold-age feature
  ("prev_age" → "ar_prev_age_embed"); the loss scores main_x/main_y at
  decision frames only and adds the duration cross-entropy; the sampler
  holds committed rows on their previous pair and counts the commitment down.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.{Heads, Sampling}
  alias ExPhil.Training.Imitation.Loss

  @embed 24
  @hidden 8
  @buckets 4
  @cap 4

  defp model(opts) do
    Policy.build_temporal(
      Keyword.merge(
        [embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1, window_size: 6,
         head: :autoregressive, axis_buckets: @buckets, shoulder_buckets: 2, dropout: 0.0,
         button_events: true, stick_events: true, stick_duration: @cap],
        opts
      )
    )
  end

  defp inputs(prev_sticks, age) do
    %{
      "state_sequence" => Nx.iota({2, 6, @embed}, type: :f32) |> Nx.divide(40.0),
      "tf_buttons" => Nx.broadcast(0.0, {2, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {2}),
      "prev_buttons" => Nx.broadcast(0.0, {2, 8}),
      "prev_sticks" => prev_sticks,
      "prev_age" => age
    }
  end

  test "age_bucket bins hold ages" do
    ages = Nx.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 23, 24, 31, 32, 500])
    assert Nx.to_list(Heads.age_bucket(ages)) == [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    assert Heads.age_buckets() == 11
  end

  test "the head grows a C-class duration component and the age embedding; output is a 7-tuple" do
    {init, predict} = Axon.build(model([]), mode: :inference)
    prev_s = Nx.tensor([[1, 1, 1, 1], [0, 3, 2, 1]], type: :s64)
    in1 = inputs(prev_s, Nx.tensor([3, 40], type: :s64))
    params = init.(in1, Axon.ModelState.empty())

    {_, width} = Nx.shape(params.data["ar_duration_logits"]["kernel"])
    assert width == @cap
    assert Nx.shape(params.data["ar_prev_age_embed"]["kernel"]) == {Heads.age_buckets(), 128}
    # zero-initialised: the age feature is inert until trained
    assert Nx.to_number(Nx.sum(Nx.abs(params.data["ar_prev_age_embed"]["kernel"]))) == 0.0

    out = predict.(params, in1)
    assert tuple_size(out) == 7
    {{_b, mx, _my, _cx, _cy, _sh}, dur} = Loss.split_duration_head(out)
    assert Nx.shape(mx) == {2, @buckets + 1}
    assert Nx.shape(dur) == {2, @cap}
  end

  test "it refuses without stick_events and on the BPTT path" do
    assert_raise ArgumentError, ~r/stick_duration/, fn ->
      Policy.build_temporal(embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1, window_size: 6,
        head: :autoregressive, axis_buckets: @buckets, shoulder_buckets: 2, stick_duration: @cap)
    end
  end

  describe "decision targets" do
    test "mask = pair changed or age is a multiple of C; duration = leading run of future matches" do
      # rows: 0 event (pair differs from prev); 1 continuation (age 8 = 2C);
      # 2 mid-hold (age 5) -> not a decision frame; 3 event at the game's end
      inputs = %{
        "prev_sticks" => Nx.tensor([[1, 1, 0, 0], [2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0]], type: :s64),
        "prev_age" => Nx.tensor([3, 8, 5, 1], type: :s64)
      }

      actions = %{
        main_x: Nx.tensor([2, 2, 2, 2], type: :s64),
        main_y: Nx.tensor([2, 2, 2, 2], type: :s64),
        # futures t+1..t+3 (chunk horizon 3 = C - 1)
        future_main_x: Nx.tensor([[2, 2, 2], [2, 0, 2], [2, 2, 2], [2, 2, 2]], type: :s64),
        future_main_y: Nx.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], type: :s64),
        future_mask: Nx.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]], type: :f32)
      }

      %{mask: mask, duration: d} = Loss.stick_decision_targets(inputs, actions, @cap)
      assert Nx.to_list(mask) == [1.0, 1.0, 0.0, 1.0]

      # a button edge makes a mid-hold frame a decision frame too
      with_buttons =
        Map.put(inputs, "prev_buttons", Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], type: :f32))

      acts_b = Map.put(actions, :buttons, Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], type: :f32))
      %{mask: mask_b} = Loss.stick_decision_targets(with_buttons, acts_b, @cap)
      assert Nx.to_list(mask_b) == [1.0, 1.0, 1.0, 1.0]
      # row 0: held through all 3 futures -> "C+" (3); row 1: changes at t+2 -> 1;
      # row 3: game ends after t+1 -> 1
      assert Nx.to_list(d) == [3, 1, 3, 1]
    end

    test "prev_age_from_window counts the trailing run of equal main-stick pairs in the prev slot" do
      offset = 4
      dim = offset + 13
      # window of 5 positions; prev slot main x/y at offset+8, offset+9 in [-1, 1]
      mk = fn pairs ->
        Nx.tensor(
          for {x, y} <- pairs do
            List.duplicate(0.0, offset + 8) ++ [x, y] ++ List.duplicate(0.0, dim - offset - 10)
          end
        )
      end

      states = Nx.stack([
        mk.([{-1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}]),
        mk.([{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}]),
        mk.([{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}])
      ])

      assert Nx.to_list(Loss.prev_age_from_window(states, offset, 16)) == [4, 5, 1]
    end
  end

  test "the sampler holds committed rows on prev and counts the commitment down" do
    {init, _predict} = Axon.build(model([]), mode: :inference)
    prev_s = Nx.tensor([[3, 0, 1, 1], [0, 3, 2, 1]], type: :s64)
    in1 = inputs(prev_s, Nx.tensor([2, 9], type: :s64))
    params = init.(in1, Axon.ModelState.empty())

    # no button edge this frame: a random init presses ~half the buttons at
    # argmax, which would (correctly) make every row a decision frame
    released = Nx.broadcast(-40.0, Nx.shape(params.data["ar_buttons_logits"]["bias"]))
    params = %{params | data: put_in(params.data, ["ar_buttons_logits", "bias"], released)}

    trunk = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: 1,
      window_size: 6, dropout: 0.0)
    {_, trunk_predict} = Axon.build(trunk, mode: :inference)
    features = trunk_predict.(params, %{"state_sequence" => in1["state_sequence"]})

    # row 0 committed for 2 more frames, row 1 deciding
    out =
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: in1["prev_buttons"], event_prev_sticks: prev_s,
        event_prev_age: in1["prev_age"], stick_commit: Nx.tensor([2, 0], type: :s64))

    [mx0, _] = Nx.to_list(out.main_x)
    [my0, _] = Nx.to_list(out.main_y)
    assert {mx0, my0} == {3, 0}
    [c0, c1] = Nx.to_list(out.stick_commit)
    assert c0 == 1
    assert c1 in 0..(@cap - 1)

    # a button edge this frame ends the commitment: force a B press via a huge
    # press logit bias and check row 0 re-decides (commit re-sampled, not 1)
    bias = params.data["ar_buttons_logits"]["bias"] |> Nx.put_slice([1], Nx.tensor([40.0]))
    dbias = params.data["ar_duration_logits"]["bias"] |> Nx.put_slice([@cap - 1], Nx.tensor([40.0]))
    pressed = %{params | data: params.data |> put_in(["ar_buttons_logits", "bias"], bias) |> put_in(["ar_duration_logits", "bias"], dbias)}

    out2 =
      Sampling.sample_autoregressive_from_features(pressed, features,
        deterministic: true, event_prev_buttons: in1["prev_buttons"], event_prev_sticks: prev_s,
        event_prev_age: in1["prev_age"], stick_commit: Nx.tensor([2, 0], type: :s64))

    assert Nx.to_list(out2.buttons) |> hd() |> Enum.at(1) == 1
    # row 0 re-decided (duration argmax = C-1 = 3) instead of counting down to 1
    assert Nx.to_list(out2.stick_commit) == [@cap - 1, @cap - 1]

    # the duration checkpoint refuses to sample blind
    assert_raise ArgumentError, ~r/stick_duration/, fn ->
      Sampling.sample_autoregressive_from_features(params, features,
        deterministic: true, event_prev_buttons: in1["prev_buttons"], event_prev_sticks: prev_s)
    end
  end
end
