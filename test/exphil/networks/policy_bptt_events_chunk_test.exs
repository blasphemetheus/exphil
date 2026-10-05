defmodule ExPhil.Networks.PolicyBpttEventsChunkTest do
  @moduledoc """
  Carried-state BPTT with the windowed recipe that works (2026-10-04): event
  heads at every timestep (previous input from the prev-action slot, trunk
  slot zeroed) and chunk targets (future heads scored against the shifted
  targets a contiguous chunk already holds).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Imitation.Loss

  @batch 2
  @time 6
  @embed 24
  @hidden 8
  @layers 1
  @offset 4
  @buckets 4

  defp model(opts) do
    Policy.build_temporal_bptt(
      Keyword.merge(
        [embed_size: @embed, backbone: :gru, hidden_size: @hidden, num_layers: @layers, window_size: @time,
         head: :autoregressive, axis_buckets: @buckets, shoulder_buckets: 2, dropout: 0.0],
        opts
      )
    )
  end

  defp tf do
    %{
      "tf_buttons" => Nx.broadcast(0.0, {@batch, @time, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time})
    }
  end

  defp actions do
    %{
      buttons: Nx.broadcast(0.0, {@batch, @time, 8}),
      main_x: Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      main_y: Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      c_x: Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      c_y: Nx.broadcast(Nx.tensor(2, type: :s64), {@batch, @time}),
      shoulder: Nx.broadcast(Nx.tensor(0, type: :s64), {@batch, @time})
    }
  end

  test "event heads at every timestep + chunk container: shapes and params" do
    {init, predict} = Axon.build(model(button_events: true, stick_events: true, chunk_horizon: 2), mode: :inference)

    inputs =
      Map.merge(tf(), %{
        "state_sequence" => Nx.broadcast(0.0, {@batch, @time, @embed}),
        "initial_hidden" => Nx.broadcast(0.0, {@batch, @layers, @hidden}),
        "prev_buttons" => Nx.broadcast(0.0, {@batch, @time, 8}),
        "prev_sticks" => Nx.broadcast(Nx.tensor(1, type: :s64), {@batch, @time, 4})
      })

    params = init.(inputs, Axon.ModelState.empty())
    {{head, futures}, hidden} = predict.(params, inputs)

    assert tuple_size(head) == 6
    assert tuple_size(futures) == 2
    assert Nx.shape(elem(head, 0)) == {@batch, @time, 8}
    assert Nx.shape(elem(head, 1)) == {@batch, @time, @buckets + 1}
    assert Nx.shape(elem(elem(futures, 1), 0)) == {@batch, @time, 8}
    assert Nx.shape(hidden) == {@batch, @layers, @hidden}
    # the event button head is the 16-logit press/release dense under the collapse
    assert Nx.shape(params.data["ar_buttons_logits"]["kernel"]) |> elem(1) == 16
    assert Enum.any?(Map.keys(params.data), &String.starts_with?(&1, "future2_"))
  end

  test "BPTT loss builder: event inputs come from the slot, future targets shift" do
    model = model(button_events: true, stick_events: true, chunk_horizon: 2)
    {init, predict} = Axon.build(model, mode: :train)

    init_inputs =
      Map.merge(tf(), %{
        "state_sequence" => Nx.broadcast(0.0, {@batch, @time, @embed}),
        "initial_hidden" => Nx.broadcast(0.0, {@batch, @layers, @hidden}),
        "prev_buttons" => Nx.broadcast(0.0, {@batch, @time, 8}),
        "prev_sticks" => Nx.broadcast(Nx.tensor(1, type: :s64), {@batch, @time, 4})
      })

    params = init.(init_inputs, Axon.ModelState.empty())

    config = %{
      head: :autoregressive, precision: :f32, button_events: true, stick_events: true,
      prev_action_offset: @offset, axis_buckets: @buckets, chunk_horizon: 2, chunk_weight: 1.0,
      focal_loss: false, label_smoothing: 0.0, entropy_weight: 0.0
    }

    loss_and_grad = Loss.build_bptt_loss_and_grad_fn(predict, config)

    # game state everywhere (zero states give zero GRU features and zero kernel
    # grads) plus a slot pattern the trunk must not see: all-ones prev buttons
    base = Nx.iota({@batch, @time, @embed}, type: :f32) |> Nx.divide(50.0)
    ones = Nx.broadcast(1.0, {@batch, @time, 8})
    states = Nx.put_slice(base, [0, 0, @offset], ones)
    weights = Nx.broadcast(1.0, {@batch, @time})
    carry = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    {{loss, {new_carry, _state}}, grads} = loss_and_grad.(params, states, actions(), weights, carry)

    assert Nx.shape(loss) == {}
    assert Nx.to_number(loss) > 0.0
    assert Nx.shape(new_carry) == {@batch, @layers, @hidden}
    # future heads received gradient (shifted targets, masked tail)
    assert Nx.to_number(Nx.sum(Nx.abs(grads.data["future1_buttons_logits"]["kernel"]))) > 0.0
    # the event button head received gradient through the collapse at every position
    assert Nx.to_number(Nx.sum(Nx.abs(grads.data["ar_buttons_logits"]["kernel"]))) > 0.0

    # the slot pattern reaches the heads, not the trunk: flipping prev buttons
    # in the slot changes the loss (release head vs press head), while the
    # same flip on a model without event heads would be the trunk's business
    states0 = Nx.put_slice(base, [0, 0, @offset], Nx.broadcast(0.0, {@batch, @time, 8}))
    {{loss0, _}, _} = loss_and_grad.(params, states0, actions(), weights, carry)
    refute Nx.to_number(loss0) == Nx.to_number(loss)
  end

  test "a batch of one-frame segments (future weights all zero) scores finite" do
    # the 2026-10-04 fatal batch: every row one real frame + padding, so
    # w[i] * w[i+j] is zero everywhere for every future head
    model = model(button_events: true, stick_events: true, chunk_horizon: 2)
    {init, predict} = Axon.build(model, mode: :train)

    init_inputs =
      Map.merge(tf(), %{
        "state_sequence" => Nx.broadcast(0.0, {@batch, @time, @embed}),
        "initial_hidden" => Nx.broadcast(0.0, {@batch, @layers, @hidden}),
        "prev_buttons" => Nx.broadcast(0.0, {@batch, @time, 8}),
        "prev_sticks" => Nx.broadcast(Nx.tensor(1, type: :s64), {@batch, @time, 4})
      })

    params = init.(init_inputs, Axon.ModelState.empty())
    config = %{
      head: :autoregressive, precision: :f32, button_events: true, stick_events: true,
      prev_action_offset: @offset, axis_buckets: @buckets, chunk_horizon: 2, chunk_weight: 1.0,
      focal_loss: false, label_smoothing: 0.0, entropy_weight: 0.0
    }
    loss_and_grad = Loss.build_bptt_loss_and_grad_fn(predict, config)

    states = Nx.iota({@batch, @time, @embed}, type: :f32) |> Nx.divide(50.0)
    weights = Nx.put_slice(Nx.broadcast(0.0, {@batch, @time}), [0, 0], Nx.broadcast(1.0, {@batch, 1}))
    carry = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    {{loss, _}, grads} = loss_and_grad.(params, states, actions(), weights, carry)
    assert Nx.to_number(Nx.is_nan(loss)) == 0
    assert Nx.to_number(Nx.sum(Nx.as_type(Nx.is_nan(grads.data["ar_buttons_logits"]["kernel"]), :s64))) == 0
  end

  test "plain BPTT build still refuses nothing and has no event inputs" do
    {init, predict} = Axon.build(model([]), mode: :inference)
    inputs = Map.merge(tf(), %{
      "state_sequence" => Nx.broadcast(0.0, {@batch, @time, @embed}),
      "initial_hidden" => Nx.broadcast(0.0, {@batch, @layers, @hidden})
    })
    params = init.(inputs, Axon.ModelState.empty())
    {head, _hidden} = predict.(params, inputs)
    assert tuple_size(head) == 6
    assert Nx.shape(params.data["ar_buttons_logits"]["kernel"]) |> elem(1) == 8
  end
end
