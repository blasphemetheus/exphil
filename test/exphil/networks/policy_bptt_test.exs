defmodule ExPhil.Networks.PolicyBpttTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy

  @batch 2
  @time 5
  @embed 32
  @hidden 24
  @layers 2

  defp templates(head) do
    base = %{
      "state_sequence" => Nx.template({@batch, @time, @embed}, :f32),
      "initial_hidden" => Nx.template({@batch, @layers, @hidden}, :f32)
    }

    case head do
      :independent ->
        base

      :autoregressive ->
        Map.merge(base, %{
          "tf_buttons" => Nx.template({@batch, @time, 8}, :f32),
          "tf_main_x" => Nx.template({@batch, @time}, :s64),
          "tf_main_y" => Nx.template({@batch, @time}, :s64),
          "tf_c_x" => Nx.template({@batch, @time}, :s64),
          "tf_c_y" => Nx.template({@batch, @time}, :s64)
        })
    end
  end

  defp inputs(head) do
    base = %{
      "state_sequence" => Nx.iota({@batch, @time, @embed}, type: :f32) |> Nx.divide(100),
      "initial_hidden" => Nx.broadcast(0.0, {@batch, @layers, @hidden})
    }

    case head do
      :independent ->
        base

      :autoregressive ->
        Map.merge(base, %{
          "tf_buttons" => Nx.broadcast(0.0, {@batch, @time, 8}),
          "tf_main_x" => Nx.broadcast(Nx.tensor(8, type: :s64), {@batch, @time}),
          "tf_main_y" => Nx.broadcast(Nx.tensor(8, type: :s64), {@batch, @time}),
          "tf_c_x" => Nx.broadcast(Nx.tensor(8, type: :s64), {@batch, @time}),
          "tf_c_y" => Nx.broadcast(Nx.tensor(8, type: :s64), {@batch, @time})
        })
    end
  end

  for head <- [:autoregressive, :independent] do
    test "build_temporal_bptt/#{head}: per-timestep logits + carried hidden shapes" do
      head = unquote(head)

      model =
        Policy.build_temporal_bptt(
          embed_size: @embed,
          backbone: :gru,
          hidden_size: @hidden,
          num_layers: @layers,
          window_size: @time,
          head: head,
          axis_buckets: 16,
          shoulder_buckets: 4
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates(head), Axon.ModelState.empty())

      {{buttons, main_x, main_y, c_x, c_y, shoulder}, hidden} =
        predict_fn.(params, inputs(head))

      assert Nx.shape(buttons) == {@batch, @time, 8}
      assert Nx.shape(main_x) == {@batch, @time, 17}
      assert Nx.shape(main_y) == {@batch, @time, 17}
      assert Nx.shape(c_x) == {@batch, @time, 17}
      assert Nx.shape(c_y) == {@batch, @time, 17}
      assert Nx.shape(shoulder) == {@batch, @time, 5}
      assert Nx.shape(hidden) == {@batch, @layers, @hidden}
    end
  end

  test "carried hidden feeds back and changes logits (state flows)" do
    model =
      Policy.build_temporal_bptt(
        embed_size: @embed,
        backbone: :gru,
        hidden_size: @hidden,
        num_layers: @layers,
        window_size: @time,
        head: :independent
      )

    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    params = init_fn.(templates(:independent), Axon.ModelState.empty())

    ins = inputs(:independent)
    {{b0, _, _, _, _, _}, h1} = predict_fn.(params, ins)

    {{b1, _, _, _, _, _}, _h2} =
      predict_fn.(params, %{ins | "initial_hidden" => h1})

    refute Nx.all_close(b0, b1, atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "non-gru backbone raises" do
    assert_raise ArgumentError, ~r/gru backbone only/, fn ->
      Policy.build_temporal_bptt(embed_size: @embed, backbone: :mamba)
    end
  end
end
