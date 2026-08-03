defmodule ExPhil.Interp.AllTimestepsTrunkTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy

  @embed 32
  @window 8
  @hidden 16

  defp trunk_output_shape(extra_opts) do
    model =
      Policy.build_temporal_trunk(
        [
          embed_size: @embed,
          backbone: :gru,
          window_size: @window,
          hidden_size: @hidden,
          num_layers: 2,
          dropout: 0.0
        ] ++ extra_opts
      )

    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    params = init_fn.(Nx.template({1, @window, @embed}, :f32), Axon.ModelState.empty())
    out = predict_fn.(params, Nx.broadcast(0.0, {1, @window, @embed}))
    Nx.shape(out)
  end

  test "default trunk returns final-timestep state {batch, hidden}" do
    assert trunk_output_shape([]) == {1, @hidden}
  end

  test "return_sequences: true returns every timestep {batch, window, hidden}" do
    assert trunk_output_shape(return_sequences: true) == {1, @window, @hidden}
  end

  test "exported-style params load into the all-timesteps trunk (names match)" do
    # The final-slice layer is param-free, so a params state initialized on
    # the default trunk must apply cleanly to the all-timesteps trunk.
    default = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru,
      window_size: @window, hidden_size: @hidden, num_layers: 2, dropout: 0.0)

    seq = Policy.build_temporal_trunk(embed_size: @embed, backbone: :gru,
      window_size: @window, hidden_size: @hidden, num_layers: 2, dropout: 0.0,
      return_sequences: true)

    {init_d, _} = Axon.build(default, mode: :inference)
    params = init_d.(Nx.template({1, @window, @embed}, :f32), Axon.ModelState.empty())

    {_, predict_seq} = Axon.build(seq, mode: :inference)
    out = predict_seq.(params, Nx.broadcast(0.0, {1, @window, @embed}))
    assert Nx.shape(out) == {1, @window, @hidden}
  end
end
