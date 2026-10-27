# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
Code.require_file("../../scripts/lib/benchmark_sweep.exs", __DIR__)
Code.require_file("../../scripts/lib/mamba_precision_experiment.exs", __DIR__)
Code.require_file("../../scripts/lib/mamba_carry_resolution.exs", __DIR__)

defmodule MambaCarryResolutionTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias Edifice.SSM.Mamba

  test "independent reset masks match chronological single-row replay" do
    Nx.default_backend(Nx.BinaryBackend)
    Process.put(:__edifice_force_fallback__, true)

    opts = [
      embed_dim: 4,
      hidden_size: 4,
      num_layers: 1,
      state_size: 2,
      conv_size: 2,
      window_size: 3,
      dropout: 0.0
    ]

    frames = Nx.iota({2, 10, 4}, type: :f32) |> Nx.multiply(0.01)
    {init, _} = Axon.build(Mamba.build(opts))
    params = init.(frames, Axon.ModelState.empty())
    initial = Mamba.init_state(params, opts ++ [batch_size: 2])
    masks = [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0], [0, 0]]

    actual =
      Nx.Defn.jit(&MambaCarryResolution.trajectory/5, compiler: Nx.Defn.Evaluator).(
        params,
        initial,
        frames,
        Nx.tensor(masks, type: :u8),
        window: 3,
        width: 4
      )

    for row <- 0..1 do
      zero = Mamba.init_state(params, opts ++ [batch_size: 1])
      input = Nx.slice_along_axis(frames, row, 1, axis: 0)

      {records, final} =
        Enum.map_reduce(0..9, zero, fn tick, carry ->
          carry =
            if Enum.at(Enum.at(masks, tick), row) == 1 do
              Enum.reduce((tick - 2)..(tick - 1), zero, fn i, state ->
                elem(Mamba.step(params, state, Nx.take(input, i, axis: 1)), 1)
              end)
            else
              carry
            end

          {output, carry} = Mamba.step(params, carry, Nx.take(input, tick, axis: 1))
          {{output, carry.conv}, carry}
        end)

      records = Enum.drop(records, 2)
      expected_output = records |> Enum.map(&elem(&1, 0)) |> Nx.stack()
      expected_conv = records |> Enum.map(&elem(&1, 1)) |> Nx.stack()

      assert_all_close(Nx.slice_along_axis(actual.output, row, 1, axis: 1), expected_output,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(Nx.slice_along_axis(actual.conv, row, 1, axis: 1), expected_conv,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(Nx.slice_along_axis(actual.h, row, 1, axis: 0), final.h,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end
  end

  test "the metric rejects a small absolute difference beyond its relative bound" do
    result = MambaCarryResolution.metrics(Nx.tensor([0.0002, 1000.01]), Nx.tensor([0.0, 1000.0]))
    assert result.failed_values == 1
    assert result.max_error_over_bound > 1
  end
end
