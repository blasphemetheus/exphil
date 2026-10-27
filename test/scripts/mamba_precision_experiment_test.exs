# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
Code.require_file("../../scripts/lib/mamba_precision_experiment.exs", __DIR__)
Code.require_file("../../scripts/lib/mamba_stage_trace.exs", __DIR__)

defmodule MambaPrecisionExperimentTest do
  # The diagnostic deliberately replaces modules in this VM. Restore them
  # between cases and do not overlap any other tests.
  use ExUnit.Case, async: false
  import Nx.Testing
  alias Edifice.SSM.{Common, Mamba}

  setup do
    Nx.default_backend(Nx.BinaryBackend)
    Process.put(:__edifice_force_fallback__, true)

    on_exit(fn ->
      Code.compile_file(Path.expand("../edifice/lib/edifice/stateful/ops.ex"))
      Code.compile_file(Path.expand("../edifice/lib/edifice/ssm/common.ex"))
    end)

    :ok
  end

  defp cast(value, type), do: Nx.Defn.Composite.traverse(value, &Nx.as_type(&1, type))

  for stage <- ["ssm", "update", "readout", "gate", "output", "ssm+output"] do
    test "#{stage}: preserves f64 equations, f32 output and carried-state types" do
      stage = unquote(stage)

      opts = [
        embed_dim: 4,
        hidden_size: 4,
        num_layers: 2,
        state_size: 3,
        conv_size: 3,
        window_size: 5,
        dropout: 0.0
      ]

      x = Nx.iota({2, 5, 4}, type: :f32) |> Nx.multiply(0.02)
      {init, _} = Axon.build(Mamba.build(opts))
      params = init.(x, Axon.ModelState.empty())
      zero = Mamba.init_state(params, opts ++ [batch_size: 2])
      {_, carry} = Mamba.step(params, zero, Nx.take(x, 0, axis: 1))
      frame = Nx.take(x, 1, axis: 1)

      {reference, reference_state} =
        Mamba.step(cast(params, :f64), cast(carry, :f64), cast(frame, :f64))

      MambaPrecisionExperiment.install("baseline+" <> stage)

      {actual, actual_state} =
        Mamba.step(cast(params, :f64), cast(carry, :f64), cast(frame, :f64))

      assert_all_close(actual, reference, atol: 1.0e-12, rtol: 1.0e-12)

      for key <- [:h, :conv],
          do:
            assert_all_close(actual_state[key], reference_state[key],
              atol: 1.0e-12,
              rtol: 1.0e-12
            )

      {output, next} = Mamba.step(params, carry, frame)
      assert Nx.type(output) == {:f, 32}
      assert Nx.type(next.h) == {:f, 32}
      assert Nx.type(next.conv) == {:f, 32}
      assert Nx.shape(next.h) == Nx.shape(carry.h)
      assert Nx.shape(next.conv) == Nx.shape(carry.conv)

      if stage in ["ssm", "ssm+output"] do
        wide_carry = %{carry | h: Nx.as_type(carry.h, :f64)}
        {wide_output, wide_next} = Mamba.step(params, wide_carry, frame)
        assert Nx.type(wide_output) == {:f, 32}
        assert Nx.type(wide_next.h) == {:f, 64}
        assert Nx.type(wide_next.conv) == {:f, 32}
      end
    end
  end

  test "readout64 removes cancellation loss without changing the update" do
    x = Nx.tensor([[0.0]])
    b = Nx.tensor([[0.0, 0.0, 0.0]])
    c = Nx.tensor([[1.0000001192092896, 1.0, 1.0]])
    dt = Nx.tensor([[0.0]])
    h = Nx.tensor([[[1.0e8, 1.0, -1.0e8]]])
    {original_y, expected_h} = Common.step_discretized_ssm(x, b, c, dt, h)
    MambaPrecisionExperiment.install("baseline+readout")
    {y, actual_h} = Common.step_discretized_ssm(x, b, c, dt, h)

    expected_y =
      expected_h
      |> Nx.as_type(:f64)
      |> Nx.multiply(Nx.new_axis(Nx.as_type(c, :f64), 1))
      |> Nx.sum(axes: [2])
      |> Nx.as_type(:f32)

    assert_equal(actual_h, expected_h)
    assert_equal(y, expected_y)
    refute Nx.to_number(Nx.all(Nx.equal(original_y, expected_y))) == 1
  end

  test "trace and prefix replay match the ordinary step, including a reset at the target" do
    opts = [
      embed_dim: 4,
      hidden_size: 4,
      num_layers: 2,
      state_size: 3,
      conv_size: 3,
      window_size: 3,
      dropout: 0.0
    ]

    frames = Nx.iota({1, 7, 4}, type: :f32) |> Nx.multiply(0.02)
    {init, _} = Axon.build(Mamba.build(opts))
    params = init.(frames, Axon.ModelState.empty())
    initial = Mamba.init_state(params, opts)
    resets = Nx.tensor([[0], [0], [0], [0], [1], [0], [0]], type: :u8)
    prefix = Nx.Defn.jit(&MambaStageTrace.state_before/6, compiler: Nx.Defn.Evaluator)

    for {stop, history} <- [{3, 0..2}, {4, 2..3}] do
      expected =
        Enum.reduce(history, initial, fn i, state ->
          elem(Mamba.step(params, state, Nx.take(frames, i, axis: 1)), 1)
        end)

      actual = prefix.(params, initial, frames, resets, Nx.tensor(stop), window: 3)

      for key <- [:conv, :h],
          do: assert_all_close(actual[key], expected[key], atol: 1.0e-6, rtol: 1.0e-6)

      for type <- [:f32, :f64] do
        p = cast(params, type)
        s = cast(actual, type)
        x = Nx.take(frames, stop, axis: 1) |> Nx.as_type(type)
        {out, _} = Mamba.step(p, s, x)
        trace = MambaStageTrace.trace(p, s, x)
        assert_all_close(trace["2/residual"], out, atol: 1.0e-6, rtol: 1.0e-6)
        recomputed = MambaStageTrace.recompute(p, trace, 2)

        for {key, value} <- recomputed,
            do: assert_all_close(value, trace[key], atol: 1.0e-6, rtol: 1.0e-6)
      end
    end
  end
end
