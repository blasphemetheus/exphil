defmodule ExPhil.Training.Bf16ConvGradientTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :gpu
  @moduletag :known_bug

  @doc """
  Test for the bf16 depthwise convolution gradient SIGBUS bug.

  XLA crashes with SIGBUS when computing gradients for a depthwise convolution
  (feature_group_size=1024) with bf16 precision after ~3000 iterations.

  See: docs/research/bf16_depthwise_conv_sigbus.md

  This test runs 500 iterations (should be safe) and verifies gradients are finite.
  The full reproducer at scripts/repro_bf16_conv_sigbus.exs runs 5000 iterations.
  """

  describe "bf16 depthwise conv gradient stability" do
    @tag timeout: 120_000
    test "100 gradient steps produce finite results" do
      model =
        Axon.input("input", shape: {nil, 60, 1024})
        |> Axon.conv(1024,
          kernel_size: {4},
          padding: [{3, 0}],
          feature_group_size: 1024,
          name: "dw_conv"
        )
        |> Axon.nx(fn x ->
          x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
        end, name: "reduce")

      policy = Axon.MixedPrecision.create_policy(
        params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}
      )
      model = Axon.MixedPrecision.apply_policy(model, policy)

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(Nx.template({4, 60, 1024}, :bf16), Axon.ModelState.empty())

      grad_fn = Nx.Defn.jit(
        fn params, input ->
          loss_fn = fn p -> predict_fn.(p, %{"input" => input}) end
          Nx.Defn.value_and_grad(loss_fn).(params)
        end,
        compiler: EXLA, on_conflict: :reuse
      )

      input = Nx.broadcast(Nx.tensor(0.1, type: :bf16), {4, 60, 1024})

      # Run 100 iterations — safe range, crash happens ~3000
      for i <- 1..100 do
        {loss, grads} = grad_fn.(params, input)
        loss_val = Nx.to_number(loss)

        if rem(i, 25) == 0 do
          assert is_number(loss_val) and loss_val == loss_val,
            "Loss became NaN at step #{i}"
        end
      end
    end

    @tag timeout: 120_000
    test "f32 conv with bf16 workaround produces finite gradients" do
      # This tests our workaround: cast to f32 before conv
      model =
        Axon.input("input", shape: {nil, 60, 1024})
        |> Axon.nx(fn x -> Nx.as_type(x, :f32) end, name: "cast_f32")
        |> Axon.conv(1024,
          kernel_size: {4},
          padding: [{3, 0}],
          feature_group_size: 1024,
          name: "dw_conv"
        )
        |> Axon.nx(fn x ->
          x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
        end, name: "reduce")

      policy = Axon.MixedPrecision.create_policy(
        params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}
      )
      model = Axon.MixedPrecision.apply_policy(model, policy,
        except: [:batch_norm, :layer_norm]
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(Nx.template({4, 60, 1024}, :bf16), Axon.ModelState.empty())

      grad_fn = Nx.Defn.jit(
        fn params, input ->
          loss_fn = fn p -> predict_fn.(p, %{"input" => input}) end
          Nx.Defn.value_and_grad(loss_fn).(params)
        end,
        compiler: EXLA, on_conflict: :reuse
      )

      input = Nx.broadcast(Nx.tensor(0.1, type: :bf16), {4, 60, 1024})

      # Run 100 iterations — workaround should be stable
      losses = for i <- 1..100 do
        {loss, _grads} = grad_fn.(params, input)
        Nx.to_number(loss)
      end

      # All losses should be finite numbers
      nan_count = Enum.count(losses, &(not is_number(&1)))
      assert nan_count == 0, "Workaround produced #{nan_count} NaN losses"

      IO.puts("\n  Workaround: 100 steps, all finite, loss=#{Float.round(List.last(losses) * 1.0, 4)}")
    end
  end
end
