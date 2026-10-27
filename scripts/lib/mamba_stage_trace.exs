# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
defmodule MambaStageTrace do
  import Nx.Defn
  alias Edifice.Stateful.Ops
  alias Edifice.SSM.Common

  defn state_before(params, initial, frames, resets, stop, opts) do
    opts = keyword!(opts, [:window])
    validate_single_row(frames)

    {_, state, _, _, _, _, _} =
      while {i = 0, state = initial, params, initial, frames, resets, stop}, i < stop do
        state = reset_if_due(params, state, initial, frames, resets, i, opts)
        {_, state} = advance(params, state, Nx.take(frames, i, axis: 1))
        {i + 1, state, params, initial, frames, resets, stop}
      end

    reset_if_due(params, state, initial, frames, resets, stop, opts)
  end

  deftransformp validate_single_row(frames) do
    unless Nx.axis_size(frames, 0) == 1,
      do: raise(ArgumentError, "Select one row before tracing its reset history")
  end

  defnp reset_if_due(params, state, initial, frames, resets, i, opts) do
    if Nx.any(Nx.take(resets, i)) do
      history = Nx.slice_along_axis(frames, i - opts[:window] + 1, opts[:window] - 1, axis: 1)
      Edifice.Stateful.Replay.run(params, initial, history, module: Edifice.SSM.Mamba)
    else
      state
    end
  end

  deftransformp(advance(params, state, frame), do: Edifice.SSM.Mamba.step(params, state, frame))

  # Exports intermediates for one frame. The runner checks this instrumented
  # calculation against the ordinary compiled step before interpreting it.
  def trace(params, state, frame) do
    params = Ops.unwrap_params(params)

    input =
      if params["input_projection"], do: Ops.dense(frame, params["input_projection"]), else: frame

    {_, traces} =
      Enum.reduce(
        1..Nx.axis_size(state.h, 1),
        {input, %{"frame" => frame, "input_projection" => input}},
        fn i, {input, traces} ->
          name = "mamba_block_#{i}"
          h = state.h |> Nx.slice_along_axis(i - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          conv = state.conv |> Nx.slice_along_axis(i - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          norm = Ops.layer_norm(input, params["#{name}_norm"])
          xz = Ops.dense(norm, params["#{name}_in_proj"])
          inner = div(Nx.axis_size(xz, 1), 2)
          x = Nx.slice_along_axis(xz, 0, inner, axis: 1)
          z = Nx.slice_along_axis(xz, inner, inner, axis: 1)
          {convolved, _} = Ops.conv1d_step(x, conv, params["#{name}_conv_dw_conv"])
          activated = Ops.silu(convolved)
          bc = Ops.dense(activated, params["#{name}_ssm_bc_proj"])
          size = Nx.axis_size(h, 2)
          b = Nx.slice_along_axis(bc, 0, size, axis: 1)
          c = Nx.slice_along_axis(bc, size, size, axis: 1)
          rank = Ops.dense(activated, params["#{name}_ssm_dt_rank"])
          raw_dt = Ops.dense(rank, params["#{name}_ssm_dt_proj"])
          dt = Ops.softplus(raw_dt)
          {y, next_h} = Common.step_discretized_ssm(activated, b, c, dt, h)
          gate = Ops.silu(z)
          gated = Nx.multiply(y, gate)
          out = Ops.dense(gated, params["#{name}_out_proj"])
          residual = Nx.add(input, out)

          values = %{
            "input" => input,
            "h" => h,
            "conv" => conv,
            "norm" => norm,
            "xz" => xz,
            "x" => x,
            "z" => z,
            "convolved" => convolved,
            "activated" => activated,
            "bc" => bc,
            "b" => b,
            "c" => c,
            "dt_rank" => rank,
            "raw_dt" => raw_dt,
            "dt" => dt,
            "next_h" => next_h,
            "readout" => y,
            "gate" => gate,
            "gated" => gated,
            "out" => out,
            "residual" => residual
          }

          traces =
            Enum.reduce(values, traces, fn {key, value}, acc ->
              Map.put(acc, "#{i}/#{key}", value)
            end)

          {residual, traces}
        end
      )

    traces
  end

  # Each operation receives exactly the inputs from the f32 trace, promoted to
  # f64. This separates its own rounding from errors already in those inputs.
  def recompute(params, traces, layers) do
    params = Ops.unwrap_params(params)

    input =
      if params["input_projection"],
        do: Ops.dense(traces["frame"], params["input_projection"]),
        else: traces["frame"]

    Enum.reduce(1..layers, %{"input_projection" => input}, fn i, result ->
      name = "mamba_block_#{i}"
      t = fn key -> Map.fetch!(traces, "#{i}/#{key}") end
      {convolved, _} = Ops.conv1d_step(t.("x"), t.("conv"), params["#{name}_conv_dw_conv"])

      {_, next_h} =
        Common.step_discretized_ssm(t.("activated"), t.("b"), t.("c"), t.("dt"), t.("h"))

      values = %{
        "norm" => Ops.layer_norm(t.("input"), params["#{name}_norm"]),
        "xz" => Ops.dense(t.("norm"), params["#{name}_in_proj"]),
        "convolved" => convolved,
        "activated" => Ops.silu(t.("convolved")),
        "bc" => Ops.dense(t.("activated"), params["#{name}_ssm_bc_proj"]),
        "dt_rank" => Ops.dense(t.("activated"), params["#{name}_ssm_dt_rank"]),
        "raw_dt" => Ops.dense(t.("dt_rank"), params["#{name}_ssm_dt_proj"]),
        "dt" => Ops.softplus(t.("raw_dt")),
        "next_h" => next_h,
        "readout" => Nx.sum(Nx.multiply(Nx.new_axis(t.("c"), 1), t.("next_h")), axes: [2]),
        "gate" => Ops.silu(t.("z")),
        "gated" => Nx.multiply(t.("readout"), t.("gate")),
        "out" => Ops.dense(t.("gated"), params["#{name}_out_proj"]),
        "residual" => Nx.add(t.("input"), t.("out"))
      }

      Enum.reduce(values, result, fn {key, value}, acc -> Map.put(acc, "#{i}/#{key}", value) end)
    end)
  end
end
