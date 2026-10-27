# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
# Run through scripts/run_recurrent_validation.exs in a fresh VM.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
Code.require_file("lib/mamba_precision_experiment.exs", __DIR__)
Code.require_file("lib/mamba_carry_resolution.exs", __DIR__)
Code.require_file("lib/mamba_stage_trace.exs", __DIR__)
alias MambaCarryResolution, as: Check
alias ExPhil.Training.Output

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [capture: :string, row: :integer, mode: :string, out: :string, target: :string]
  )

data = opts |> Keyword.fetch!(:capture) |> File.read!() |> Nx.deserialize()
row = Keyword.get(opts, :row, 4)
window = Nx.to_number(data.window)
width = Nx.to_number(data.width)
layers = Nx.axis_size(data.initial.h, 1)
initial = Map.new(data.initial, fn {k, v} -> {k, Check.row(v, row)} end)
frames = Check.row(data.frames, row)
resets = Nx.slice_along_axis(data.resets, row, 1, axis: 1)
mode = Keyword.get(opts, :mode, "dense_norm")

unless mode in ["baseline", "dense_norm", "chunk8"],
  do: raise("Trace only an unmodified SSM/gate baseline")

MambaPrecisionExperiment.install(mode)
jit = fn fun -> Nx.Defn.jit(fun, compiler: EXLA, precision: :highest) end
run = jit.(&Check.trajectory/5)
target = Keyword.get(opts, :target, "output")
unless target in ["output", "conv"], do: raise("target must be output or conv")
Output.puts("Finding the worst #{target} coordinate on row #{row}")
a = run.(data.params, initial, frames, resets, window: window, width: width) |> Check.host()

b =
  run.(Check.cast(data.params, :f64), Check.cast(initial, :f64), Check.cast(frames, :f64), resets,
    window: window,
    width: width
  )
  |> Check.host()

select = fn trace ->
  if target == "conv",
    do: Nx.slice_along_axis(trace.conv, Nx.axis_size(trace.conv, 3) - 1, 1, axis: 3),
    else: trace.output
end

selected_a = select.(a)
selected_b = select.(b)

index =
  Nx.with_default_backend(Nx.BinaryBackend, fn ->
    diff = Nx.abs(Nx.subtract(Nx.as_type(selected_a, :f64), selected_b))
    ratio = Nx.divide(diff, Nx.add(1.0e-4, Nx.multiply(1.0e-4, Nx.abs(selected_b))))
    ratio |> Nx.flatten() |> Nx.argmax() |> Nx.to_number()
  end)

per_frame = div(Nx.size(selected_a), Nx.axis_size(selected_a, 0))
tick = div(index, per_frame)
channel = rem(index, if(target == "conv", do: Nx.axis_size(selected_a, 4), else: width))
position = tick + window - 1
Output.puts("Tracing real frame #{tick}, output channel #{channel}")
prefix = jit.(&MambaStageTrace.state_before/6)
state32 = prefix.(data.params, initial, frames, resets, Nx.tensor(position), window: window)

state64 =
  prefix.(
    Check.cast(data.params, :f64),
    Check.cast(initial, :f64),
    Check.cast(frames, :f64),
    resets,
    Nx.tensor(position),
    window: window
  )

frame = Nx.take(frames, position, axis: 1)
trace = jit.(&MambaStageTrace.trace/3)
t32 = trace.(data.params, state32, frame) |> Check.host()
t64 = trace.(Check.cast(data.params, :f64), state64, Check.cast(frame, :f64)) |> Check.host()

local =
  jit.(fn p, t -> MambaStageTrace.recompute(p, t, layers) end).(
    Check.cast(data.params, :f64),
    Check.cast(t32, :f64)
  )
  |> Check.host()

step = jit.(&Edifice.SSM.Mamba.step/3)
{y32, s32} = step.(data.params, state32, frame)
{y64, s64} = step.(Check.cast(data.params, :f64), state64, Check.cast(frame, :f64))

parity = %{
  f32: Check.metrics(t32["#{layers}/residual"], Check.host(y32)),
  f64: Check.metrics(t64["#{layers}/residual"], Check.host(y64))
}

parity =
  Enum.reduce([{t32, s32, "f32"}, {t64, s64, "f64"}], parity, fn {trace, state, type}, acc ->
    h = for(i <- 1..layers, do: trace["#{i}/next_h"]) |> Nx.stack(axis: 1)
    inner = Nx.axis_size(state.conv, 3)

    latest =
      for(i <- 1..layers, do: Nx.slice_along_axis(trace["#{i}/xz"], 0, inner, axis: 1))
      |> Nx.stack(axis: 1)

    expected_latest =
      state.conv |> Nx.take(Nx.axis_size(state.conv, 2) - 1, axis: 2) |> Check.host()

    acc
    |> Map.put("h_#{type}", Check.metrics(h, Check.host(state.h)))
    |> Map.put("latest_conv_#{type}", Check.metrics(latest, expected_latest))
  end)

if Enum.any?(parity, fn {_, metrics} -> metrics.failed_values > 0 end),
  do: raise("Instrumentation changed the step beyond tolerance: #{inspect(parity)}")

stages =
  Map.new(local, fn {key, value} ->
    coordinate =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        actual = Nx.as_type(t32[key], :f64) |> Nx.flatten()
        reference = Nx.flatten(t64[key])
        recomputed = Nx.flatten(value)
        diff = Nx.subtract(actual, reference)

        worst =
          Nx.divide(Nx.abs(diff), Nx.add(1.0e-4, Nx.multiply(1.0e-4, Nx.abs(reference))))
          |> Nx.argmax()
          |> Nx.to_number()

        get = fn tensor -> tensor |> Nx.take(worst) |> Nx.to_number() end

        %{
          flat_index: worst,
          actual: get.(actual),
          reference: get.(reference),
          total_error: get.(diff),
          local_error: get.(Nx.subtract(actual, recomputed)),
          propagated_error: get.(Nx.subtract(recomputed, reference))
        }
      end)

    {key,
     %{
       total: Check.metrics(t32[key], t64[key]),
       local_rounding: Check.metrics(t32[key], value),
       propagated: Check.metrics(value, t64[key]),
       at_worst_total: coordinate
     }}
  end)

report = %{
  client: EXLA.Client.default_name(),
  mode: mode,
  row: row,
  tick: tick,
  channel: channel,
  target: target,
  target_layer:
    if(target == "conv",
      do: div(rem(index, per_frame), Nx.axis_size(selected_a, 4)) + 1,
      else: layers
    ),
  target_uses_latest_conv_slot: target == "conv",
  instrumentation_vs_step: parity,
  stages: stages
}

File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true))
Output.puts("Saved #{opts[:out]}")
