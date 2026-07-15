# NaN autopsy: replay the captured fatal training step offline and bisect.
#
#   mix run scripts/replay_crime_scene.exs [--scene PATH]
#
# The scene (saved by the grad detector, task #25) holds the exact batch,
# pre-step params, and optimizer state of the step whose backward produced
# non-finite buttons-head gradients with a FINITE forward loss. Each
# experiment below re-runs that single backward under a different
# condition — seconds per hypothesis instead of a 40-90 min approach run.
#
# Experiments:
#   A. exact repro   (bf16 params/states, frame-weighted loss — as trained)
#   B. all-f32       (params + states cast to f32: precision-dependent?)
#   C. no frame-weights (standard mean path: weighted-path-dependent?)
#   D. logit stats   (are buttons logits extreme? which rows?)

alias ExPhil.Networks
alias ExPhil.Networks.Policy
alias ExPhil.Training.Utils
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [scene: :string])

scene_path =
  opts[:scene] ||
    "interp/crime_scenes/" |> File.ls!() |> Enum.sort() |> List.last() |> then(&("interp/crime_scenes/" <> &1))

Output.banner("NaN crime-scene autopsy")
Output.puts("scene: #{scene_path}")

scene = scene_path |> File.read!() |> :erlang.binary_to_term()
config = scene.config

Output.puts("step=#{scene.step} recorded_loss=#{inspect(scene.loss)}")

model =
  Networks.Policy.build_temporal(
    embed_size: config[:embed_size],
    backbone: config[:backbone],
    window_size: config[:window_size],
    num_heads: config[:num_heads] || 4,
    head_dim: config[:head_dim] || 64,
    hidden_size: config[:hidden_size] || 256,
    num_layers: config[:num_layers] || 2,
    state_size: config[:state_size] || 16,
    expand_factor: config[:expand_factor] || 2,
    conv_size: config[:conv_size] || 4,
    dropout: config[:dropout] || 0.0,
    axis_buckets: config[:axis_buckets] || 16,
    shoulder_buckets: config[:shoulder_buckets] || 4
  )

{_init_fn, predict_fn} = Utils.build_compiled(model)

states = scene.batch.states
actions = scene.batch.actions
frame_weights = scene.batch.frame_weights
params = Utils.ensure_model_state(scene.params)

grad_report = fn label, params_in, states_in, fw ->
  loss_fn = fn p ->
    {buttons, main_x, main_y, c_x, c_y, shoulder} =
      predict_fn.(Utils.ensure_model_state(p), states_in)

    logits = %{buttons: buttons, main_x: main_x, main_y: main_y, c_x: c_x, c_y: c_y, shoulder: shoulder}

    opts = [
      label_smoothing: config[:label_smoothing] || 0.0,
      button_weight: config[:button_weight] || 1.0
    ]

    opts = if fw, do: Keyword.put(opts, :frame_weights, fw), else: opts
    Policy.imitation_loss(logits, actions, opts)
  end

  {loss, grads} = Nx.Defn.jit_apply(
    fn p -> Nx.Defn.value_and_grad(p, loss_fn) end,
    [params_in],
    compiler: EXLA
  )

  loss_num = Nx.to_number(loss)

  per_layer =
    grads
    |> Map.get(:data)
    |> Enum.map(fn {layer, ps} ->
      maxes =
        ps
        |> Map.values()
        |> Enum.filter(&is_struct(&1, Nx.Tensor))
        |> Enum.map(fn t -> Nx.abs(t) |> Nx.reduce_max() |> Nx.to_number() end)

      {layer, if(Enum.all?(maxes, &is_number/1), do: Float.round(Enum.max([0.0 | maxes]), 6), else: :NONFINITE)}
    end)
    |> Enum.sort()

  bad = for {l, :NONFINITE} <- per_layer, do: l
  Output.puts("#{label}: loss=#{inspect(loss_num)} nonfinite=#{inspect(bad)}")
  if bad == [], do: Output.puts("  per-layer max: #{inspect(per_layer, limit: :infinity)}")
  bad
end

# A. exact repro (as trained: bf16)
Output.puts("")
bad_a = grad_report.("A exact-repro (bf16)", params, states, frame_weights)

# B. everything f32
params_f32 =
  update_in(params.data, fn data ->
    walk = fn walk, v ->
      cond do
        is_struct(v, Nx.Tensor) -> if elem(Nx.type(v), 0) in [:f, :bf], do: Nx.as_type(v, :f32), else: v
        is_map(v) and not is_struct(v) -> Map.new(v, fn {k, x} -> {k, walk.(walk, x)} end)
        true -> v
      end
    end

    walk.(walk, data)
  end)

bad_b = grad_report.("B all-f32", params_f32, Nx.as_type(states, :f32), frame_weights)

# C. no frame weights (standard mean path), original precision
bad_c = grad_report.("C no-frame-weights (bf16)", params, states, nil)

# D. logit stats on the fatal batch
{buttons, _mx, _my, _cx, _cy, _sh} = predict_fn.(params, states)
bmax = Nx.reduce_max(buttons) |> Nx.to_number()
bmin = Nx.reduce_min(buttons) |> Nx.to_number()
row_max = Nx.reduce_max(Nx.abs(buttons), axes: [1])
worst = Nx.argmax(row_max) |> Nx.to_number()
Output.puts("")
Output.puts("D buttons logits: min=#{inspect(bmin)} max=#{inspect(bmax)} worst_row=#{worst}")

Output.puts("")
Output.puts("READING: A nonfinite + B clean => precision-dependent (bf16 backward op).")
Output.puts("A + B both nonfinite => data/params-driven at any precision (op hunt in f32).")
Output.puts("A nonfinite + C clean => the frame-weighted per-sample path is the trigger.")
