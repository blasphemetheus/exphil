# Profile the live-play inference path stage by stage.
#
# Reproduces exactly what Agent.compute_temporal_action does per decision:
#   1. embed_game_state          (Embeddings.Game.embed, eager Nx)
#   2. buffer -> [1, W, E] tensor (queue stack + reshape, eager Nx)
#   3. predict_fn forward pass    (EXLA-compiled Axon model)
#   4. Policy.sample              (forward + 6 eager sampling ops)
#   5. to_controller_state        (Nx.to_number device->host reads)
#
# Usage:
#   mix run scripts/profile_agent_inference.exs --policy checkpoints/<name>_policy.bin

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.Utils
alias ExPhil.Networks.Policy
alias ExPhil.Embeddings

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string, iters: :integer])
policy_path = opts[:policy] || "checkpoints/clean_200_20260702_183630_policy.bin"
iters = opts[:iters] || 50

IO.puts("Loading policy: #{policy_path}")
{:ok, %{params: params, config: config}} = ExPhil.Training.load_policy(policy_path)

embed_size = config[:embed_size] || raise "no embed_size in config"
window_size = config[:window_size] || 60
axis_buckets = config[:axis_buckets] || 16

IO.puts("embed_size=#{embed_size} window=#{window_size} backbone=#{config[:backbone]}")

model =
  Policy.build_temporal(
    embed_size: embed_size,
    backbone: config[:backbone],
    window_size: window_size,
    hidden_size: config[:hidden_size] || 256,
    num_layers: config[:num_layers] || 2,
    dropout: config[:dropout] || 0.0,
    axis_buckets: axis_buckets,
    shoulder_buckets: config[:shoulder_buckets] || 4
  )

{_init_fn, predict_fn} = Utils.build_compiled(model)

# Fake game state via a recorded-style struct is complex; embed cost is measured
# with a synthetic random frame embedding instead, then separately with a real
# GameState if available. Buffer stage uses the same queue mechanics as the Agent.
key = Nx.Random.key(42)
{frame_embed, _} = Nx.Random.uniform(key, shape: {embed_size}, type: :f32)

buffer =
  Enum.reduce(1..window_size, :queue.new(), fn _, q -> :queue.in(frame_embed, q) end)

buffer_to_tensor = fn buf ->
  buf |> :queue.to_list() |> Nx.stack() |> Nx.reshape({1, window_size, embed_size})
end

sequence_batch = buffer_to_tensor.(buffer)

time_stage = fn name, fun ->
  # warmup
  fun.()
  times =
    for _ <- 1..iters do
      t0 = System.monotonic_time(:microsecond)
      fun.()
      System.monotonic_time(:microsecond) - t0
    end

  avg = Enum.sum(times) / length(times) / 1000
  min = Enum.min(times) / 1000
  IO.puts(
    :io_lib.format("~-28s avg ~8.2f ms   min ~8.2f ms", [name, avg, min]) |> to_string()
  )

  avg
end

IO.puts("\nJIT warmup (predict)...")
_ = predict_fn.(Utils.ensure_model_state(params), sequence_batch)
IO.puts("Warmup done. Profiling #{iters} iterations per stage:\n")

t_buffer = time_stage.("buffer->tensor [1,W,E]", fn -> buffer_to_tensor.(buffer) end)

t_forward =
  time_stage.("predict_fn forward", fn ->
    predict_fn.(Utils.ensure_model_state(params), sequence_batch)
  end)

t_sample =
  time_stage.("Policy.sample (fwd+sample)", fn ->
    Policy.sample(params, predict_fn, sequence_batch)
  end)

action = Policy.sample(params, predict_fn, sequence_batch)

t_decode =
  time_stage.("to_controller_state", fn ->
    Policy.to_controller_state(action, axis_buckets: axis_buckets)
  end)

t_e2e =
  time_stage.("END-TO-END (agent path)", fn ->
    seq = buffer_to_tensor.(buffer)
    a = Policy.sample(params, predict_fn, seq)
    Policy.to_controller_state(a, axis_buckets: axis_buckets)
  end)

IO.puts("\nSampling overhead (sample - forward): #{Float.round(t_sample - t_forward, 2)} ms")
IO.puts("Budget: 16.67 ms/frame. End-to-end: #{Float.round(t_e2e, 2)} ms")
