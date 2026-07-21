# Mamba-2 SSD training-speed microbench (round-velocity work, 2026-07-21).
#
#   mix run --no-compile scripts/bench_ssd_scan.exs [--iters 30] [--batch 64]
#
# Motivation: mamba_2 trains ~1.75x slower per epoch than mamba-1
# (2.12h vs ~1.2h per 30ep on the 54-game pool). Mamba-1's speed comes
# from the fused CUDA selective-scan kernel (Blelloch fallback); SSD's
# training path (ssd_matmul_chunk_impl) materializes a
# {batch, chunk, chunk, hidden*expand, state} transfer tensor —
# ~2.1 GB per materialization at drill shapes with chunk 32.
#
# Matrix (same recurrence h[t] = a[t] h[t-1] + bx[t] in all variants —
# ONLY the algorithm differs):
#   ssd_c32 / ssd_c16 / ssd_c8   MambaSSD matmul path, chunk-size sweep
#                                 (transfer tensor shrinks quadratically)
#   ssd_seq                       MambaSSD non-matmul path
#                                 (Common.sequential_scan via
#                                 training_mode: false)
#   mamba1                        Edifice.SSM.Mamba (fused kernel /
#                                 Blelloch) — the reference to beat
#
# Times forward AND forward+backward (value_and_grad of mean output —
# training cost is dominated by backward). JIT-warmed before timing;
# median + p10/p90 reported. Decision rule: if the best SSD variant gets
# within ~1.2x of mamba1 fwd+bwd, pure-Nx wins and no NIF is needed;
# otherwise the edifice SSD kernel on the Nx.block/CustomCall protocol
# (task #13 spike) is justified.
#
# GPU-INTENSIVE: run only when no training is in flight (#67).

alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [iters: :integer, batch: :integer, variants: :string]
  )

iters = opts[:iters] || 30
batch = opts[:batch] || 64

# Drill shapes (mirror build_mamba2_backbone defaults)
embed = 288
window = 60
base = [
  embed_dim: embed,
  hidden_size: 256,
  state_size: 16,
  expand_factor: 2,
  num_layers: 2,
  dropout: 0.0,
  window_size: window
]

variants =
  case opts[:variants] do
    nil ->
      [
        {"ssd_c32", Edifice.SSM.MambaSSD, base ++ [chunk_size: 32, training_mode: true]},
        {"ssd_c16", Edifice.SSM.MambaSSD, base ++ [chunk_size: 16, training_mode: true]},
        {"ssd_c8", Edifice.SSM.MambaSSD, base ++ [chunk_size: 8, training_mode: true]},
        {"ssd_seq", Edifice.SSM.MambaSSD, base ++ [chunk_size: 32, training_mode: false]},
        {"mamba1", Edifice.SSM.Mamba, base ++ [conv_size: 4]}
      ]

    s ->
      wanted = String.split(s, ",", trim: true)

      [
        {"ssd_c32", Edifice.SSM.MambaSSD, base ++ [chunk_size: 32, training_mode: true]},
        {"ssd_c16", Edifice.SSM.MambaSSD, base ++ [chunk_size: 16, training_mode: true]},
        {"ssd_c8", Edifice.SSM.MambaSSD, base ++ [chunk_size: 8, training_mode: true]},
        {"ssd_seq", Edifice.SSM.MambaSSD, base ++ [chunk_size: 32, training_mode: false]},
        {"mamba1", Edifice.SSM.Mamba, base ++ [conv_size: 4]}
      ]
      |> Enum.filter(fn {name, _, _} -> name in wanted end)
  end

Output.banner("SSD scan microbench")
Output.config([
  {"Shapes", "batch=#{batch} seq=#{window} embed=#{embed} hidden=256 (d_inner 512) state=16 layers=2"},
  {"Iters", iters},
  {"Variants", Enum.map(variants, &elem(&1, 0))}
])

key = Nx.Random.key(42)
{x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {batch, window, embed}, type: :f32)
x = Nx.backend_transfer(x, EXLA.Backend)

stats = fn times ->
  sorted = Enum.sort(times)
  n = length(sorted)
  at = fn p -> Enum.at(sorted, min(trunc(p * n), n - 1)) end
  {at.(0.5), at.(0.1), at.(0.9)}
end

time_ms = fn fun ->
  t0 = System.monotonic_time(:microsecond)
  fun.()
  (System.monotonic_time(:microsecond) - t0) / 1000
end

results =
  Enum.map(variants, fn {name, mod, build_opts} ->
    Output.puts("")
    Output.puts("#{name}: building...")

    model = mod.build(build_opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :train, compiler: EXLA)
    params = init_fn.(Nx.template({batch, window, embed}, :f32), Axon.ModelState.empty())

    fwd = fn p, input ->
      out = predict_fn.(p, input)
      # mode: :train returns %{prediction: ...}
      pred = if is_map(out) and Map.has_key?(out, :prediction), do: out.prediction, else: out
      Nx.mean(pred)
    end

    grad_fn =
      Nx.Defn.jit(
        fn p, input -> Nx.Defn.value_and_grad(fn p2 -> fwd.(p2, input) end).(p) end,
        compiler: EXLA,
        on_conflict: :reuse
      )

    # Warmup (JIT compile both paths)
    Output.puts("#{name}: JIT warmup...")
    _ = fwd.(params, x) |> Nx.to_number()
    {_l, _g} = grad_fn.(params, x)

    fwd_times =
      for _ <- 1..iters do
        time_ms.(fn -> fwd.(params, x) |> Nx.to_number() end)
      end

    bwd_times =
      for _ <- 1..iters do
        time_ms.(fn ->
          {l, _g} = grad_fn.(params, x)
          Nx.to_number(l)
        end)
      end

    {f50, f10, f90} = stats.(fwd_times)
    {b50, b10, b90} = stats.(bwd_times)

    Output.puts(
      "#{name}: fwd #{Float.round(f50, 1)}ms [#{Float.round(f10, 1)}-#{Float.round(f90, 1)}]  " <>
        "fwd+bwd #{Float.round(b50, 1)}ms [#{Float.round(b10, 1)}-#{Float.round(b90, 1)}]"
    )

    %{variant: name, fwd_ms: f50, fwd_p10: f10, fwd_p90: f90, bwd_ms: b50, bwd_p10: b10, bwd_p90: b90}
  end)

Output.puts("")
Output.puts("=== summary (fwd+bwd median, ratio vs mamba1) ===")
ref = Enum.find(results, &(&1.variant == "mamba1"))

Enum.each(results, fn r ->
  ratio = if ref, do: " (#{Float.round(r.bwd_ms / ref.bwd_ms, 2)}x)", else: ""
  Output.puts("  #{String.pad_trailing(r.variant, 10)} #{Float.round(r.bwd_ms, 1)}ms#{ratio}")
end)

out = "logs/bench_ssd_scan_#{Date.utc_today() |> Date.to_iso8601(:basic)}.json"
File.write!(out, Jason.encode!(%{batch: batch, window: window, iters: iters, results: results}, pretty: true))
Output.success("results -> #{out}")
