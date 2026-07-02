# Minimal repro for fused selective_scan backward segfault
#
# The backward kernel used to allocate 128KB/thread on CUDA stack
# (h_prev_store[1024][32]), causing segfaults at batch_size >= 64.
# Fixed by replacing with cudaMallocAsync global memory workspace.
#
# Usage: EDIFICE_LOCAL_NX=1 EDIFICE_DISABLE_FUSED=0 mix run scripts/test_selective_scan_backward.exs

alias Edifice.CUDA.FusedScan

IO.puts("=== Selective Scan Backward Segfault Test ===\n")

hidden = 256
state_size = 16
seq_len = 60

for batch <- [4, 64, 128] do
  IO.write("  batch=#{batch} ... ")

  key = Nx.Random.key(42)
  {x, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
  {dt, key} = Nx.Random.uniform(key, 0.001, 0.1, shape: {batch, seq_len, hidden}, type: :f32)
  {a, key} = Nx.Random.uniform(key, -1.0, -0.01, shape: {hidden, state_size}, type: :f32)
  {b, key} = Nx.Random.normal(key, shape: {batch, seq_len, state_size}, type: :f32)
  {c, _key} = Nx.Random.normal(key, shape: {batch, seq_len, state_size}, type: :f32)

  # Forward + backward through fused kernel
  grad_fn =
    Nx.Defn.jit(
      fn x, dt, a, b, c ->
        Nx.Defn.value_and_grad(x, fn x ->
          FusedScan.selective_scan(x, dt, a, b, c) |> Nx.sum()
        end)
      end,
      compiler: EXLA
    )

  {loss, grad} = grad_fn.(x, dt, a, b, c)

  loss_val = Nx.to_number(loss)
  grad_finite? = grad |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0

  if grad_finite? do
    IO.puts("PASS (loss=#{Float.round(loss_val, 4)})")
  else
    IO.puts("FAIL (NaN in gradients)")
    System.halt(1)
  end
end

IO.puts("\n=== All batch sizes passed (no segfault) ===")
