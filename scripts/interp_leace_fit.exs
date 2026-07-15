# Interp-surgical (task #18): fit a closed-form LEACE eraser for the
# COPY SIGNAL — the trunk subspace predicting the policy's previous
# buttons — and evaluate the erased policy offline against the intact one
# on the causal-confusion metrics.
#
#   mix run scripts/interp_leace_fit.exs [--policy PATH]
#
# Success bar: erased reproduces the ablation's improvements (shield
# holdability collapses, re-press probability after A rises = fair
# follow-ups unblocked) while stick heads stay sane — surgical, where
# ablation is amputation.
#
# Button order: a=0 b=1 x=2 y=3 z=4 l=5 r=6 d_up=7. NO-MIX: one beam.

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Erase}
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string])
policy = opts[:policy] || "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin"

slippi = Path.expand("~/Slippi")

replays =
  [
    "Game_20260713T015257.slp",
    "Game_20260713T070100.slp",
    "Game_20260714T115716.slp"
  ]
  |> Enum.map(&Path.join(slippi, &1))
  |> Enum.filter(&File.exists?/1)

Output.banner("Interp-surgical: LEACE copy-signal erasure")
Output.config([{"Policy", policy}, {"Replays", length(replays)}])

trunk = Activations.load_trunk(policy)
heads = Activations.load_heads_only(policy)
window = trunk.window

# ---------------------------------------------------------------------------
# X: trunk activations; Z: previous-frame buttons (the copy concept)
# ---------------------------------------------------------------------------
button_fields = [:button_a, :button_b, :button_x, :button_y, :button_z,
                 :button_l, :button_r, :button_d_up]

{x_parts, z_parts} =
  replays
  |> Enum.map(fn path ->
    cap = Activations.capture_replay(trunk, path, labels: false)

    {:ok, replay} = Peppi.parse(path)

    controllers =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.controller)

    # Z row i = buttons at frame (i + window - 2): the a_{t-1} the
    # prev-action channel carries at decision frame t = i + window - 1
    z =
      controllers
      |> Enum.slice(window - 2, cap.n)
      |> Enum.map(fn c ->
        Enum.map(button_fields, fn f -> if Map.get(c, f, false), do: 1, else: 0 end)
      end)
      |> Nx.tensor(type: :u8, backend: Nx.BinaryBackend)

    if Nx.axis_size(z, 0) != cap.n, do: raise("Z/X misalignment on #{path}")
    {cap.activations, z}
  end)
  |> Enum.unzip()

x = Nx.concatenate(x_parts, axis: 0)
z = Nx.concatenate(z_parts, axis: 0)
n = Nx.axis_size(x, 0)
Output.puts("X: {#{n}, #{Nx.axis_size(x, 1)}}  Z: {#{n}, 8} (prev buttons)")

# Fit on GPU-resident copies (covariances), small stage on CPU inside fit
x_dev = Nx.backend_copy(x, EXLA.Backend)
z_dev = Nx.backend_copy(z, EXLA.Backend)

File.mkdir_p!("interp/erasers")
out = "interp/erasers/#{Path.basename(policy, "_policy.bin")}_prevbtn_eraser.bin"

# Load a previously-fitted eraser if present — the ~6.5 min eigh is the
# expensive artifact; never re-pay it (delete the file to force a refit).
eraser =
  if File.exists?(out) do
    Output.puts("Loading saved eraser: #{out}")
    out |> File.read!() |> :erlang.binary_to_term()
  else
    e = Erase.fit(x_dev, z_dev)
    File.write!(out, :erlang.term_to_binary(e))
    Output.success("Eraser fitted (rank #{e.rank}) and saved: #{out}")
    e
  end

# Erasure guarantee check — DIRECT: LEACE's theorem is exactly
# "cross-cov(erase(X), Z) = 0", so assert that on the GPU in milliseconds
# instead of training probes to detect it indirectly (probe-based verify
# spent 20+ CPU-min per attempt; see CLAUDE.md observability notes).
Output.puts("guarantee check (cross-covariance)...")
eraser_gpu = %{eraser | mu: Nx.backend_copy(eraser.mu, EXLA.Backend), a: Nx.backend_copy(eraser.a, EXLA.Backend)}
x_gpu0 = Nx.backend_copy(x, EXLA.Backend)
z_gpu0 = Nx.backend_copy(z, EXLA.Backend) |> Nx.as_type(:f32)

crosscov = fn xt ->
  xc = Nx.subtract(xt, Nx.mean(xt, axes: [0]))
  zc = Nx.subtract(z_gpu0, Nx.mean(z_gpu0, axes: [0]))
  Nx.dot(Nx.transpose(xc), zc) |> Nx.divide(n) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
end

cc_before = crosscov.(x_gpu0)
cc_after = crosscov.(Erase.erase(eraser_gpu, x_gpu0))
Output.puts("max |cross-cov(X, Z)|: before=#{cc_before} after=#{cc_after} (guarantee: after ≈ 0)")

# ---------------------------------------------------------------------------
# Offline behavioral eval: heads(X) vs heads(erase(X))
# ---------------------------------------------------------------------------
Output.puts("heads eval stage...")
x_gpu = x_gpu0
x_erased = Erase.erase(eraser_gpu, x_gpu)

run_heads = fn xin ->
  {b, _mx, _my, _cx, _cy, _sh} = heads.predict_fn.(heads.params, Nx.as_type(xin, :f32))
  Nx.sigmoid(b) |> Nx.backend_transfer(Nx.BinaryBackend)
end

p_intact = run_heads.(x_gpu)
p_erased = run_heads.(x_erased)

mean_col = fn probs, idx -> probs |> Nx.slice_along_axis(idx, 1, axis: 1) |> Nx.mean() |> Nx.to_number() |> Float.round(4) end
z_col = fn idx -> z |> Nx.slice_along_axis(idx, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.as_type(:f32) end

cond_mean = fn probs, idx, mask ->
  p = probs |> Nx.slice_along_axis(idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
  Nx.sum(Nx.multiply(p, mask)) |> Nx.divide(Nx.max(Nx.sum(mask), 1)) |> Nx.to_number() |> Float.round(4)
end

prev_a = z_col.(0)
shield_prev = Nx.max(z_col.(5), z_col.(6))

report = fn label, probs ->
  shield_now = Nx.max(
    probs |> Nx.slice_along_axis(5, 1, axis: 1) |> Nx.squeeze(axes: [1]),
    probs |> Nx.slice_along_axis(6, 1, axis: 1) |> Nx.squeeze(axes: [1])
  )
  hold_persist = Nx.sum(Nx.multiply(shield_now, shield_prev)) |> Nx.divide(Nx.max(Nx.sum(shield_prev), 1)) |> Nx.to_number() |> Float.round(4)

  Output.puts(
    "#{label}: A=#{mean_col.(probs, 0)} A|prevA=#{cond_mean.(probs, 0, prev_a)} " <>
      "jumpXY=#{Float.round((mean_col.(probs, 2) + mean_col.(probs, 3)) / 2, 4)} " <>
      "shieldLR=#{Float.round((mean_col.(probs, 5) + mean_col.(probs, 6)) / 2, 4)} " <>
      "shieldHoldPersistence=#{hold_persist}"
  )
end

Output.puts("")
report.("INTACT", p_intact)
report.("ERASED", p_erased)

Output.puts("")
Output.puts("Reference (test-time ablation, from p3 offline): shield holds collapse,")
Output.puts("jump probs RISE. Surgical success = shieldHoldPersistence drops toward")
Output.puts("the unconditional shield rate AND A|prevA rises (follow-ups unblocked),")
Output.puts("with stick heads unchanged (checked live next).")
