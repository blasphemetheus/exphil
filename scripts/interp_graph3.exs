# Flagship Stage 3, pass 3 (2026-08-06): decision-sited validation,
# injection steering, and the (now in-distribution) delay-id contrast.
#
# Pass-2 lessons applied:
#   - Sites are DECISION-defined: frames where the head's own logit is
#     high (X > 0 for JC decisions, B > 0 airborne for aerial presses) —
#     not animation frames ~decode-lag later.
#   - The dictionary was refit multi-id, so id-4 embeddings are
#     in-distribution; feature indices are re-derived fresh.
#
# Sections:
#   1. Re-derive the X-drive node(s): attribution at X-decision sites.
#   2. Full-path suppression at decision sites (the approximation-bearing
#      edge validation pass 2 could not deliver).
#   3. INJECTION counterfactual: add the drive feature's decoded vector
#      into platform shine-hold trunk states — does X wake? (The
#      feature-level version of the y-patch RESTORE test.)
#   4. Delay-id margin carriers: B-head attribution at aerial-B decision
#      sites under id-3 vs id-4 embeddings of the same states.
#
#   mix run scripts/interp_graph3.exs [--out eval_runs/interp/graph_pass3.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, dict: :string, out: :string]
  )

policy_path = opts[:policy] || "checkpoints/ms_g10b_human.bin"
dict_path = opts[:dict] || "eval_runs/interp/transcoder_ms_g10b_human.bin"
out_path = opts[:out] || "eval_runs/interp/graph_pass3.json"

ground_replay = "eval_runs/0804_cycle3b_stand/r1.slp"
plat_replay = "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"

Output.banner("Stage 3 pass 3")

trunk = Activations.load_trunk(policy_path)
heads = Activations.load_heads_only(policy_path)
window = trunk.window
dict = dict_path |> File.read!() |> :erlang.binary_to_term()
tc_data = if match?(%Axon.ModelState{}, dict.params), do: dict.params.data, else: dict.params
tc_params = ExPhil.Training.Utils.ensure_model_state(tc_data)
tc_model = Edifice.Interpretability.Transcoder.build(Keyword.put(dict.build_opts, :output, :container))
{_i, tc_predict} = Axon.build(tc_model, mode: :inference)
tc_key = tc_model |> Axon.get_inputs() |> Map.keys() |> hd()

w_dec = tc_data["transcoder_decoder"]["kernel"] |> Nx.backend_copy(EXLA.Backend)
y_std = dict.y_std |> Nx.squeeze() |> Nx.backend_copy(EXLA.Backend)
x_mean = Nx.backend_copy(dict.x_mean, EXLA.Backend) |> Nx.squeeze()
x_std = Nx.backend_copy(dict.x_std, EXLA.Backend) |> Nx.squeeze()
std = fn x -> Nx.divide(Nx.subtract(x, x_mean), Nx.add(x_std, 1.0e-6)) end

logits_col = fn t_batch, idx ->
  heads.predict_fn.(heads.params, t_batch) |> elem(0) |> then(& &1[[.., idx]])
end

load_replay = fn path, delay_id ->
  cap = Activations.capture_replay(trunk, path, delay_id: delay_id, labels: false)
  acts = Nx.backend_copy(cap.activations, EXLA.Backend)

  frames =
    path
    |> Path.expand()
    |> then(fn p -> {:ok, r} = Peppi.parse(p); r end)
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))

  ds = Activations.embed_frames(frames, trunk.config, delay_id: delay_id)
  emb = Nx.backend_copy(Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend), EXLA.Backend)
  {acts, emb, frames}
end

{g_acts, g_emb, _g_frames} = load_replay.(ground_replay, 3)
{p_acts, p_emb, p_frames} = load_replay.(plat_replay, 3)

pair_x = fn acts, emb, s ->
  Nx.concatenate(
    [Nx.slice_along_axis(acts, s - 1, 1, axis: 0), Nx.slice_along_axis(emb, s - 1 + window, 1, axis: 0)],
    axis: 1
  )
end

# ---------------------------------------------------------------------------
# 1. Decision sites + re-derived X-drive features
# ---------------------------------------------------------------------------
n_rows = Nx.axis_size(g_acts, 0)
x_all = logits_col.(g_acts, 2)
x_dec = Nx.greater(x_all, 0.0) |> Nx.to_flat_list()

dec_sites =
  x_dec
  |> Enum.with_index()
  |> Enum.filter(fn {fired, s} -> fired == 1 and s > 1 and s < n_rows - 1 end)
  |> Enum.map(&elem(&1, 1))
  |> Enum.take_every(4)
  |> Enum.take(60)

Output.puts("1. X-decision sites on FD: #{length(dec_sites)} (of #{Enum.count(x_dec, &(&1 == 1))} firing rows)")

fd_eps = 1.0e-2

grad_at = fn trunk_row, idx ->
  base = Nx.squeeze(trunk_row)
  eye = Nx.multiply(Nx.eye(256), fd_eps)
  batch = Nx.concatenate([Nx.add(Nx.new_axis(base, 0), eye), Nx.subtract(Nx.new_axis(base, 0), eye)], axis: 0)
  col = logits_col.(batch, idx)
  Nx.divide(Nx.subtract(Nx.slice_along_axis(col, 0, 256, axis: 0), Nx.slice_along_axis(col, 256, 256, axis: 0)), 2 * fd_eps)
end

site_data =
  for s <- dec_sites do
    trunk_s = Nx.slice_along_axis(g_acts, s, 1, axis: 0)
    h = tc_predict.(tc_params, %{tc_key => std.(pair_x.(g_acts, g_emb, s))}).hidden |> Nx.squeeze()
    g = grad_at.(trunk_s, 2)
    proj = Nx.dot(Nx.multiply(w_dec, Nx.new_axis(y_std, 0)), g)
    %{s: s, trunk: trunk_s, h: h, attr: Nx.multiply(h, proj)}
  end

mean_attr = site_data |> Enum.map(& &1.attr) |> Nx.stack() |> Nx.mean(axes: [0])
drive = Nx.argsort(mean_attr, direction: :desc) |> Nx.to_flat_list() |> Enum.take(5)

Output.puts("  top X-drive features at DECISION sites: " <>
  Enum.map_join(drive, ", ", fn j -> "f#{j} (#{Float.round(Nx.to_number(mean_attr[j]), 3)})" end))

# ---------------------------------------------------------------------------
# 2. Full-path suppression at decision sites
# ---------------------------------------------------------------------------
trunk_rows = site_data |> Enum.map(& &1.trunk) |> Nx.concatenate(axis: 0)
base_x = Nx.to_number(Nx.mean(logits_col.(trunk_rows, 2)))
h_rows = site_data |> Enum.map(& &1.h) |> Nx.stack()

Output.puts("")
Output.puts("2. Full-path suppression at decision sites (base X-logit #{Float.round(base_x, 3)}):")

suppression =
  for j <- Enum.take(drive, 3) do
    dec_j = Nx.multiply(w_dec[j], y_std)
    h_j = h_rows[[.., j]]
    supp = Nx.subtract(trunk_rows, Nx.multiply(Nx.new_axis(h_j, 1), Nx.new_axis(dec_j, 0)))
    actual = Nx.to_number(Nx.mean(logits_col.(supp, 2))) - base_x
    predicted = -Nx.to_number(Nx.mean(Nx.multiply(h_j, mean_attr[j] |> Nx.divide(Nx.max(Nx.mean(h_j), 1.0e-6)))))
    Output.puts("  suppress f#{j}: X-logit #{Float.round(actual, 3)} (attr-implied #{Float.round(-Nx.to_number(mean_attr[j]), 3)})")
    %{feature: j, x_delta: actual, attr_implied: -Nx.to_number(mean_attr[j]), predicted: predicted}
  end

# ---------------------------------------------------------------------------
# 3. Injection counterfactual at platform shine-hold states
# ---------------------------------------------------------------------------
reflector = [360, 361, 362]

plat_sites =
  p_frames
  |> Enum.with_index()
  |> Enum.filter(fn {f, t} ->
    p = f.game_state.players[1]

    t >= window + 1 and t < Nx.axis_size(p_acts, 0) + window - 1 and p.action in reflector and
      p.y > 15
  end)
  |> Enum.map(&elem(&1, 1))
  |> Enum.take_every(4)
  |> Enum.take(40)
  |> Enum.map(fn t -> t - (window - 1) end)

plat_trunks = plat_sites |> Enum.map(fn s -> Nx.slice_along_axis(p_acts, s, 1, axis: 0) end) |> Nx.concatenate(axis: 0)
plat_base = Nx.to_number(Nx.mean(logits_col.(plat_trunks, 2)))

# ground-typical activation level of each drive feature
h_scale = fn j -> Nx.to_number(Nx.mean(h_rows[[.., j]])) end

Output.puts("")
Output.puts("3. Injection at #{length(plat_sites)} platform shine-holds (base X-logit #{Float.round(plat_base, 3)}):")

injection =
  for j <- Enum.take(drive, 3) do
    dec_j = Nx.multiply(w_dec[j], y_std)
    scale = h_scale.(j)
    injected = Nx.add(plat_trunks, Nx.multiply(scale, Nx.new_axis(dec_j, 0)))
    actual = Nx.to_number(Nx.mean(logits_col.(injected, 2))) - plat_base
    Output.puts("  inject f#{j} (scale #{Float.round(scale, 2)}): X-logit +#{Float.round(actual, 3)}")
    %{feature: j, scale: scale, x_delta: actual}
  end

# all three at once (the "restore the drive circuit" counterfactual)
combined =
  Enum.reduce(Enum.take(drive, 3), plat_trunks, fn j, acc ->
    Nx.add(acc, Nx.multiply(h_scale.(j), Nx.new_axis(Nx.multiply(w_dec[j], y_std), 0)))
  end)

combined_delta = Nx.to_number(Nx.mean(logits_col.(combined, 2))) - plat_base
Output.puts("  inject ALL top-3: X-logit +#{Float.round(combined_delta, 3)}")

# ---------------------------------------------------------------------------
# 4. Delay-id margin carriers (in-distribution this time)
# ---------------------------------------------------------------------------
{_, g_emb4, _} = load_replay.(ground_replay, 4)

b_all = logits_col.(g_acts, 1)

b_sites =
  Nx.greater(b_all, 0.0)
  |> Nx.to_flat_list()
  |> Enum.with_index()
  |> Enum.filter(fn {fired, s} -> fired == 1 and s > 1 and s < n_rows - 1 end)
  |> Enum.map(&elem(&1, 1))
  |> Enum.take_every(6)
  |> Enum.take(50)

Output.puts("")
Output.puts("4. Delay-id contrast at #{length(b_sites)} B-decision sites:")

attr_b = fn emb_variant ->
  per =
    for s <- b_sites do
      trunk_s = Nx.slice_along_axis(g_acts, s, 1, axis: 0)
      g = grad_at.(trunk_s, 1)
      proj = Nx.dot(Nx.multiply(w_dec, Nx.new_axis(y_std, 0)), g)
      h = tc_predict.(tc_params, %{tc_key => std.(pair_x.(g_acts, emb_variant, s))}).hidden |> Nx.squeeze()
      Nx.multiply(h, proj)
    end

  per |> Nx.stack() |> Nx.mean(axes: [0])
end

a3 = attr_b.(g_emb)
a4 = attr_b.(g_emb4)
d = Nx.subtract(a4, a3)
carriers = Nx.argsort(d, direction: :desc) |> Nx.to_flat_list() |> Enum.take(6)
total = Nx.to_number(Nx.sum(d))

rows =
  for j <- carriers do
    Output.puts("  f#{String.pad_leading(to_string(j), 4)} gain #{Float.round(Nx.to_number(d[j]), 4)} (id3 #{Float.round(Nx.to_number(a3[j]), 4)} -> id4 #{Float.round(Nx.to_number(a4[j]), 4)})")
    %{feature: j, gain: Nx.to_number(d[j]), id3: Nx.to_number(a3[j]), id4: Nx.to_number(a4[j])}
  end

Output.puts("  total B-attribution shift id3->id4: #{Float.round(total, 4)}")

File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(%{
    drive_features: drive,
    suppression: suppression,
    injection: injection,
    combined_injection: combined_delta,
    delay_carriers: rows,
    total_shift: total
  })
)

Output.success("Wrote #{out_path}")
