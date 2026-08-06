# Flagship Stage 3, pass 2 (2026-08-06): nonlinear-path edge validation,
# empirical feature naming, and the aerial-B / delay-id graph.
#
#  1. EDGE VALIDATION THROUGH THE FULL PATH: suppress feature k's decoded
#     update from the ACTUAL trunk state at ground JC sites and read the
#     ACTUAL X logit through the heads (top-k, ReLU, nonlinearity and
#     all). Prediction chains Stage-3's pre-act edge into Stage-2's
#     feature->logit projection. This is the approximation-bearing test
#     the pass-1 pre-act checks could not provide.
#  2. FEATURE NAMING: top-activating rows across the corpus for each
#     graph node (f126 + its parents) -> game-state profile (action
#     family, y-band, grounded, af) -> empirical names.
#  3. AERIAL-B / DELAY-ID GRAPH: attribution of the B head at aerial-B
#     press-edge sites (the per-link Bernoulli site from the delay-break
#     study), computed twice — same states embedded at delay_id 3 vs 4.
#     Features whose B-contribution grows under id 4 are the carriers of
#     the id-4 margin advantage (p10 +0.09 -> +0.25).
#
#   mix run scripts/interp_graph2.exs [--out eval_runs/interp/graph_pass2.json]

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
out_path = opts[:out] || "eval_runs/interp/graph_pass2.json"

ground_replay = "eval_runs/0804_cycle3b_stand/r1.slp"
plat_replay = "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"
reflector = [360, 361, 362]
graph_features = [126, 1186, 1665, 958, 760, 1889]

Output.banner("Stage 3 pass 2")

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

logits_for = fn t_batch, idx ->
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

{g_acts, g_emb, g_frames} = load_replay.(ground_replay, 3)
{p_acts, p_emb, p_frames} = load_replay.(plat_replay, 3)

jc_sites = fn frames, acts, plat? ->
  n_rows = Nx.axis_size(acts, 0)

  frames
  |> Enum.with_index()
  |> Enum.filter(fn {f, t} ->
    p = f.game_state.players[1]

    t >= window + 1 and t < n_rows + window - 1 and p.action in reflector and
      p.action_frame >= 1 and if plat?, do: p.y > 15, else: p.y < 15
  end)
  |> Enum.map(&elem(&1, 1))
  |> Enum.take_every(2)
  |> Enum.take(40)
end

g_sites = jc_sites.(g_frames, g_acts, false)

# ---------------------------------------------------------------------------
# 1. Full-path edge validation at ground JC sites
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("1. Full-path edge validation (suppress k in ACTUAL trunk; ACTUAL X-logit delta):")

pair_x = fn acts, emb, s ->
  Nx.concatenate(
    [Nx.slice_along_axis(acts, s - 1, 1, axis: 0), Nx.slice_along_axis(emb, s - 1 + window, 1, axis: 0)],
    axis: 1
  )
end

trunk_rows = g_sites |> Enum.map(fn t -> Nx.slice_along_axis(g_acts, t - (window - 1), 1, axis: 0) end) |> Nx.concatenate(axis: 0)
x_rows = g_sites |> Enum.map(fn t -> pair_x.(g_acts, g_emb, t - (window - 1)) end) |> Nx.concatenate(axis: 0)
h_rows = tc_predict.(tc_params, %{tc_key => std.(x_rows)}).hidden
base_x_logit = Nx.mean(logits_for.(trunk_rows, 2)) |> Nx.to_number()

full_path_checks =
  for k <- [1186, 1665, 958] do
    dec_k = Nx.multiply(w_dec[k], y_std)
    h_k = h_rows[[.., k]]
    suppressed = Nx.subtract(trunk_rows, Nx.multiply(Nx.new_axis(h_k, 1), Nx.new_axis(dec_k, 0)))
    actual = Nx.to_number(Nx.mean(logits_for.(suppressed, 2))) - base_x_logit
    Output.puts("  suppress f#{k}: X-logit delta #{Float.round(actual, 4)} (base #{Float.round(base_x_logit, 3)})")
    %{from: k, x_logit_delta: actual}
  end

# ---------------------------------------------------------------------------
# 2. Empirical feature naming (top-activating rows across both replays)
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("2. Feature naming (top-32 activating states per node):")

all_pairs = fn acts, emb ->
  n = Nx.axis_size(acts, 0)

  Nx.concatenate(
    [
      Nx.slice_along_axis(acts, 0, n - 1, axis: 0),
      Nx.slice_along_axis(emb, window, n - 1, axis: 0)
    ],
    axis: 1
  )
end

xg_all = all_pairs.(g_acts, g_emb)
xp_all = all_pairs.(p_acts, p_emb)
h_all = tc_predict.(tc_params, %{tc_key => std.(Nx.concatenate([xg_all, xp_all], axis: 0))}).hidden
n_g = Nx.axis_size(xg_all, 0)

frame_at = fn row ->
  {frames, idx} = if row < n_g, do: {g_frames, row}, else: {p_frames, row - n_g}
  # pair row i -> update INTO decision at frame (i+1)+(window-1) = i+window
  Enum.at(frames, idx + window)
end

names =
  for j <- graph_features do
    top = Nx.argsort(h_all[[.., j]], direction: :desc) |> Nx.to_flat_list() |> Enum.take(32)

    profile =
      top
      |> Enum.map(frame_at)
      |> Enum.reject(&is_nil/1)
      |> Enum.map(fn f ->
        p = f.game_state.players[1]
        {ShineChain.family(p.action), p.on_ground, p.y > 15, f.controller.button_b, f.controller.button_x}
      end)

    fams = profile |> Enum.frequencies_by(&elem(&1, 0)) |> Enum.sort_by(fn {_, c} -> -c end) |> Enum.take(2)
    grounded = Enum.count(profile, &elem(&1, 1)) / max(length(profile), 1)
    plat = Enum.count(profile, &elem(&1, 2)) / max(length(profile), 1)
    b_rate = Enum.count(profile, &elem(&1, 3)) / max(length(profile), 1)
    x_rate = Enum.count(profile, &elem(&1, 4)) / max(length(profile), 1)

    Output.puts(
      "  f#{String.pad_leading(to_string(j), 4)}: fams=#{inspect(fams)} grounded=#{Float.round(grounded, 2)} " <>
        "plat=#{Float.round(plat, 2)} B=#{Float.round(b_rate, 2)} X=#{Float.round(x_rate, 2)}"
    )

    %{feature: j, families: Enum.map(fams, fn {f, c} -> [to_string(f), c] end), grounded: grounded, plat: plat, b_rate: b_rate, x_rate: x_rate}
  end

# ---------------------------------------------------------------------------
# 3. Aerial-B press-edge sites: delay-id 3 vs 4 attribution contrast
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("3. Aerial-B decision, delay-id 3 vs 4 (same states, different declared id):")

{_, g_emb4, _} = load_replay.(ground_replay, 4)

b_edges =
  g_frames
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.with_index(1)
  |> Enum.filter(fn {[f0, f1], t} ->
    p = f1.game_state.players[1]
    fam = ShineChain.family(p.action)

    t >= window + 1 and t < Nx.axis_size(g_acts, 0) + window - 1 and
      f1.controller.button_b and not f0.controller.button_b and
      fam in [:jumpsquat, :aerial_jump, :air_reflect]
  end)
  |> Enum.map(&elem(&1, 1))
  |> Enum.take_every(3)
  |> Enum.take(40)

Output.puts("  #{length(b_edges)} aerial-B press-edge sites")

# B-head gradient at the sites (id-agnostic: heads read the trunk)
fd_eps = 1.0e-2

b_grad = fn trunk_row ->
  base = Nx.squeeze(trunk_row)
  eye = Nx.multiply(Nx.eye(256), fd_eps)
  batch = Nx.concatenate([Nx.add(Nx.new_axis(base, 0), eye), Nx.subtract(Nx.new_axis(base, 0), eye)], axis: 0)
  col = logits_for.(batch, 1)
  Nx.divide(Nx.subtract(Nx.slice_along_axis(col, 0, 256, axis: 0), Nx.slice_along_axis(col, 256, 256, axis: 0)), 2 * fd_eps)
end

attr_for = fn emb_variant ->
  per_site =
    for t <- b_edges do
      s = t - (window - 1)
      trunk_s = Nx.slice_along_axis(g_acts, s, 1, axis: 0)
      g = b_grad.(trunk_s)
      proj = Nx.dot(Nx.multiply(w_dec, Nx.new_axis(y_std, 0)), g)
      x_row = pair_x.(g_acts, emb_variant, s)
      h = tc_predict.(tc_params, %{tc_key => std.(x_row)}).hidden |> Nx.squeeze()
      Nx.multiply(h, proj)
    end

  per_site |> Nx.stack() |> Nx.mean(axes: [0])
end

attr3 = attr_for.(g_emb)
attr4 = attr_for.(g_emb4)
delta = Nx.subtract(attr4, attr3)
top_carriers = Nx.argsort(delta, direction: :desc) |> Nx.to_flat_list() |> Enum.take(8)

Output.puts("  Top id-4 margin carriers (B-attribution gain, id4 - id3):")

carriers =
  for j <- top_carriers do
    d = Nx.to_number(delta[j])
    Output.puts("    f#{String.pad_leading(to_string(j), 4)}  gain=#{Float.round(d, 4)} (id3 #{Float.round(Nx.to_number(attr3[j]), 4)} -> id4 #{Float.round(Nx.to_number(attr4[j]), 4)})")
    %{feature: j, gain: d, id3: Nx.to_number(attr3[j]), id4: Nx.to_number(attr4[j])}
  end

total_gain = Nx.to_number(Nx.sum(delta))
Output.puts("  total B-attribution gain id3->id4: #{Float.round(total_gain, 4)} (margin sweep measured +0.19 at p10)")

File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(%{
    full_path_checks: full_path_checks,
    base_x_logit: base_x_logit,
    names: names,
    delay_carriers: carriers,
    total_b_gain_id3_to_id4: total_gain
  })
)

Output.success("Wrote #{out_path}")
