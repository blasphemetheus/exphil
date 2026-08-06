# Flagship Stage 2 (2026-08-06): frozen-gate single-decision attribution.
#
# Uses the Stage-1 trunk-update dictionary (R^2 0.56) to answer "which
# update FEATURES drive head H at decision site T". Approximation: within
# the residualized frame, trunk_T ~= trunk_{T-K} + sum of decoded feature
# updates (identity state-carry — the GRU's gates 'frozen'), so feature j
# at step t<=T contributes
#
#     a_{j,t} = h_t[j] * < W_dec[:,j] (.) y_std , dLogit/dtrunk_T >
#
# Every top feature gets a CAUSAL CHECK: subtract its decoded
# contributions from trunk_T, re-run the heads, compare the actual logit
# delta to the predicted one — approximation error is REPORTED, not
# hidden (the Stage-3 edge-validation ethos, applied early).
#
# First customer: the platform X-silence. Sites = JC-phase frames
# (reflector af>=3) split ground (y<15, X fires) vs platform (y>15, X
# hard-off); the differential top features are the silencing candidates.
#
#   mix run scripts/interp_attribution.exs \
#     [--policy checkpoints/ms_g10b_human.bin] \
#     [--dict eval_runs/interp/transcoder_ms_g10b_human.bin] \
#     [--delay-id 3] [--k-steps 12] [--sites 40] [--head-index 2] \
#     [--out eval_runs/interp/attribution_platform_x.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      dict: :string,
      delay_id: :integer,
      k_steps: :integer,
      sites: :integer,
      head_index: :integer,
      out: :string
    ]
  )

policy_path = opts[:policy] || "checkpoints/ms_g10b_human.bin"
dict_path = opts[:dict] || "eval_runs/interp/transcoder_ms_g10b_human.bin"
delay_id = opts[:delay_id] || 3
k_steps = opts[:k_steps] || 12
n_sites = opts[:sites] || 40
# buttons head order [a, b, x, y, z, l, r, d_up] — default X (index 2)
head_index = opts[:head_index] || 2
out_path = opts[:out] || "eval_runs/interp/attribution_platform_x.json"

ground_replay = "eval_runs/0804_cycle3b_stand/r1.slp"
plat_replay = "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"
reflector = [360, 361, 362]

Output.banner("Stage 2: frozen-gate attribution (X head, plat vs ground JC)")

trunk = Activations.load_trunk(policy_path)
heads = Activations.load_heads_only(policy_path)
window = trunk.window

dict = dict_path |> File.read!() |> :erlang.binary_to_term()
tc_model = Edifice.Interpretability.Transcoder.build(Keyword.put(dict.build_opts, :output, :container))
{_init, tc_predict} = Axon.build(tc_model, mode: :inference)
tc_input_key = tc_model |> Axon.get_inputs() |> Map.keys() |> hd()

# Saved params may be a bare data map (post-GOTCHA#1 deep transfer) or a
# full ModelState; normalize to both forms.
tc_data = if match?(%Axon.ModelState{}, dict.params), do: dict.params.data, else: dict.params
tc_params = ExPhil.Training.Utils.ensure_model_state(tc_data)

# Decoder kernel {dict_size, 256} and the de-standardization row vector
w_dec = tc_data["transcoder_decoder"]["kernel"] |> Nx.backend_copy(EXLA.Backend)
y_std = dict.y_std |> Nx.squeeze() |> Nx.backend_copy(EXLA.Backend)
x_mean = Nx.backend_copy(dict.x_mean, EXLA.Backend)
x_std = Nx.backend_copy(dict.x_std, EXLA.Backend)

# dLogit/dtrunk via central finite differences (the heads-only model is a
# tiny MLP — one batched predict over 512 perturbed rows beats fighting
# nested Axon/defn jit for an exact grad).
fd_eps = 1.0e-2

grad_fn = fn params, t ->
  base = Nx.squeeze(t)
  eye = Nx.multiply(Nx.eye(256), fd_eps)
  plus = Nx.add(Nx.new_axis(base, 0), eye)
  minus = Nx.subtract(Nx.new_axis(base, 0), eye)
  batch = Nx.concatenate([plus, minus], axis: 0)
  logits = heads.predict_fn.(params, batch) |> elem(0)
  col = logits[[.., head_index]]
  lp = Nx.slice_along_axis(col, 0, 256, axis: 0)
  lm = Nx.slice_along_axis(col, 256, 256, axis: 0)
  Nx.divide(Nx.subtract(lp, lm), 2 * fd_eps)
end

logit_fn = fn t ->
  heads.predict_fn.(heads.params, t) |> elem(0) |> Nx.squeeze() |> Nx.slice([head_index], [1]) |> Nx.to_flat_list() |> hd()
end

analyze = fn replay_path, plat?, label ->
  cap = Activations.capture_replay(trunk, replay_path, delay_id: delay_id, labels: false)
  acts = Nx.backend_copy(cap.activations, EXLA.Backend)

  frames =
    replay_path
    |> Path.expand()
    |> then(fn p -> {:ok, r} = Peppi.parse(p); r end)
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))

  ds = Activations.embed_frames(frames, trunk.config, delay_id: delay_id)
  emb = Nx.backend_copy(Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend), EXLA.Backend)

  n_rows = Nx.axis_size(acts, 0)

  sites =
    frames
    |> Enum.with_index()
    |> Enum.filter(fn {f, t} ->
      p = f.game_state.players[1]

      t >= window + k_steps and t < n_rows + window - 1 and p.action in reflector and
        p.action_frame >= 1 and if plat?, do: p.y > 15, else: p.y < 15
    end)
    |> Enum.map(&elem(&1, 1))
    |> Enum.take_every(2)
    |> Enum.take(n_sites)

  Output.puts("#{label}: #{length(sites)} JC sites")

  site_results =
    for site_frame <- sites do
      s = site_frame - (window - 1)
      trunk_s = Nx.slice_along_axis(acts, s, 1, axis: 0)
      g = grad_fn.(heads.params, trunk_s) |> Nx.squeeze()

      # projection of each feature's decoded direction onto the gradient
      # (de-standardized): {dict_size}
      proj = Nx.dot(Nx.multiply(w_dec, Nx.new_axis(y_std, 0)), Nx.multiply(g, 1.0))

      # transcoder hiddens for the K update steps into trunk_s:
      # pair i = [act_i ++ emb_{i+window}] -> update into row i+1
      pair_idx = (s - k_steps)..(s - 1)

      xs =
        Nx.concatenate(
          [
            Nx.slice_along_axis(acts, s - k_steps, k_steps, axis: 0),
            Nx.slice_along_axis(emb, s - k_steps + window, k_steps, axis: 0)
          ],
          axis: 1
        )

      xs_std = Nx.divide(Nx.subtract(xs, x_mean), Nx.add(x_std, 1.0e-6))
      hidden = tc_predict.(tc_params, %{tc_input_key => xs_std}).hidden

      # a_{j} = sum_t h_t[j] * proj[j]  -> {dict_size}
      attr = Nx.multiply(Nx.sum(hidden, axes: [0]), proj)

      %{
        site: site_frame,
        logit: logit_fn.(trunk_s),
        attr: attr,
        hidden_sum: Nx.sum(hidden, axes: [0]),
        trunk: trunk_s,
        _pair_idx: pair_idx
      }
    end

  # Aggregate: mean attribution per feature across sites
  mean_attr =
    site_results |> Enum.map(& &1.attr) |> Nx.stack() |> Nx.mean(axes: [0])

  {site_results, mean_attr}
end

{_ground_sites, ground_attr} = analyze.(ground_replay, false, "ground")
{plat_sites, plat_attr} = analyze.(plat_replay, true, "platform")

# Differential attribution: features pushing X DOWN on platform relative
# to ground (most negative delta = silencing candidates)
delta = Nx.subtract(plat_attr, ground_attr)
order = Nx.argsort(delta, direction: :asc) |> Nx.to_flat_list() |> Enum.take(10)

Output.puts("")
Output.puts("Top X-silencing candidates (platform-vs-ground differential attribution):")

candidates =
  for j <- order do
    d = Nx.to_number(delta[j])
    p = Nx.to_number(plat_attr[j])
    gr = Nx.to_number(ground_attr[j])
    Output.puts("  f#{String.pad_leading(to_string(j), 4)}  delta=#{Float.round(d, 4)} (plat #{Float.round(p, 4)} / ground #{Float.round(gr, 4)})")
    %{feature: j, delta: d, plat: p, ground: gr}
  end

# ---------------------------------------------------------------------------
# CAUSAL CHECK: ablate the top-3 candidates' decoded contributions from the
# platform trunk states, re-run the heads, compare actual vs predicted
# X-logit change.
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("Causal check (ablate feature contribution from trunk state; predicted vs actual X-logit delta):")

checks =
  for %{feature: j} <- Enum.take(candidates, 3) do
    dec_j = Nx.multiply(w_dec[j], y_std)

    deltas =
      for site <- Enum.take(plat_sites, 12) do
        h_j = Nx.to_number(site.hidden_sum[j])
        predicted = -Nx.to_number(site.attr[j])
        ablated = Nx.subtract(site.trunk, Nx.multiply(h_j, Nx.new_axis(dec_j, 0)))
        actual = logit_fn.(ablated) - site.logit
        {predicted, actual}
      end

    mp = deltas |> Enum.map(&elem(&1, 0)) |> Enum.sum() |> Kernel./(length(deltas))
    ma = deltas |> Enum.map(&elem(&1, 1)) |> Enum.sum() |> Kernel./(length(deltas))

    agree =
      Enum.count(deltas, fn {p, a} -> p * a > 0 or (abs(p) < 0.01 and abs(a) < 0.01) end) /
        length(deltas)

    Output.puts(
      "  f#{String.pad_leading(to_string(j), 4)}  predicted_mean=#{Float.round(mp, 4)} actual_mean=#{Float.round(ma, 4)} sign_agree=#{Float.round(agree, 2)}"
    )

    %{feature: j, predicted_mean: mp, actual_mean: ma, sign_agreement: agree}
  end

File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(%{
    head_index: head_index,
    k_steps: k_steps,
    candidates: candidates,
    causal_checks: checks
  })
)

Output.success("Wrote #{out_path}")
