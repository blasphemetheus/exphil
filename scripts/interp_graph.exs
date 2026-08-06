# Flagship Stage 3 (2026-08-06): mini attribution graph around f126 —
# the cycle-drive feature Stage 2 found carrying the X/JC decision.
#
# Three products, every edge validated:
#   A. GATE DECOMPOSITION — why is f126 silent on platforms? The pre-act
#      gap (ground minus platform) decomposes exactly over encoder
#      weights: gap_i = W_enc[i,126] * (mean_x_ground[i] - mean_x_plat[i]).
#      Top dims are named EMPIRICALLY (correlation with own-y / own-action
#      / airborne over the corpus — no layout archaeology) and validated
#      by single-dim patching (set the dim to its ground mean at platform
#      sites; does f126's pre-act recover by the predicted amount?).
#   B. FEATURE->FEATURE EDGES — static edge weight into f126 from any
#      prior update feature k: E[k] = < W_enc[:256,126] / x_std_trunk,
#      W_dec[k,:] (.) y_std >  (feature k's decoded trunk delta, read
#      through f126's trunk-side encoder row). Combined with mean h_k at
#      ground sites for dynamic strength.
#   C. EDGE VALIDATION — top feature edges: suppress k's decoded update
#      from the trunk input, recompute f126's pre-act, compare predicted.
#
#   mix run scripts/interp_graph.exs [--feature 126] [--delay-id 3]
#     [--sites 40] [--out eval_runs/interp/graph_f126.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [feature: :integer, delay_id: :integer, sites: :integer, dict: :string, policy: :string, out: :string]
  )

feature = opts[:feature] || 126
delay_id = opts[:delay_id] || 3
n_sites = opts[:sites] || 40
policy_path = opts[:policy] || "checkpoints/ms_g10b_human.bin"
dict_path = opts[:dict] || "eval_runs/interp/transcoder_ms_g10b_human.bin"
out_path = opts[:out] || "eval_runs/interp/graph_f#{feature}.json"

ground_replay = "eval_runs/0804_cycle3b_stand/r1.slp"
plat_replay = "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp"
reflector = [360, 361, 362]

Output.banner("Stage 3: attribution graph around f#{feature}")

trunk = Activations.load_trunk(policy_path)
window = trunk.window
dict = dict_path |> File.read!() |> :erlang.binary_to_term()
tc_data = if match?(%Axon.ModelState{}, dict.params), do: dict.params.data, else: dict.params

# encoder kernel {592, 2048}, bias {2048}; decoder kernel {2048, 256}
w_enc = tc_data["transcoder_encoder"]["kernel"] |> Nx.backend_copy(EXLA.Backend)
b_enc = tc_data["transcoder_encoder"]["bias"] |> Nx.backend_copy(EXLA.Backend)
w_dec = tc_data["transcoder_decoder"]["kernel"] |> Nx.backend_copy(EXLA.Backend)
y_std = dict.y_std |> Nx.squeeze() |> Nx.backend_copy(EXLA.Backend)
x_mean = Nx.backend_copy(dict.x_mean, EXLA.Backend) |> Nx.squeeze()
x_std = Nx.backend_copy(dict.x_std, EXLA.Backend) |> Nx.squeeze()

enc_col = w_enc[[.., feature]]
bias_f = Nx.to_number(b_enc[feature])

collect = fn replay_path, plat? ->
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

      t >= window + 1 and t < n_rows + window - 1 and p.action in reflector and
        p.action_frame >= 1 and if plat?, do: p.y > 15, else: p.y < 15
    end)
    |> Enum.map(&elem(&1, 1))
    |> Enum.take_every(2)
    |> Enum.take(n_sites)

  # Pair input INTO row s: x = [act_{s-1} ++ emb_{s-1+window}]
  xs =
    sites
    |> Enum.map(fn site ->
      s = site - (window - 1)

      Nx.concatenate(
        [
          Nx.slice_along_axis(acts, s - 1, 1, axis: 0),
          Nx.slice_along_axis(emb, s - 1 + window, 1, axis: 0)
        ],
        axis: 1
      )
    end)
    |> Nx.concatenate(axis: 0)

  own_y = Enum.map(sites, fn t -> Enum.at(frames, t).game_state.players[1].y end)
  {xs, own_y, sites}
end

{x_ground, _yg, ground_sites} = collect.(ground_replay, false)
{x_plat, _yp, plat_sites} = collect.(plat_replay, true)
Output.puts("sites: ground #{length(ground_sites)}, platform #{length(plat_sites)}")

std = fn x -> Nx.divide(Nx.subtract(x, x_mean), Nx.add(x_std, 1.0e-6)) end
xg_std = std.(x_ground)
xp_std = std.(x_plat)

pre = fn x -> Nx.add(Nx.dot(x, enc_col), bias_f) end
pre_g = pre.(xg_std)
pre_p = pre.(xp_std)

Output.puts(
  "f#{feature} pre-act: ground mean #{Float.round(Nx.to_number(Nx.mean(pre_g)), 3)} " <>
    "vs platform #{Float.round(Nx.to_number(Nx.mean(pre_p)), 3)} (ReLU gate at 0)"
)

# ---------------------------------------------------------------------------
# A. Gate decomposition
# ---------------------------------------------------------------------------
mean_gap = Nx.subtract(Nx.mean(xg_std, axes: [0]), Nx.mean(xp_std, axes: [0]))
contrib = Nx.multiply(enc_col, mean_gap)
top_dims = Nx.argsort(Nx.abs(contrib), direction: :desc) |> Nx.to_flat_list() |> Enum.take(12)

Output.puts("")
Output.puts("A. Gate decomposition (why silent on platform) — top dims of the pre-act gap:")

dim_rows =
  for i <- top_dims do
    c = Nx.to_number(contrib[i])
    part = if i < 256, do: "trunk", else: "input(emb #{i - 256})"
    Output.puts("  dim #{String.pad_leading(to_string(i), 3)} [#{part}]  contribution #{Float.round(c, 4)}")
    %{dim: i, part: part, contribution: c}
  end

trunk_share =
  Nx.to_number(Nx.sum(Nx.abs(Nx.slice_along_axis(contrib, 0, 256, axis: 0)))) /
    max(Nx.to_number(Nx.sum(Nx.abs(contrib))), 1.0e-9)

Output.puts("  trunk-side share of |gap|: #{Float.round(trunk_share, 3)} (rest = direct input)")

# Validation A: patch the top INPUT dims at platform sites to ground means
input_top = Enum.filter(top_dims, &(&1 >= 256)) |> Enum.take(3)

input_checks =
  for i <- input_top do
    predicted = Nx.to_number(contrib[i])
    xg_mean_i = Nx.to_number(Nx.mean(xg_std[[.., i]]))
    xp_patched = Nx.put_slice(xp_std, [0, i], Nx.broadcast(xg_mean_i, {Nx.axis_size(xp_std, 0), 1}))
    actual = Nx.to_number(Nx.subtract(Nx.mean(pre.(xp_patched)), Nx.mean(pre_p)))
    Output.puts("  patch dim #{i} -> ground mean: pre-act recovers #{Float.round(actual, 4)} (predicted #{Float.round(predicted, 4)})")
    %{dim: i, predicted: predicted, actual: actual}
  end

# ---------------------------------------------------------------------------
# B. Feature->feature edges into f#{feature}
# ---------------------------------------------------------------------------
# E[k] = < enc_col[:256] / x_std[:256], W_dec[k,:] (.) y_std >
enc_trunk = Nx.divide(Nx.slice_along_axis(enc_col, 0, 256, axis: 0), Nx.add(Nx.slice_along_axis(x_std, 0, 256, axis: 0), 1.0e-6))
edge_static = Nx.dot(Nx.multiply(w_dec, Nx.new_axis(y_std, 0)), enc_trunk)

# dynamic strength: mean hidden over ground pair inputs
tc_model = Edifice.Interpretability.Transcoder.build(Keyword.put(dict.build_opts, :output, :container))
{_i, tc_predict} = Axon.build(tc_model, mode: :inference)
tc_key = tc_model |> Axon.get_inputs() |> Map.keys() |> hd()
tc_params = ExPhil.Training.Utils.ensure_model_state(tc_data)
h_ground = tc_predict.(tc_params, %{tc_key => xg_std}).hidden
h_mean = Nx.mean(h_ground, axes: [0])
edge_dyn = Nx.multiply(edge_static, h_mean)

top_edges = Nx.argsort(Nx.abs(edge_dyn), direction: :desc) |> Nx.to_flat_list() |> Enum.take(10)

Output.puts("")
Output.puts("B. Feature->feature edges into f#{feature} (static x mean ground activation):")

edge_rows =
  for k <- top_edges do
    Output.puts(
      "  f#{String.pad_leading(to_string(k), 4)} -> f#{feature}  E=#{Float.round(Nx.to_number(edge_static[k]), 4)} " <>
        "h_mean=#{Float.round(Nx.to_number(h_mean[k]), 3)} strength=#{Float.round(Nx.to_number(edge_dyn[k]), 4)}"
    )

    %{from: k, static: Nx.to_number(edge_static[k]), h_mean: Nx.to_number(h_mean[k]), strength: Nx.to_number(edge_dyn[k])}
  end

# Validation B: suppress feature k's decoded update from the trunk input
Output.puts("")
Output.puts("C. Edge validation (suppress k's decoded update in the trunk input; f#{feature} pre-act delta):")

edge_checks =
  for k <- Enum.take(top_edges, 3), k != feature do
    dec_k_raw = Nx.multiply(w_dec[k], y_std)
    h_k = h_ground[[.., k]]
    # raw trunk part of x_ground minus h_k * dec_k, restandardized
    raw_trunk = Nx.slice_along_axis(x_ground, 0, 256, axis: 1)
    suppressed = Nx.subtract(raw_trunk, Nx.multiply(Nx.new_axis(h_k, 1), Nx.new_axis(dec_k_raw, 0)))
    x_mod = Nx.concatenate([suppressed, Nx.slice_along_axis(x_ground, 256, 336, axis: 1)], axis: 1)
    actual = Nx.to_number(Nx.subtract(Nx.mean(pre.(std.(x_mod))), Nx.mean(pre_g)))
    predicted = -Nx.to_number(Nx.mean(Nx.multiply(h_k, edge_static[k])))
    Output.puts("  suppress f#{k}: pre-act delta #{Float.round(actual, 4)} (predicted #{Float.round(predicted, 4)})")
    %{from: k, predicted: predicted, actual: actual}
  end

File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(%{
    feature: feature,
    pre_act: %{ground: Nx.to_number(Nx.mean(pre_g)), platform: Nx.to_number(Nx.mean(pre_p))},
    gate_decomposition: dim_rows,
    trunk_share: trunk_share,
    input_patch_checks: input_checks,
    edges: edge_rows,
    edge_checks: edge_checks
  })
)

Output.success("Wrote #{out_path}")
