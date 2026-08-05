# Trunk-activation OOD scalar (ML_FIELDS_ROADMAP F2): Mahalanobis distance
# to a policy's in-distribution activation manifold.
#
# Fit: mean + ridge-regularized covariance of trunk activations over the
# policy's own stand-FD eval replays (the on-policy, in-distribution
# proxy). Score: per-decision d^2 on held-out replays.
#
# Validations this was built for (2026-08-04):
#   (a) absorber entry — does the score rise on YS platform frames / before
#       spell entry, separating absorbed from good runs? (W1 second angle)
#   (b) human gap — are human-game states further OOD for ms_g6_sp1
#       (human-zero) than for ms_g4_d2mix (human-best)?
#
# Floor test (GOTCHA #79): fit replays scored against their own fit must
# come out LOW (chi^2(256) ~ 256 +/- 23) — printed as `self` rows.
#
# Netplay replays scramble ports per session; --bot-code resolves the
# bot's port per replay from PlayerMeta.netplay_code (fullwidth # aware).
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_ood_score.exs \
#     --policy checkpoints/ms_g4_d2mix.bin --delay-id 3 \
#     --fit "eval_runs/0804_stage_final_destination/r*.slp" \
#     --score "eval_runs/0804_stage_yoshis_story/r*.slp,eval_runs/0804_direct_acab_g4/2026-08-Mainline/*.slp" \
#     [--bot-code "EXPH#288"] [--ridge 0.05] [--out eval_runs/interp/ood_g4.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      fit: :string,
      score: :string,
      delay_id: :integer,
      bot_code: :string,
      ridge: :float,
      out: :string
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
fit_globs = opts[:fit] || raise "--fit required"
score_globs = opts[:score] || ""
delay_id = opts[:delay_id]
ridge = opts[:ridge] || 0.05
out_path = opts[:out]

expand = fn globs ->
  globs |> String.split(",", trim: true) |> Enum.flat_map(&Path.wildcard/1) |> Enum.sort()
end

fit_paths = expand.(fit_globs)
if fit_paths == [], do: raise("--fit matched nothing")

normalize_code = fn c ->
  c |> to_string() |> String.replace("＃", "#") |> String.upcase() |> String.trim()
end

bot_port_for = fn path ->
  with code when code != nil <- opts[:bot_code],
       {:ok, replay} <- Peppi.parse(path),
       players when is_list(players) <- replay.metadata.players,
       %{port: port} <-
         Enum.find(players, fn p ->
           normalize_code.(p.netplay_code || "") == normalize_code.(code)
         end) do
    port
  else
    _ -> 1
  end
end

Output.banner("Trunk OOD score (Mahalanobis)")
Output.config([
  {"Policy", policy_path},
  {"Fit replays", length(fit_paths)},
  {"Score globs", score_globs},
  {"Ridge", ridge}
])

trunk = Activations.load_trunk(policy_path)
window = trunk.window
score_paths = expand.(score_globs)

capture = fn path ->
  port = bot_port_for.(path)
  opp = if port == 1, do: 2, else: 1

  %{activations: acts} =
    Activations.capture_replay(trunk, path,
      labels: false,
      player_port: port,
      opponent_port: opp,
      delay_id: delay_id
    )

  {Nx.backend_transfer(acts, Nx.BinaryBackend), port}
end

# Truncated netplay replays (SD-flake / 1-frame) crash capture — skip.
long_enough? = fn path ->
  case Peppi.parse(path) do
    {:ok, r} ->
      n = r |> Peppi.to_training_frames(player_port: 1, opponent_port: 2) |> length()

      if n < window + 30 do
        Output.warning("skip #{Path.basename(path)} (#{n} frames < window+30)")
        false
      else
        true
      end

    _ ->
      Output.warning("skip #{Path.basename(path)} (unparseable)")
      false
  end
end

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
fit_acts =
  fit_paths
  |> Enum.map(fn p ->
    {a, _} = capture.(p)
    a
  end)

fit_all = Nx.concatenate(fit_acts, axis: 0)
{n_fit, dim} = Nx.shape(fit_all)
Output.puts("Fit activations: {#{n_fit}, #{dim}}")

mu = Nx.mean(fit_all, axes: [0])
centered = Nx.subtract(fit_all, Nx.new_axis(mu, 0))
cov = Nx.divide(Nx.dot(Nx.transpose(centered), centered), n_fit - 1)

# Ridge scaled to mean variance; inversion on BinaryBackend (small matrix;
# EXLA decompositions can hang in XLA compile — CLAUDE.md).
mean_var = Nx.mean(Nx.take_diagonal(cov)) |> Nx.to_number()
eye = Nx.eye(dim)
cov_r = Nx.add(cov, Nx.multiply(eye, ridge * mean_var))

# Cholesky precision (covr = L L^T => covr^-1 = L^-T L^-1), f64 on
# BinaryBackend: Nx.LinAlg.invert NaNs on this matrix even in f64 (no
# pivoting on the near-singular covariance), and EXLA decompositions can
# hang in XLA compile — the LEACE lesson, re-learned 2026-08-04.
prev_backend = Nx.default_backend(Nx.BinaryBackend)

l =
  cov_r
  |> Nx.backend_transfer(Nx.BinaryBackend)
  |> Nx.as_type(:f64)
  |> Nx.LinAlg.cholesky()

l_inv = Nx.LinAlg.triangular_solve(l, Nx.eye(dim, type: :f64), lower: true)
precision = Nx.dot(Nx.transpose(l_inv), l_inv)
Nx.default_backend(prev_backend)

# Score on EXLA in f64 (mixed backends crash inconsistently — keep all
# score-path tensors on one backend).
precision = Nx.backend_transfer(precision, EXLA.Backend)
mu64 = mu |> Nx.as_type(:f64) |> Nx.backend_transfer(EXLA.Backend)

score_fn = fn acts ->
  c = Nx.subtract(Nx.as_type(acts, :f64), Nx.new_axis(mu64, 0))
  Nx.sum(Nx.multiply(Nx.dot(c, precision), c), axes: [1]) |> Nx.to_flat_list()
end

pct = fn sorted, p -> Enum.at(sorted, min(trunc(p * length(sorted)), length(sorted) - 1)) end

report = fn scores, label, extra ->
  s = Enum.sort(scores)

  Output.puts(
    "  #{String.pad_trailing(label, 34)} n=#{length(s)} p50=#{round(pct.(s, 0.5))} " <>
      "p95=#{round(pct.(s, 0.95))} max=#{round(List.last(s))}#{extra}"
  )

  %{label: label, n: length(s), p50: pct.(s, 0.5), p95: pct.(s, 0.95), max: List.last(s)}
end

Output.puts("")
Output.puts("Floor test (self-fit; chi^2(#{dim}) baseline ~#{dim}):")

self_rows =
  Enum.zip(fit_paths, fit_acts)
  |> Enum.map(fn {p, a} -> report.(score_fn.(a), "self #{Path.basename(p)}", "") end)

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("Scored replays:")

rows =
  for path <- score_paths, long_enough?.(path) do
    {acts, port} = capture.(path)
    scores = score_fn.(acts) |> Enum.reject(&(is_atom(&1) or &1 != &1))

    # Platform/ground split when y data warrants (offline stage runs)
    {:ok, replay} = Peppi.parse(path)

    ys =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.game_state.players[port].y)

    aligned = Enum.zip(scores, Enum.drop(ys, window - 1))
    plat = for {s, y} <- aligned, y > 15, do: s
    gnd = for {s, y} <- aligned, y <= 15, do: s

    extra =
      if length(plat) > 30 do
        pm = Enum.sum(plat) / length(plat)
        gm = Enum.sum(gnd) / max(length(gnd), 1)
        " | plat_mean=#{round(pm)} (n=#{length(plat)}) ground_mean=#{round(gm)}"
      else
        ""
      end

    tag = if port != 1, do: " [port #{port}]", else: ""

    label =
      path
      |> Path.split()
      |> Enum.drop(1)
      |> Enum.reject(&(&1 == "2026-08-Mainline"))
      |> Enum.join("/")

    report.(scores, "#{label}#{tag}", extra)
  end

if out_path do
  File.mkdir_p!(Path.dirname(out_path))
  File.write!(out_path, Jason.encode!(%{self: self_rows, scored: rows}))
  Output.success("Wrote #{out_path}")
end
