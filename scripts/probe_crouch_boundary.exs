# Decision-boundary cartography for the crouch absorber (INIT_FORENSICS_OPTIONS
# option 1 + 1b). For each policy: construct the SAME crouch-basin block the
# training synthesis produces (16 real lead-in frames -> Squat x2 -> SquatWait
# af 1..40), embed it under three prev-action regimes, and read the buttons
# head's B logit at every basin depth. One forward pass per variant per seed —
# the policy's opinion about the basin, measured offline in seconds.
#
# Prev-action variants (the train/deploy gap made explicit):
#   train  - frames as build_crouch emits them: tail frames share the source
#            frame NUMBER, so precompute_frame_embeddings' consecutive-frame
#            check fails and the prev channel embeds as ABSENT — this is what
#            training actually saw (discovered 2026-07-27; verify with
#            --dump-prev).
#   live_absorbed - consecutive frame numbers, tail controllers = constant
#            stick-down-no-B: what an absorbed policy feeds itself.
#   live_expert   - consecutive frame numbers, expert's alternating labels as
#            controllers: what an escaping policy feeds itself.
#
# T-calibration (option 1b): live decode fires B per frame with
# p = sigmoid(logit_B / T) (agents/agent.ex sample_buttons). From the measured
# logits we predict expected frames-to-first-B per temperature and compare to
# the measured flat rescue curve (EXPOSURE_BIAS item 9).
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_crouch_boundary.exs \
#     [--policies "checkpoints/ms_crouch_*.bin"] [--fixture path.slp] \
#     [--out eval_runs/interp/boundary_map.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.{Peppi, RecoverySynth}
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, fixture: :string, out: :string, dump_prev: :boolean]
  )

policy_glob = opts[:policies] || "checkpoints/ms_crouch_*.bin"
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
out_path = opts[:out] || "eval_runs/interp/boundary_map.json"

policies = Path.wildcard(policy_glob) |> Enum.sort()
if policies == [], do: raise("no policies match #{policy_glob}")

Output.banner("Crouch-basin boundary probe")
Output.config([{"Policies", length(policies)}, {"Fixture", fixture}, {"Out", out_path}])

# ---------------------------------------------------------------------------
# 1. Fixture frames, filtered exactly like train_multishine_policy.exs
# ---------------------------------------------------------------------------
{:ok, replay} = Peppi.parse(fixture)

frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
      not c.button_b and not c.button_x
  end)

# One synthetic block, same construction as training (lead_in 16, max_af 40).
# Tiny ratio -> budget forces a single block.
block =
  RecoverySynth.build_crouch(frames,
    port: 1,
    max_af: 40,
    lead_in: 16,
    ratio: 0.001
  )

n = length(block)
Output.puts("Probe block: #{n} frames (expect 58 = 16 lead + 2 squat + 40 squat_wait)")

# Position -> basin label. Lead-in occupies 0..(n-42-1); tail is the last 42.
tail_start = n - 42

state_of = fn j ->
  cond do
    j < tail_start -> {:lead, j - tail_start}
    j < tail_start + 2 -> {:squat, j - tail_start + 1}
    true -> {:squat_wait, j - tail_start - 1}
  end
end

# ---------------------------------------------------------------------------
# 2. Prev-action variants
# ---------------------------------------------------------------------------
down_no_b = %ExPhil.Bridge.ControllerState{
  main_stick: %{x: 0.5, y: 0.0},
  c_stick: %{x: 0.5, y: 0.5},
  l_shoulder: 0.0,
  r_shoulder: 0.0,
  button_a: false,
  button_b: false,
  button_x: false,
  button_y: false,
  button_z: false,
  button_l: false,
  button_r: false
}

renumber = fn frames ->
  base = hd(frames).game_state.frame

  frames
  |> Enum.with_index()
  |> Enum.map(fn {f, i} -> %{f | game_state: %{f.game_state | frame: base + i}} end)
end

down_held_b = %{down_no_b | button_b: true}

variants = %{
  train: block,
  live_expert: renumber.(block),
  live_absorbed:
    block
    |> renumber.()
    |> Enum.with_index()
    |> Enum.map(fn {f, j} ->
      if j >= tail_start, do: %{f | controller: down_no_b}, else: f
    end),
  # The REAL absorbed self-feed (discovered via probe v2 parity on seed g's
  # replay): the stuck policies HOLD B — Melee registers edges, so held B is
  # a no-op and "press B forever" is behaviorally identical to never
  # pressing. Escape requires a RELEASE. This variant asks: with prev = B
  # held, does the policy release?
  live_held:
    block
    |> renumber.()
    |> Enum.with_index()
    |> Enum.map(fn {f, j} ->
      if j >= tail_start, do: %{f | controller: down_held_b}, else: f
    end)
}

# ---------------------------------------------------------------------------
# 3. Embed each variant, build sliding windows {n-15, 16, embed}
# ---------------------------------------------------------------------------
embed_windows = fn frames ->
  ds =
    frames
    |> Data.from_frames()
    |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

  emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
  {total, _embed} = Nx.shape(emb)

  windows =
    for j <- 15..(total - 1) do
      Nx.slice_along_axis(emb, j - 15, 16, axis: 0)
    end

  {Nx.stack(windows), 15..(total - 1)}
end

embedded = Map.new(variants, fn {name, fr} -> {name, embed_windows.(fr)} end)

if opts[:dump_prev] do
  # Verify the train-variant prev channel really is absent on tail frames:
  # diff the train vs live_absorbed embeddings on the last frame of the
  # deepest window (pure SquatWait state, controllers identical by content
  # except threading).
  {tw, _} = embedded.train
  {aw, _} = embedded.live_absorbed
  last_t = tw[[-1, -1]] |> Nx.to_flat_list()
  last_a = aw[[-1, -1]] |> Nx.to_flat_list()

  diff_dims =
    Enum.zip(last_t, last_a)
    |> Enum.with_index()
    |> Enum.filter(fn {{a, b}, _} -> a != b end)
    |> Enum.map(fn {{a, b}, i} -> {i, Float.round(a, 3), Float.round(b, 3)} end)

  Output.puts("train vs live_absorbed deepest-frame dims differing: #{inspect(diff_dims)}")
end

# ---------------------------------------------------------------------------
# 4. Per policy: B logits per window per variant
# ---------------------------------------------------------------------------
temps = [0.3, 0.5, 1.0]

results =
  Enum.map(policies, fn path ->
    seed = Path.basename(path, ".bin")
    loaded = Activations.load_heads(path)

    per_variant =
      Map.new(embedded, fn {vname, {windows, idx_range}} ->
        out = loaded.predict_fn.(loaded.params, windows)
        buttons = elem(out, 0)
        b_logits = buttons[[.., 1]] |> Nx.to_flat_list()

        rows =
          Enum.zip(Enum.to_list(idx_range), b_logits)
          |> Enum.map(fn {j, logit} ->
            {state, af} = state_of.(j)
            %{state: state, af: af, b_logit: Float.round(logit * 1.0, 4)}
          end)

        # Escape stats over the basin (squat + squat_wait windows only)
        basin = Enum.filter(rows, &(&1.state in [:squat, :squat_wait]))
        det_fire = Enum.count(basin, &(&1.b_logit > 0.0)) / max(length(basin), 1)

        # Sequential escape-time prediction per T: walking the tail frame by
        # frame, P(first B at frame k) — report expected frames-to-first-B.
        expected_escape = fn t ->
          {exp_frames, p_none} =
            Enum.reduce(Enum.with_index(basin, 1), {0.0, 1.0}, fn {row, k}, {acc, alive} ->
              p = 1.0 / (1.0 + :math.exp(-row.b_logit / t))
              {acc + k * alive * p, alive * (1.0 - p)}
            end)

          # If the tail runs out with probability p_none, count it as > tail
          # length (censored) — report both.
          %{expected_frames: Float.round(exp_frames + p_none * length(basin), 1),
            p_still_absorbed_after_tail: Float.round(p_none, 4)}
        end

        {vname,
         %{
           rows: rows,
           deterministic_fire_fraction: Float.round(det_fire, 3),
           mean_basin_logit:
             Float.round(Enum.sum(Enum.map(basin, & &1.b_logit)) / max(length(basin), 1), 3),
           escape_by_temp: Map.new(temps, fn t -> {t, expected_escape.(t)} end)
         }}
      end)

    # RELEASE CONDITIONING — the escape-competence scalar. Deep-basin
    # (SquatWait af >= 10) mean B logit with prev = no-B minus prev = held-B.
    # A policy that presses on no-B and RELEASES on held-B (positive gap,
    # sign flip) sustains the edge cycle = multishine escape. A policy with
    # gap ~0 holds B forever = the absorber.
    deep = fn v ->
      per_variant[v].rows
      |> Enum.filter(&(&1.state == :squat_wait and &1.af >= 10))
      |> then(fn rs -> Enum.sum(Enum.map(rs, & &1.b_logit)) / max(length(rs), 1) end)
    end

    gap = Float.round(deep.(:live_absorbed) - deep.(:live_held), 3)
    releases? = deep.(:live_held) < 0.0

    per_variant =
      Map.put(per_variant, :release_conditioning, %{
        gap: gap,
        deep_logit_prev_no_b: Float.round(deep.(:live_absorbed), 3),
        deep_logit_prev_held_b: Float.round(deep.(:live_held), 3),
        releases_on_held: releases?
      })

    Output.puts(
      "#{String.pad_trailing(seed, 14)} gap=#{gap} held->#{Float.round(deep.(:live_held), 2)} " <>
        "noB->#{Float.round(deep.(:live_absorbed), 2)} releases=#{releases?}  " <>
        Enum.map_join([:train, :live_expert], "  ", fn v ->
          m = per_variant[v]
          "#{v}: fire=#{m.deterministic_fire_fraction}"
        end)
    )

    {seed, per_variant}
  end)
  |> Map.new()

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(results, pretty: true))
Output.success("Boundary map written: #{out_path}")
