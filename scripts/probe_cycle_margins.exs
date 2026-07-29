# Margin cartography over the multishine cycle (INIT_FORENSICS follow-up:
# the sustain question). For each seed, run its heads over its OWN live
# replay's windows and read the SIGNED logit margin toward the expert-correct
# button at each cycle phase. Hypothesis: per-cycle slip probability (~1/max
# chain) is set by margin thinness at the single-frame-critical decisions —
# X at the JC frame, B on the first airborne frame.
#
# Phases (expert rules from record_multishine.exs closed_loop):
#   ground_shine_hold  ground reflector af<=2   -> B hold    (+B logit)
#   jc_window          ground reflector af 3..4 -> X press   (+X logit)  CRITICAL
#   landed_reflector   ground reflector af>=5   -> X press   (+X logit)  CRITICAL
#   jumpsquat_early    jumpsquat af 1..2        -> B release (-B logit)
#   jumpsquat_late     jumpsquat af>=3          -> B press   (+B logit)  CRITICAL
#   airborne_no_shine  aerial jump family       -> B press   (+B logit)  CRITICAL
#   air_shine_hold     aerial reflector         -> B hold    (+B logit)
#
# Positive margin = agrees with the expert rule; magnitude = distance from
# the flip. Reports per-seed p10/min margins on critical phases and a
# Spearman rank correlation against measured live max chains.
#
# Usage: XLA_TARGET=cpu mix run scripts/probe_cycle_margins.exs \
#          [--policies "checkpoints/ms_crouch*.bin"] [--out eval_runs/interp/cycle_margins.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, out: :string, offset: :integer, replay: :string]
  )

policy_glob = opts[:policies] || "checkpoints/ms_crouch*.bin"
out_path = opts[:out] || "eval_runs/interp/cycle_margins.json"

# Bridge delay: controller[t] was decided from the window ending at t+OFFSET
# (live loop: read frame t-1 -> infer -> input lands on frame t, so -1).
# The script also prints a per-seed empirical calibration: B/X parity
# restricted to TRANSITION frames (constant stretches are alignment-blind)
# at offsets 0/-1/-2 — the margin tables use --offset (default -1).
offset = opts[:offset] || -1

# Measured live max chains (best across runs; INIT_FORENSICS + farm logs)
chains = %{
  "ms_crouch_a" => 22, "ms_crouch_b" => 2, "ms_crouch_c" => 11,
  "ms_crouch_d" => 2, "ms_crouch_e" => 1, "ms_crouch_f" => 3,
  "ms_crouch_g" => 0, "ms_crouch_h" => 2, "ms_crouch_i" => 3,
  "ms_crouch_j" => 2, "ms_crouch_k" => 4, "ms_crouch_l" => 3,
  "ms_crouchfix_m" => 0, "ms_crouchfix_n" => 18, "ms_crouchfix_o" => 3,
  "ms_crouchfix_p" => 0, "ms_crouchfix_q" => 21, "ms_crouchfix_r" => 0,
  "ms_crouchfix_s" => 1,
  "ms_rerec_a" => 5, "ms_shift_a" => 1, "ms_shift_b" => 1, "ms_shiftss_a" => 2, "ms_delay1_a" => 1, "ms_delay1_b" => 1, "ms_delay1_c" => 1,
  "ms_open_x" => 6, "ms_open_y" => 15, "ms_open_z" => 27, "ms_open_zz" => 1
}

replay_for = fn policy_path ->
  short =
    policy_path
    |> Path.basename(".bin")
    |> String.replace("ms_", "")

  Path.wildcard("eval_runs/*#{short}_idle*/r*.slp") |> Enum.sort() |> List.first()
end

# Margins are measured at the bot's own successful critical EVENTS (button
# EDGES — v1 of this script labeled every phase frame as "should press" and
# produced flip fractions near 1.0 for the champion: presses are
# single-frame events, holds are not presses). Events:
#   jc_event           X edge while in grounded reflector (the jump-cancel)
#   aerial_shine_event B edge while airborne without reflector (THE input)
#   ground_shine_event B edge while grounded, non-reflector (cycle re-entry)
event_of = fn player, ctrl, prev_ctrl ->
  x_edge = ctrl.button_x and not (prev_ctrl && prev_ctrl.button_x)
  b_edge = ctrl.button_b and not (prev_ctrl && prev_ctrl.button_b)

  case ShineChain.family(player.action) do
    :ground_reflect when x_edge -> :jc_event
    :aerial_jump when b_edge -> :aerial_shine_event
    :jumpsquat when b_edge -> :aerial_shine_event
    fam when fam not in [:ground_reflect, :air_reflect] and b_edge -> :ground_shine_event
    _ -> nil
  end
end

critical = [:jc_event, :aerial_shine_event]
event_btn = %{jc_event: :x, aerial_shine_event: :b, ground_shine_event: :b}

policies = Path.wildcard(policy_glob) |> Enum.sort()
Output.banner("Cycle margin cartography")

percentile = fn sorted, p ->
  n = length(sorted)
  Enum.at(sorted, min(trunc(p * n), n - 1))
end

results =
  policies
  |> Enum.map(fn path ->
    seed = Path.basename(path, ".bin")
    replay = replay_for.(path)

    # SD-flake replays are truncated and fail to parse — try each candidate.
    # --replay overrides the auto-derived path (e.g. to probe sync-runner
    # replays for offset calibration).
    candidates =
      case opts[:replay] do
        nil ->
          (replay && Path.wildcard("eval_runs/*#{String.replace(Path.basename(path, ".bin"), "ms_", "")}_idle*/r*.slp"))
          |> List.wrap()
          |> Enum.sort()

        override ->
          Path.wildcard(override) |> Enum.sort()
      end

    parsed =
      Enum.find_value(candidates, fn p ->
        case Peppi.parse(p) do
          {:ok, parsed} -> {p, parsed}
          _ -> nil
        end
      end)

    if parsed == nil do
      Output.warning("#{seed}: no parseable replay, skipping")
      nil
    else
      {replay, parsed} = parsed

      frames =
        parsed
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      ds =
        frames
        |> Data.from_frames()
        |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

      emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
      {total, _} = Nx.shape(emb)
      loaded = Activations.load_heads(path)
      window = loaded.window

      logits =
        (window - 1)..(total - 1)
        |> Enum.chunk_every(512)
        |> Enum.flat_map(fn ts ->
          wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
          out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
          buttons = elem(out, 0)
          b = buttons[[.., 1]] |> Nx.to_flat_list()
          x = buttons[[.., 2]] |> Nx.to_flat_list()
          Enum.zip(b, x)
        end)

      frames_arr = List.to_tuple(frames)
      logits_arr = List.to_tuple(logits)
      logit_at = fn t -> if t >= window - 1 and t < total, do: elem(logits_arr, t - (window - 1)) end

      # Offset calibration: transition-frame parity at candidate offsets.
      transitions =
        for t <- window..(total - 1),
            f = elem(frames_arr, t),
            p = elem(frames_arr, t - 1),
            f.controller.button_b != p.controller.button_b or
              f.controller.button_x != p.controller.button_x,
            do: t

      cal =
        Map.new([0, -1, -2, -3, -4], fn off ->
          hits =
            Enum.count(transitions, fn t ->
              case logit_at.(t + off) do
                nil ->
                  false

                {b, x} ->
                  f = elem(frames_arr, t)
                  (b > 0.0) == f.controller.button_b and (x > 0.0) == f.controller.button_x
              end
            end)

          {off, Float.round(hits / max(length(transitions), 1), 3)}
        end)

      by_phase =
        Enum.reduce((window - 1)..(total - 1), %{}, fn t, acc ->
          f = elem(frames_arr, t)
          prev_f = if t > 0, do: elem(frames_arr, t - 1)
          player = f.game_state.players[1]

          with event when event != nil <-
                 event_of.(player, f.controller, prev_f && prev_f.controller),
               {b, x} <- logit_at.(t + offset) do
            logit = if event_btn[event] == :b, do: b, else: x
            Map.update(acc, event, [logit], &[logit | &1])
          else
            _ -> acc
          end
        end)

      stats =
        Map.new(by_phase, fn {phase, margins} ->
          sorted = Enum.sort(margins)

          {phase,
           %{
             n: length(margins),
             mean: Float.round(Enum.sum(margins) / length(margins), 3),
             p10: Float.round(percentile.(sorted, 0.10) * 1.0, 3),
             min: Float.round(hd(sorted) * 1.0, 3),
             flip_frac: Float.round(Enum.count(margins, &(&1 < 0)) / length(margins), 3)
           }}
        end)

      crit_p10s = critical |> Enum.map(&get_in(stats, [&1, :p10])) |> Enum.reject(&is_nil/1)
      crit_score = if crit_p10s == [], do: nil, else: Float.round(Enum.min(crit_p10s) * 1.0, 3)

      Output.puts(
        "#{String.pad_trailing(seed, 16)} chains=#{String.pad_trailing(to_string(chains[seed]), 4)} " <>
          "cal=#{inspect(cal)} crit_p10_min=#{inspect(crit_score)} " <>
          Enum.map_join(critical, " ", fn ph ->
            case stats[ph] do
              nil -> "#{ph}=-"
              s -> "#{ph}: p10=#{s.p10} flip=#{s.flip_frac}"
            end
          end)
      )

      {seed, %{replay: replay, chains: chains[seed], crit_score: crit_score, phases: stats}}
    end
  end)
  |> Enum.reject(&is_nil/1)
  |> Map.new()

# Spearman rank correlation: crit_score vs chains (seeds with both)
pairs =
  results
  |> Enum.filter(fn {_, v} -> v.crit_score != nil and v.chains != nil end)
  |> Enum.map(fn {_, v} -> {v.crit_score, v.chains} end)

rank = fn values ->
  values
  |> Enum.with_index()
  |> Enum.sort_by(&elem(&1, 0))
  |> Enum.with_index(1)
  |> Enum.sort_by(fn {{_, orig}, _} -> orig end)
  |> Enum.map(fn {{_, _}, r} -> r end)
end

if length(pairs) > 3 do
  {xs, ys} = Enum.unzip(pairs)
  rx = rank.(xs)
  ry = rank.(ys)
  n = length(rx)
  d2 = Enum.zip(rx, ry) |> Enum.map(fn {a, b} -> (a - b) * (a - b) end) |> Enum.sum()
  rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
  Output.puts("Spearman(crit_p10_min, max_chain) over #{n} seeds: #{Float.round(rho, 3)}")
end

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(results, pretty: true))
Output.success("Written: #{out_path}")
