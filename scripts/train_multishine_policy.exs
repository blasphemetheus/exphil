# Hunt #16 (live-loop closure): train a policy to memorize the recorded Fox
# multishine replay and export it as a loadable policy .bin.
#
# The exported policy is then driven live via play_dolphin_async.exs — if the
# deployed loop (bridge gamestate → agent embed → decode → controller) is
# correct, the bot shines periodically in a real game; a broken seam shows as
# neutral/garbage. Afterward, the Slippi recording of THAT session can be
# parsed to count shine inputs objectively.
#
# Usage: mix run scripts/train_multishine_policy.exs

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.{Data, Imitation, Output}
alias ExPhil.Data.Peppi
alias ExPhil.Embeddings

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, out: :string, robust: :boolean, prev_action: :boolean, action_delay: :integer, prev_action_dropout: :float, window: :integer, synth_recovery: :boolean, synth_max_af: :integer, synth_ratio: :float, synth_crouch: :boolean, crouch_max_af: :integer, crouch_ratio: :float, synth_ledge: :boolean, ledge_strategy: :string, ledge_ratio: :float, ledge_max_af: :integer, noise: :float, noise_prob: :float, scheduled_sampling: :float, ss_ramp: :integer, probe_basin: :boolean, probe_every: :integer, probe_entries: :string, reject_at: :integer, reject_on: :string, select_by: :string, post_converge: :integer, x_hold_extend: :integer, synth_opening: :boolean, opening_replays: :string]
  )

# --replays accepts a dir or glob of .slp files; default = the single
# canonical fixture (pipeline-gate mode). --robust enables regularization
# for the generalization experiment (task #20) instead of pure memorization.
replay_glob = opts[:replays] || "test/fixtures/replays/fox_multishine_closed.slp"
out_path = opts[:out] || "checkpoints/multishine_probe_policy.bin"
robust = opts[:robust] || false
prev_action = opts[:prev_action] || false
prev_action_dropout = opts[:prev_action_dropout] || 0.0
action_delay = opts[:action_delay] || 0
# Recurrent context length. Exposed for the exposure-bias context ablation
# (docs/planning/EXPOSURE_BIAS.md item 3): the policy fails live in states it
# has seen pointwise but reached through an unseen 16-frame history, so a
# shorter context may be more robust off-trajectory.
window = opts[:window] || 16

Output.banner("Multishine Probe Policy Trainer")

replay_paths =
  if File.dir?(replay_glob) do
    Path.wildcard(Path.join(replay_glob, "*.slp"))
  else
    Path.wildcard(replay_glob)
  end

Output.puts("Replays: #{length(replay_paths)}")

frames =
  replay_paths
  |> Enum.flat_map(fn path ->
    {:ok, replay} = Peppi.parse(path)

    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    # Drop the SD tail (recorder holds pure left with no buttons to end the
    # game) — filter by that exact input signature rather than a fixed count.
    # Second signature (2026-07-28): the LRAS quit — A+L+R held with a
    # neutral stick (Start isn't recorded in parsed controller state).
    |> Enum.reject(fn %{controller: c} ->
      (c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
         not c.button_b and not c.button_x) or
        (c.button_a and c.button_l and c.button_r and
           abs(c.main_stick.x - 0.5) < 0.1)
    end)
  end)

# Both synthesis modes source segments and lead-ins from the REAL fixture
# frames, not from each other's output.
base_frames = frames

# --synth-recovery: manufacture the off-trajectory states the policy actually
# falls into and let the expert label them (EXPOSURE_BIAS.md item 2). The
# fixture has grounded reflector only at af 1..2; live the policy sits at af
# 3..28 and never escapes. No rollouts or Dolphin needed.
frames =
  if opts[:synth_recovery] do
    synth =
      ExPhil.Data.RecoverySynth.build(frames,
        port: 1,
        max_af: opts[:synth_max_af] || 30,
        lead_in: window,
        ratio: opts[:synth_ratio] || 1.0
      )

    Output.puts("Synthetic recovery frames: #{length(synth)}")
    frames ++ synth
  else
    frames
  end

# --synth-crouch: manufacture the crouch absorber (EXPOSURE_BIAS
# 6-replication) — a state the teacher NEVER visits, so build/2 can't reach
# it. Grafts Squat -> SquatWait tails onto post-shine frames, labelled
# "start a shine" with edge alternation.
frames =
  if opts[:synth_crouch] do
    crouch_synth =
      ExPhil.Data.RecoverySynth.build_crouch(base_frames,
        port: 1,
        max_af: opts[:crouch_max_af] || 40,
        lead_in: window,
        ratio: opts[:crouch_ratio] || 0.5
      )

    Output.puts("Synthetic crouch frames: #{length(crouch_synth)}")
    frames ++ crouch_synth
  else
    frames
  end

# --synth-ledge: manufacture the ledge valley (EXPOSURE_BIAS item 8's "next
# basin") — CliffCatch -> CliffWait tails at the FD edge, escape labelled by
# LedgeExpert (--ledge-strategy getup|attack|roll|jump|drop_jump).
frames =
  if opts[:synth_ledge] do
    ledge_synth =
      ExPhil.Data.RecoverySynth.build_ledge(base_frames,
        port: 1,
        max_af: opts[:ledge_max_af] || 30,
        lead_in: window,
        ratio: opts[:ledge_ratio] || 0.3,
        strategy: String.to_atom(opts[:ledge_strategy] || "getup")
      )

    Output.puts("Synthetic ledge frames: #{length(ledge_synth)}")
    frames ++ ledge_synth
  else
    frames
  end

# --synth-opening: cover the GAME-OPENING route (spawn-platform fall ->
# crouch), the absorber that kills ~half of otherwise-competent seeds
# (INIT_FORENSICS_OPTIONS.md 2026-07-28: m/g/p/r/s all absorb at frame ~104
# via entry 324x20 > 29x10 > 42x30, a history the fixture's post-shine
# lead-ins never cover). Grafts expert-labelled crouch tails onto opening
# lead-ins from the fixture itself, plus (--opening-replays GLOB) the real
# absorbed openings harvested from dead seeds' replays.
frames =
  if opts[:synth_opening] do
    extra_frames =
      case opts[:opening_replays] do
        nil ->
          []

        glob ->
          glob
          |> Path.wildcard()
          |> Enum.flat_map(fn path ->
            case ExPhil.Data.Peppi.parse(path) do
              {:ok, replay} ->
                replay
                |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
                |> Enum.reject(&(&1.game_state.frame < 0))

              _ ->
                []
            end
          end)
      end

    opening_synth =
      ExPhil.Data.RecoverySynth.build_opening(base_frames,
        port: 1,
        max_af: opts[:crouch_max_af] || 40,
        lead_in: window,
        extra_sources: extra_frames
      )

    Output.puts("Synthetic opening frames: #{length(opening_synth)}")
    frames ++ opening_synth
  else
    frames
  end

# --x-hold-extend N: widen the jump-cancel X press from a 2-frame spike
# into a hold that persists through jumpsquat (release stays before
# landing, preserving the landing X-edge the cycle pivots on). Motivation
# (INIT_FORENSICS 2026-07-28): sustainers hold X through the JC window,
# breakers emit razor spikes that evaporate under one frame of drift —
# make the fat hold the LABEL so spikes no longer fit the data.
frames =
  case opts[:x_hold_extend] || 0 do
    0 ->
      frames

    n_extend ->
      jumpsquat = 24
      arr = List.to_tuple(frames)
      total = tuple_size(arr)

      extended =
        Enum.map(0..(total - 1), fn i ->
          f = elem(arr, i)

          if f.controller.button_x do
            f
          else
            # Should this released-X frame become held? Yes if an X press
            # ended within the previous n_extend frames (consecutive frame
            # numbers only — never across synthesis block boundaries) and
            # we are still in jumpsquat.
            in_jumpsquat = f.game_state.players[1].action == jumpsquat

            recently_pressed =
              in_jumpsquat and
                Enum.any?(1..n_extend, fn back ->
                  j = i - back

                  j >= 0 and
                    elem(arr, j).controller.button_x and
                    elem(arr, j).game_state.frame == f.game_state.frame - back
                end)

            if recently_pressed do
              %{f | controller: %{f.controller | button_x: true}}
            else
              f
            end
          end
        end)

      changed = Enum.count(Enum.zip(frames, extended), fn {a, b} -> a != b end)
      Output.puts("X-hold extend (#{n_extend}f through jumpsquat): #{changed} frames widened")
      extended
  end

# --noise: DART-style state perturbation. Synthesis only covers EXTENSIONS of
# segments the fixture already visits; noise widens training into a tube
# AROUND the trajectory, so the two are complementary rather than redundant.
# Only continuous fields are perturbed (positions, percent, velocities) —
# never action state, buttons or flags, which would corrupt the label.
frames =
  case opts[:noise] do
    nil ->
      frames

    scale ->
      prob = opts[:noise_prob] || 0.5

      noised =
        Enum.map(frames, fn f ->
          ExPhil.Training.Augmentation.maybe_add_noise(f, scale: scale, probability: prob)
        end)

      Output.puts("Noise: scale=#{scale} p=#{prob} applied to #{length(noised)} frames")
      noised
  end

Output.puts("Training frames: #{length(frames)}")

# Kept as a named binding: the cycle-margin probe computes its event index
# on the SHIFTED stream so margins are label-aligned by construction.
shifted_frames = Data.shift_actions(frames, action_delay)

dataset =
  shifted_frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings(
    use_prev_action: prev_action,
    prev_action_dropout: prev_action_dropout
  )

embed_size = Embeddings.embedding_size(dataset.embed_config)

trainer =
  Imitation.new(
    embed_config: dataset.embed_config,
    use_prev_action: prev_action,
    embed_size: embed_size,
    temporal: true,
    backbone: :gru,
    window_size: window,
    hidden_size: 256,
    num_layers: 1,
    learning_rate: 5.0e-4,
    max_grad_norm: 0.5,
    # Gate mode: pure memorization (no regularization). Robust mode
    # (task #20): mild regularization — the goal is transfer, not replication.
    label_smoothing: if(robust, do: 0.05, else: 0.0),
    dropout: if(robust, do: 0.1, else: 0.0),
    # --scheduled-sampling P: self-conditioned prev-action on P of samples
    # (requires --prev-action; EXPOSURE_BIAS item 6). Ramped below.
    scheduled_sampling: opts[:scheduled_sampling] || 0.0
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

# Per-epoch probes — DEFAULT ON since 2026-08-02 (disable: --no-probe-basin).
# Two instruments, one JSONL row per probe epoch (<out>.basin_probe.jsonl):
#   1. Basin mental rollout (INIT_FORENSICS option 6 — watch the init
#      lottery get decided): escape@entry per entry, nil = absorbed.
#      Entries: synthetic training-style + seed g's real absorbed entry
#      when that replay exists.
#   2. Cycle margins on the fixture's own expert edges (jc/aerial/ground;
#      the 2026-07-28 sustain separator): jc_p10/jc_flip/crit_p10_min etc.
# --reject-at N (default off): from epoch N on, halt WITHOUT exporting and
# exit 3 when the --reject-on criterion holds ("basin" = all rollout
# entries absorbed; "margin" = jc_flip > 0.5 with n >= 5; "either").
# Farms need no changes: no .bin appears, so [ -f ... ] guards skip evals.
{basin_probe, probe_entry_names} =
  if opts[:probe_basin] != false do
    alias ExPhil.Interp.BasinRollout
    alias ExPhil.Interp.CycleMargins
    {_init, probe_predict} = ExPhil.Training.Utils.build_compiled(trainer.policy_model, mode: :inference)

    g_replay = "eval_runs/0727_crouch_g_idle/r1.slp"
    default_fixture = "test/fixtures/replays/fox_multishine_closed.slp"

    # BasinRollout embeds with use_prev_action: true — without --prev-action
    # the entry windows would not match the model's embed size. Margins use
    # dataset.embedded_frames (always config-matched), so they stay on.
    entries =
      if prev_action and File.exists?(default_fixture),
        do: [{"synthetic", BasinRollout.entry_synthetic()}],
        else: []

    # --probe-entries: comma list of globs of DEAD-seed replays; each
    # contributes an auto-detected absorbed entry (first sustained
    # SquatWait run — no hand-curated frame numbers). Task #3: every known
    # opening-death route becomes a foreign-hole probe. Defaults: the two
    # documented routes (g and m). Universal escapers exit foreign holes
    # (INIT_FORENSICS tier taxonomy); a seed absorbed in ANY hole is
    # early-reject material.
    default_entry_replays = "#{g_replay},eval_runs/0728_crouchfix_m_idle/r1.slp"

    entry_replays =
      (opts[:probe_entries] || default_entry_replays)
      |> String.split(",", trim: true)
      |> Enum.flat_map(&Path.wildcard/1)
      |> Enum.filter(&File.exists?/1)

    entries =
      entries ++
        if prev_action do
          Enum.flat_map(entry_replays, fn path ->
            tag = path |> Path.dirname() |> Path.basename() |> String.replace(~r/^\d+_/, "")

            case BasinRollout.entry_from_absorbed_replay(path) do
              {:ok, {entry, at}} -> [{"#{tag}@#{at}", entry}]
              :none -> []
            end
          end)
        else
          []
        end

    # Margin prep: event index + gathered windows, computed once. Cheap per
    # epoch (one batched forward over <=512 event windows).
    margin_prep =
      case CycleMargins.events(shifted_frames) do
        [] -> nil
        evs -> CycleMargins.prepare(dataset.embedded_frames, evs, window)
      end

    probe_path = out_path <> ".basin_probe.jsonl"
    File.rm(probe_path)

    n_events =
      case margin_prep do
        nil -> 0
        {_, kept} -> length(kept)
      end

    Output.puts("Probes: #{length(entries)} basin entries, #{n_events} margin events -> #{probe_path}")

    closure = fn epoch, loss, params ->
      params = ExPhil.Training.Utils.ensure_model_state(params)

      results =
        Map.new(entries, fn {name, entry} ->
          case BasinRollout.rollout(probe_predict, params, entry, max_frames: 120) do
            {:escape, t} -> {name, t}
            {:absorbed, _} -> {name, nil}
          end
        end)

      results =
        if margin_prep,
          do: Map.merge(results, CycleMargins.margins(probe_predict, params, margin_prep)),
          else: results

      line = Jason.encode!(Map.merge(%{epoch: epoch, loss: loss}, results))
      File.write!(probe_path, line <> "\n", [:append])
      results
    end

    {closure, Enum.map(entries, fn {name, _} -> name end)}
  else
    {nil, []}
  end

probe_every = opts[:probe_every] || 5
reject_at = opts[:reject_at]
reject_on = opts[:reject_on] || "basin"

# --select-by margin (task #4, the round-4 thin-margin lever): export the
# probe epoch with the FATTEST critical margin (max crit_p10_min) instead
# of the final/converged params — loss says "memorized" while margins say
# "barely" (d1 DAgger: loss 0.0006, margins 0.3-0.9, chains 2-4).
# --post-converge N keeps training N epochs past the loss bar so margins
# can fatten before selection. Default: loss (unchanged behavior).
select_by = opts[:select_by] || "loss"
post_converge = opts[:post_converge] || 0

if select_by not in ["loss", "margin"],
  do: raise(ArgumentError, "--select-by #{select_by} (want loss|margin)")

reject_check = fn results ->
  basin_fail =
    probe_entry_names != [] and Enum.all?(probe_entry_names, &is_nil(Map.get(results, &1)))

  margin_fail = (results["jc_flip"] || 0.0) > 0.5 and (results["jc_n"] || 0) >= 5

  case reject_on do
    "basin" -> basin_fail
    "margin" -> margin_fail
    "either" -> basin_fail or margin_fail
    other -> raise ArgumentError, "--reject-on #{other} (want basin|margin|either)"
  end
end

# Robust mode can't use an absolute loss bar: label smoothing imposes a
# FLOOR (≈0.33/categorical head at ε=0.05 → ~1.62 total) that no model can
# go below. Stop on plateau instead: <1e-3 improvement over 100 epochs.
memorized_loss = if robust, do: :plateau, else: 2.0e-3
max_epochs = if robust, do: 800, else: 2000

{trainer, final_loss, epochs_used, margin_best} =
  Enum.reduce_while(1..max_epochs, {trainer, nil, 0, [], nil, nil}, fn epoch,
                                                                      {tr, _, _, history,
                                                                       margin_best, converged_at} ->
    # Scheduled sampling ramp: 0 -> P over --ss-ramp epochs (default 10)
    tr =
      case opts[:scheduled_sampling] do
        nil ->
          tr

        p_max ->
          ramp = max(opts[:ss_ramp] || 10, 1)
          %{tr | config: Map.put(tr.config, :ss_p_current, min(p_max, p_max * epoch / ramp))}
      end

    {tr, epoch_loss} =
      dataset
      |> Data.batched_sequences(
        batch_size: 64,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: true,
        drop_last: false,
        seed: 42
      )
      |> Enum.reduce({tr, nil}, fn batch, {tr_acc, _} ->
        {tr_next, metrics} = Imitation.train_step(tr_acc, batch, loss_fn)
        {tr_next, metrics.loss}
      end)

    loss = Nx.to_number(epoch_loss)
    if rem(epoch, 25) == 0, do: Output.puts("epoch #{epoch}: loss=#{inspect(loss)}")

    probe_results =
      if basin_probe && (epoch == 1 or rem(epoch, probe_every) == 0) and is_number(loss) do
        results = basin_probe.(epoch, loss, tr.policy_params)

        if rem(epoch, 25) == 0 do
          Output.puts("  basin probe: #{inspect(results)}")
        end

        results
      end

    history = Enum.take([loss | history], 100)

    plateaued? =
      length(history) == 100 and is_number(loss) and
        List.last(history) - Enum.min(history) < 1.0e-3

    rejected? =
      reject_at != nil and epoch >= reject_at and is_map(probe_results) and
        reject_check.(probe_results)

    # Margin-best tracking: trainer structs are immutable, so holding the
    # reference pins that epoch's params (same idiom as the drill's `best`).
    margin_best =
      with true <- select_by == "margin",
           crit when is_number(crit) <- is_map(probe_results) && probe_results["crit_p10_min"] do
        case margin_best do
          {_, best_crit, _} when best_crit >= crit -> margin_best
          _ -> {tr, crit, epoch}
        end
      else
        _ -> margin_best
      end

    converged_now? =
      (memorized_loss == :plateau and plateaued?) or
        (is_number(memorized_loss) and is_number(loss) and loss < memorized_loss)

    converged_at = converged_at || if converged_now?, do: epoch

    acc = {tr, loss, epoch, history, margin_best, converged_at}

    cond do
      not is_number(loss) -> {:halt, acc}
      rejected? -> {:halt, {tr, {:rejected, loss, probe_results}, epoch, history, margin_best, converged_at}}
      converged_at != nil and epoch >= converged_at + post_converge -> {:halt, acc}
      true -> {:cont, acc}
    end
  end)
  |> then(fn {tr, loss, epoch, _, margin_best, _} -> {tr, loss, epoch, margin_best} end)

# Early-reject: no export, marker file, exit 3 (distinct from divergence's 1)
# — farms skip the Dolphin eval either way via the missing-.bin guard.
case final_loss do
  {:rejected, loss, probe_row} ->
    File.write!(
      out_path <> ".rejected.json",
      Jason.encode!(Map.merge(%{rejected_at_epoch: epochs_used, loss: loss, reject_on: reject_on}, probe_row))
    )

    Output.warning(
      "EARLY-REJECT (#{reject_on}) at epoch #{epochs_used}: #{inspect(probe_row)} — no policy exported"
    )

    System.halt(3)

  _ ->
    :ok
end

if not is_number(final_loss) do
  Output.error("Training diverged (#{inspect(final_loss)}) — no policy exported")
  System.halt(1)
end

Output.puts("Memorized: loss=#{Float.round(final_loss * 1.0, 5)} after #{epochs_used} epochs")

# --select-by margin: export the fattest-critical-margin probe epoch
# instead of the final params (falls back with a warning when no margin
# data was collected).
export_trainer =
  case {select_by, margin_best} do
    {"margin", {best_tr, crit, epoch}} ->
      Output.puts(
        "Margin-select: exporting epoch #{epoch} (crit_p10_min=#{crit}) " <>
          "over final epoch #{epochs_used}"
      )

      best_tr

    {"margin", nil} ->
      Output.warning("--select-by margin but no margin probes ran — exporting final params")
      trainer

    _ ->
      trainer
  end

case Imitation.export_policy(export_trainer, out_path) do
  :ok -> Output.success("Policy exported: #{out_path}")
  {:error, reason} ->
    Output.error("Export failed: #{inspect(reason)}")
    System.halt(1)
end
