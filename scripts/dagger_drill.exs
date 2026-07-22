# Generalized DAgger retrain for scripted-expert drills: aggregate a drill
# fixture with live rollout replays relabeled by the drill's expert, retrain.
# Same protocol as dagger_multishine.exs (which it generalizes): rollouts are
# ordinary Slippi replays of the policy playing; the expert corrects every
# visited frame; the policy's ACTUAL press rides in the prev-action channel.
#
#   mix run scripts/dagger_drill.exs --expert mewtwo_fair \
#     --rollouts "~/Slippi/Game_X.slp,~/Slippi/Game_Y.slp" \
#     --out checkpoints/mewtwo_fair_dagger1_policy.bin
#
# Experts:
#   multishine  - Fox multishine (ExPhil.Agents.MultishineExpert)
#   mewtwo_fair - Mewtwo SH/FH/DJ fair chains with L-cancels
#                 (ExPhil.Agents.MewtwoFairExpert)

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.{ConversionSampling, Data, Imitation, MemoryLedger, Output, ProbeRegularizer}
alias ExPhil.Data.Peppi
alias ExPhil.Embeddings

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      expert: :string,
      rollouts: :string,
      fixture: :string,
      out: :string,
      prev_action: :boolean,
      action_delay: :integer,
      prev_action_dropout: :float,
      port: :integer,
      lr: :float,
      backbone: :string,
      target_loss: :float,
      max_epochs: :integer,
      hidden_size: :integer,
      nan_forensics: :boolean,
      mixed_precision: :boolean,
      window: :integer,
      transition_weight: :float,
      conversion_weight: :float,
      probe_reg: :float,
      probe_reg_every: :integer,
      probe_eval_every: :integer,
      bc_replays: :string,
      bc_sample: :integer,
      mamba_chunk_size: :integer,
      mamba_matmul_scan: :boolean,
      debug_grads_after: :integer,
      resume: :boolean,
      preflight: :boolean,
      memory_check: :string
    ]
  )

# expert_char: the expert's character's internal ID — fixtures auto-detect
# which port the expert player is on (recordings land on whichever port the
# controller happened to claim; observed: same drill recorded P1 one day,
# P2 the next after an adapter re-plug)
{expert_mod, default_fixture, default_out, expert_char} =
  case opts[:expert] || "multishine" do
    "multishine" ->
      {ExPhil.Agents.MultishineExpert, "test/fixtures/replays/fox_multishine_closed.slp",
       "checkpoints/multishine_dagger_policy.bin", 1}

    "mewtwo_fair" ->
      # Varied recording (SH/FH/DJ fair) + dense metronomic SH-fair-only +
      # approach-fair (varied-distance dash-ins for the distance-keyed table)
      {ExPhil.Agents.MewtwoFairExpert,
       "test/fixtures/replays/mewtwo_fair_chains.slp," <>
         "test/fixtures/replays/mewtwo_shfair_only.slp," <>
         "test/fixtures/replays/mewtwo_approach_fair.slp," <>
         "test/fixtures/replays/mewtwo_turnaround_fair.slp," <>
         "test/fixtures/replays/mewtwo_oos_chains.slp," <>
         "test/fixtures/replays/mewtwo_ground_neutral.slp",
       "checkpoints/mewtwo_fair_dagger_policy.bin", 16}

    "fox_recovery" ->
      # Rules-only (recovery is pure geometry): no fixture; every ordinary
      # replay with offstage moments is a rollout to relabel
      {ExPhil.Agents.FoxRecoveryExpert, nil, "checkpoints/fox_recovery_dagger_policy.bin", 1}

    "mewtwo_techchase" ->
      # Rules-only REACTION drill: reads the opponent's tech choice from
      # their action state. Rollouts = games vs --dummy tech_random.
      {ExPhil.Agents.MewtwoTechChaseExpert, nil,
       "checkpoints/mewtwo_techchase_dagger_policy.bin", 16}

    "mewtwo_combo" ->
      # Composite: fair expert when the opponent stands, chase expert when
      # they're down — the full approach->fair->knockdown->punish cycle
      {ExPhil.Agents.MewtwoComboExpert,
       ExPhil.Agents.FixtureSets.mewtwo_combo_csv(),
       "checkpoints/mewtwo_combo_dagger_policy.bin", 16}

    other ->
      Output.error("Unknown expert #{inspect(other)} (multishine | mewtwo_fair | fox_recovery)")
      System.halt(1)
  end

# --fixture accepts a comma-separated list: multiple recordings of the same
# drill combine into one expert table (denser where they overlap, variance
# averaged out) and all contribute training frames. Rules-only experts
# (fox_recovery) have no fixture at all.
fixture_paths =
  (opts[:fixture] || default_fixture || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
out_path = opts[:out] || default_out
prev_action = Keyword.get(opts, :prev_action, true)
prev_action_dropout = opts[:prev_action_dropout] || 0.1
action_delay = Keyword.get(opts, :action_delay, 2)
port = opts[:port] || 1
# Constant LR blows up late as the aggregate grows; keep it modest (see
# dagger_multishine.exs — same convergence setup)
learning_rate = opts[:lr] || 2.0e-4

# --backbone tests whether drill conclusions transfer across architectures.
# Each backbone brings its OWN shape via Config.backbone_defaults — mamba
# under the GRU drill shape (window 16, 1 layer) diverges at epoch 4.
backbone = String.to_atom(opts[:backbone] || "gru")
bb_defaults = ExPhil.Training.Config.backbone_defaults(backbone) || []
# --window overrides bb_defaults (gru default is 60 — NOT 16; the || 16
# fallback below never fires for known backbones). Exploding-BPTT
# hypothesis (2026-07-13): both forensics runs detonated at param_max ~5.2
# regardless of precision — 60 backward Jacobians overflow at a critical
# weight scale. Shorter windows raise that ceiling steeply.
# Fallbacks match the common bb_defaults values (60/2) so they can't lie —
# the old `|| 16` never fired for any known backbone yet had the whole team
# believing "window 16" for a week (2026-07-13 discovery).
window = opts[:window] || bb_defaults[:window_size] || 60

rollout_paths =
  (opts[:rollouts] || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

if rollout_paths == [] do
  if fixture_paths == [] do
    Output.error("Fixture-less expert with no rollouts — nothing to train on. Pass --rollouts.")
    System.halt(1)
  end

  Output.warning(
    "No rollouts — bootstrap mode: training on the fixture alone " <>
      "(iteration 0 of a new drill). Play a game with the result and feed " <>
      "the replay back as the first rollout."
  )
end

Output.banner("Drill DAgger Trainer")

Output.config([
  {"Expert", inspect(expert_mod)},
  {"Fixtures", Enum.map(fixture_paths, &Path.basename/1)},
  {"Rollouts", length(rollout_paths)},
  {"Prev-action", prev_action},
  {"Prev-action dropout", prev_action_dropout},
  {"Action delay", action_delay},
  {"Conversion weight", opts[:conversion_weight] || "off"},
  {"Probe regularizer", if(opts[:probe_reg], do: "#{opts[:probe_reg]} (refit every #{opts[:probe_reg_every] || 5})", else: "off")},
  {"Out", out_path}
])

# Which port is the expert's character on in this replay? (nil char or no
# match falls back to the given default port)
detect_port = fn replay, default ->
  case replay |> Peppi.to_training_frames(player_port: default) |> Enum.take(1) do
    [f] ->
      Enum.find_value(f.game_state.players, default, fn {p, pl} ->
        if pl && trunc(pl.character || -1) == expert_char, do: p
      end)

    _ ->
      default
  end
end

load_frames = fn path, detect? ->
  # Archive + timeout-truncated seed .slp include corrupt/partial files
  # (2026-07-22). Skip them (-> []) rather than crash the whole run; the
  # BC min-frames filter and empty-rollout handling drop them downstream.
  case Peppi.parse(path) do
    {:error, reason} ->
      Output.warning("skipping unparseable replay #{Path.basename(path)}: #{inspect(reason)}")
      []

    {:ok, replay} ->
      use_port = if detect? and expert_char, do: detect_port.(replay, port), else: port

      if use_port != port do
        Output.puts("  #{Path.basename(path)}: expert character on port #{use_port} — normalizing")
      end

      frames =
        replay
        |> Peppi.to_training_frames(
          player_port: use_port,
          opponent_port: if(use_port == 1, do: 2, else: 1)
        )
        |> Enum.reject(&(&1.game_state.frame < 0))

      # Normalize so the expert is ALWAYS players[1] downstream — the shared
      # table build and relabeling assume one port across all sources
      if use_port == port do
        frames
      else
        Enum.map(frames, fn f ->
          gs = f.game_state
          swapped = %{1 => gs.players[use_port], 2 => gs.players[port]}
          %{f | game_state: %{gs | players: swapped}}
        end)
      end
  end
end

# The multishine fixture ends with an SD (recorder holds pure left, no
# buttons, to end the game) — those frames are junk training targets
fixture_filter = fn frames ->
  if expert_mod == ExPhil.Agents.MultishineExpert do
    Enum.reject(frames, fn %{controller: c} ->
      c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
        not c.button_b and not c.button_x
    end)
  else
    frames
  end
end

# Fixture: human = expert; recorded controllers are the labels. Kept as
# per-replay lists so shift_actions never crosses replay boundaries.
fixture_frame_lists = Enum.map(fixture_paths, fn p -> fixture_filter.(load_frames.(p, true)) end)
fixture_frames = List.flatten(fixture_frame_lists)
Output.puts("Fixture frames: #{length(fixture_frames)} across #{length(fixture_paths)} recording(s)")

# --bc-replays (r16): human demonstration replays as DIRECT behavior
# cloning — recorded controllers are the labels (port-normalized like
# fixtures), added to the training pool but NOT fed to the expert table.
# This supplies neutral/initiation vocabulary the scripted expert lacks
# (the 2026-07-19 "never initiates in neutral" gap) without polluting the
# teacher. --bc-sample N picks N replays at random (seeded) to control the
# BC fraction so human frames don't drown the drill signal.
bc_paths =
  (opts[:bc_replays] || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

bc_paths =
  case opts[:bc_sample] do
    nil -> bc_paths
    n -> :rand.seed(:exsss, {7, 7, 7}) && Enum.take(Enum.shuffle(bc_paths), n)
  end

# Archive quality varies — some replays parse empty/truncated (observed
# 2026-07-22 on corpus/archive/mewtwo). Skip anything under @bc_min_frames.
bc_min_frames = 600

bc_frame_lists =
  bc_paths
  |> Enum.map(fn p -> load_frames.(p, true) end)
  |> Enum.filter(fn frames -> length(frames) >= bc_min_frames end)

bc_frames = List.flatten(bc_frame_lists)

if bc_paths != [] do
  skipped = length(bc_paths) - length(bc_frame_lists)

  Output.puts(
    "BC replays: #{length(bc_frames)} frames across #{length(bc_frame_lists)} human games" <>
      if(skipped > 0, do: " (#{skipped} skipped: empty/short)", else: "")
  )
end

# Expert table from ALL fixture recordings combined (BC replays excluded)
expert = expert_mod.from_frames(fixture_frames, player_port: port)

opp_port = if port == 1, do: 2, else: 1

relabel = fn frames, recorded ->
  Enum.flat_map(frames, fn frame ->
    prev = recorded[frame.game_state.frame + action_delay - 1]

    case expert_mod.label(expert, frame.game_state.players[port], prev, frame.game_state.players[opp_port]) do
      {:ok, correction} ->
        [
          frame
          |> Map.put(:controller, correction)
          |> Map.put(:prev_controller, prev)
        ]

      :skip ->
        []
    end
  end)
end

button_sig = fn c ->
  {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r}
end

rollout_frame_lists =
  Enum.map(rollout_paths, fn path ->
    raw = load_frames.(path, false)
    recorded = Map.new(raw, fn f -> {f.game_state.frame, f.controller} end)
    relabeled = relabel.(raw, recorded)

    disagreements =
      Enum.count(relabeled, fn f ->
        button_sig.(f.controller) != button_sig.(recorded[f.game_state.frame])
      end)

    pct = Float.round(100.0 * disagreements / max(length(relabeled), 1), 1)
    Output.puts("  #{Path.basename(path)}: #{length(relabeled)} frames, #{pct}% corrected")

    relabeled
  end)

all_frame_lists = fixture_frame_lists ++ bc_frame_lists ++ rollout_frame_lists
all_frames = List.flatten(all_frame_lists)

bc_pct = Float.round(100.0 * length(bc_frames) / max(length(all_frames), 1), 1)

Output.puts(
  "Aggregate: #{length(all_frames)} frames " <>
    "(#{length(fixture_frames)} fixture, #{length(bc_frames)} BC = #{bc_pct}%)"
)

if bc_pct > 60.0 do
  Output.warning("BC frames are #{bc_pct}% of the pool — human demos may drown the drill; --bc-sample to reduce")
end

# --memory-check warn|strict|off (default warn): predict this pool's peak
# RSS from the ledger of past runs BEFORE paying the embed + JIT cost —
# a pool that won't fit should be refused here, in seconds, not die at
# hour 3. Every run appends its own (pool_frames, VmHWM) point on exit,
# so the model sharpens with each run.
pool_frames = length(all_frames)
memory_check = opts[:memory_check] || "warn"

if memory_check != "off" do
  case MemoryLedger.headroom_check(pool_frames, window: window) do
    {:no_data, _} ->
      Output.puts(
        "memory-check: no ledger entries yet (#{MemoryLedger.default_path()}) — " <>
          "prediction unavailable; this run records the first point"
      )

    {:ok, info} ->
      Output.puts(
        "memory-check: predicted peak #{MemoryLedger.format_bytes(info.predicted_bytes)} " <>
          "fits budget #{MemoryLedger.format_bytes(info.budget_bytes)} " <>
          "(#{info.model}, #{info.points} run(s), #{info.slope_bytes_per_frame} B/frame)"
      )

    {:warn, info} ->
      Output.warning(
        "memory-check: predicted peak #{MemoryLedger.format_bytes(info.predicted_bytes)} " <>
          "(x#{info.margin} margin) EXCEEDS budget #{MemoryLedger.format_bytes(info.budget_bytes)} " <>
          "(#{info.model} from #{info.points} run(s)) — this pool may OOM"
      )

      if memory_check == "strict" do
        Output.error(
          "--memory-check strict: refusing to start. Shrink the pool " <>
            "(--bc-sample / fewer rollouts) or free RAM."
        )

        System.halt(5)
      end
  end
end

# Shift per source replay — never across concat boundaries. Kept as
# per-replay lists so conversion spans (below) can't cross them either.
shifted_frame_lists =
  Enum.map(all_frame_lists, &Data.shift_actions(&1, action_delay))

shifted_frames = List.flatten(shifted_frame_lists)

# --conversion-weight W (r15): windows whose supervised frame falls in a
# converting-approach span (closure start -> opponent-hitstun payoff) are
# sampled ~W times per epoch instead of once — the go-in decisions the
# 2026-07-19 human demo showed the policy never learned. Computed on the
# SHIFTED lists so frame indices align with dataset.frames exactly.
sampling_weights =
  if cw = opts[:conversion_weight] do
    {weights, cstats} = ConversionSampling.frame_weights(shifted_frame_lists, cw)

    pct = Float.round(100.0 * cstats.upweighted / max(cstats.frames, 1), 1)

    Output.puts(
      "Conversion weighting: #{cstats.conversions}/#{cstats.approaches} approaches " <>
        "converted; #{cstats.upweighted}/#{cstats.frames} frames (#{pct}%) upweighted x#{cw}"
    )

    if cstats.conversions == 0 do
      Output.warning("--conversion-weight set but no conversions found — sampling is uniform")
    end

    weights
  end

dataset =
  shifted_frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings(
    use_prev_action: prev_action,
    prev_action_dropout: prev_action_dropout
  )

embed_size = Embeddings.embedding_size(dataset.embed_config)

post_embed_mem = MemoryLedger.process_memory()

Output.puts(
  "memory: post-embed rss=#{MemoryLedger.format_bytes(post_embed_mem.rss_bytes)} " <>
    "(#{round(post_embed_mem.rss_bytes / max(pool_frames, 1))} B/frame observed)"
)

# --probe-reg W [--probe-reg-every K] (r15, the steering-decided lever):
# penalize trunk alignment with the shield-lock direction during training.
# The direction refits from CURRENT activations every K epochs (fresh
# rounds can't reuse the r14 vector — wrong basis); until the first
# successful refit the direction is zero and training is plain BC.
probe_reg = opts[:probe_reg] || 0.0
probe_reg_every = opts[:probe_reg_every] || 5

probe_frame_labels =
  if probe_reg > 0 do
    labels = ProbeRegularizer.frame_labels(shifted_frames)
    shield_frac = Float.round(100.0 * Enum.sum(labels) / max(length(labels), 1), 1)
    Output.puts("Probe regularizer: #{shield_frac}% of pool frames are shield-family")
    labels
  end

# Cosine-decay the LR: constant LR diverged to NaN late in run after run
# (multishine at 90k frames, mewtwo at 31k) — more steps per epoch means more
# chances for one step to blow up. Decay over ~200 epochs' worth of steps
# (400 left LR too high too long: a 90k-frame pool still NaN'd at epoch 168);
# healthy runs hit the loss bar or plateau well before the floor.
steps_per_epoch = div(dataset.size, 64) + 1

trainer =
  Imitation.new(
    embed_config: dataset.embed_config,
    use_prev_action: prev_action,
    embed_size: embed_size,
    temporal: true,
    backbone: backbone,
    window_size: window,
    # --hidden-size: bake-off capacity screens (e.g. GRU 2x = 512); the
    # export config carries it, so agents/probes rebuild the right shape.
    hidden_size: opts[:hidden_size] || 256,
    num_layers: bb_defaults[:num_layers] || 2,
    state_size: bb_defaults[:state_size] || 16,
    expand_factor: bb_defaults[:expand_factor] || 2,
    conv_size: bb_defaults[:conv_size] || 4,
    # DELIBERATE divergences from bb_defaults (which also carry
    # precision: :f32 and dropout: 0.1 for recurrent backbones):
    #   precision — stays bf16 for speed; the NaN instability that
    #     motivated bb_defaults' f32 is fixed at the loss boundary
    #     (Policy.Loss.imitation_loss casts to f32, 2026-07-14).
    #   dropout 0.0 — drills WANT memorization of the expert table.
    learning_rate: learning_rate,
    lr_schedule: :cosine,
    warmup_steps: steps_per_epoch,
    decay_steps: steps_per_epoch * 200,
    max_grad_norm: 0.5,
    label_smoothing: 0.0,
    dropout: 0.0,
    # --mixed-precision: FP32 master weights + BF16 compute. (Tested
    # 2026-07-13: does NOT fix the NaN detonations — kept for future use.)
    mixed_precision: opts[:mixed_precision] || false,
    # --debug-grads-after N: from step N on, fused per-step grad finiteness
    # check; per-layer dump on the first non-finite step (GRAD_DETONATION).
    debug_grads_after: opts[:debug_grads_after],
    # --probe-reg: probe-as-regularizer weight (0.0 = off, r15)
    probe_reg_weight: probe_reg,
    # SSD scan tuning (mamba_2 only; bench_ssd_scan.exs picks values).
    # Both nil = the r15-lineage sequential path.
    chunk_size: opts[:mamba_chunk_size],
    training_mode: opts[:mamba_matmul_scan]
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

# --target-loss: stop as soon as loss drops below this. The default (2e-3)
# trains to memorization — but live scores peak at MODERATE loss (36.4%
# conversion from a 0.16-loss export, 23.1% from 0.08) and collapse to 0%
# at the floor (0.002-loss policy = statue, 2026-07-12). Pass ~0.08-0.15
# to export in the behaviorally-alive band.
memorized_loss = opts[:target_loss] || 2.0e-3
max_epochs = opts[:max_epochs] || 1000

# --resume: continue an interrupted run from OUT.trainer.ckpt (full trainer
# state incl. Adam moments, published atomically every 10 epochs below).
# The fingerprint refuses cross-experiment resumes loudly: same-named
# checkpoints with a different pool/recipe must retrain, not silently blend
# (built after reboot #2 killed r13 at epoch 70/100, 2026-07-18).
trainer_ckpt = out_path <> ".trainer.ckpt"

# --probe-eval-every K: feature-formation curves (TrainingProbes) — probe
# a small ground-truth feature set on the current trunk every K epochs,
# log one line + append JSONL next to the checkpoint. Trend instrument
# for the epoch budget; ~30-60s per eval. Off by default.
probe_eval_every = opts[:probe_eval_every] || 0

{probe_eval_trunk_fn, probe_eval_labels} =
  if probe_eval_every > 0 do
    labels =
      shifted_frames
      |> ExPhil.Interp.GroundTruth.frame_labels()
      |> :erlang.list_to_tuple()

    # ProbeRegularizer lives in Training, not Interp — the wrong namespace
    # here only detonated when --probe-eval-every ran WITHOUT --probe-reg
    # (only probe-reg builds trainer.trunk_predict_fn), which is why
    # r15/r16 (both flags on) never hit it.
    tfn = trainer.trunk_predict_fn || ProbeRegularizer.build_trunk_fn(trainer.config)
    Output.puts("Probe-eval every #{probe_eval_every} epochs (#{tuple_size(labels)} labeled frames)")
    {tfn, labels}
  else
    {nil, nil}
  end

probe_curves_path = out_path <> ".probe_curves.jsonl"

# One (pool_frames, peak RSS) point per run — the data the memory-check
# prediction above is fit from. VmHWM is the kernel high-water mark, so
# transient spikes (the epoch-boundary suspect class) are captured even
# without a sampler running at the right moment.
record_ledger = fn mode ->
  mem = MemoryLedger.process_memory()

  gpu_mb =
    case ExPhil.Training.GPUUtils.get_memory_info() do
      {:ok, info} -> info.used_mb
      _ -> nil
    end

  MemoryLedger.append(%{
    mode: mode,
    tag: Path.basename(out_path),
    pool_frames: pool_frames,
    embed_size: embed_size,
    window: window,
    backbone: backbone,
    peak_rss_bytes: mem.hwm_bytes,
    gpu_used_mb: gpu_mb
  })

  mem
end

# --preflight: full-pool dress rehearsal, no training. Exercises every
# stage that has ever killed a run at THIS run's exact pool and config —
# parse + embed (the RAM peak), one JIT-compiled train step (the GPU
# peak), probe-reg refit, probe-eval (the r16 killer, 2026-07-22),
# policy export + trainer snapshot — records the memory-ledger point,
# then exits 0. Minutes spent here instead of an overnight lost at
# epoch 10; launchers gate the real run on this exit code.
if opts[:preflight] do
  Output.banner("PREFLIGHT (dress rehearsal — no training)")
  preflight_t0 = System.monotonic_time(:millisecond)

  [first_batch] =
    dataset
    |> Data.batched_sequences(
      batch_size: 64,
      window_size: window,
      stride: 1,
      lazy: true,
      shuffle: true,
      drop_last: false,
      seed: 42,
      transition_weight: opts[:transition_weight],
      sampling_weights: sampling_weights
    )
    |> Enum.take(1)

  Output.puts("preflight: JIT compiling one train step (may take minutes)...")
  {tr_pf, pf_metrics} = Imitation.train_step(trainer, first_batch, loss_fn)
  Output.puts("preflight: train step OK (loss=#{inspect(Nx.to_number(pf_metrics.loss))})")

  tr_pf =
    if probe_reg > 0 do
      {tr2, rstats} =
        ProbeRegularizer.refit(tr_pf, dataset, probe_frame_labels, window_size: window)

      Output.puts("preflight: probe-reg refit OK #{inspect(rstats)}")
      tr2
    else
      tr_pf
    end

  if probe_eval_every > 0 do
    bas =
      ExPhil.Interp.TrainingProbes.eval(tr_pf, probe_eval_trunk_fn, dataset, probe_eval_labels,
        window_size: window
      )

    Output.puts("preflight: probe-eval OK #{ExPhil.Interp.TrainingProbes.format(bas)}")
  end

  pf_bin = out_path <> ".preflight.bin"
  pf_ckpt = out_path <> ".preflight.ckpt"

  case Imitation.export_policy(tr_pf, pf_bin) do
    :ok ->
      Output.puts("preflight: policy export OK")

    {:error, reason} ->
      Output.error("preflight: policy export FAILED: #{inspect(reason)}")
      System.halt(1)
  end

  Imitation.save_checkpoint(tr_pf, pf_ckpt, meta: %{epoch: 0, fingerprint: :preflight})
  Output.puts("preflight: trainer snapshot OK")
  File.rm(pf_bin)
  File.rm(pf_ckpt)

  pf_mem = record_ledger.("preflight")
  pf_secs = div(System.monotonic_time(:millisecond) - preflight_t0, 1000)

  Output.success(
    "PREFLIGHT PASSED in #{pf_secs}s — peak rss " <>
      "#{MemoryLedger.format_bytes(pf_mem.hwm_bytes)} over #{pool_frames} frames; " <>
      "ledger updated (#{MemoryLedger.default_path()})"
  )

  System.halt(0)
end

# New recipe flags join the fingerprint only when set (TAGGED, so two
# different flags with equal values can't collide) — snapshots written
# before a flag existed (e.g. tonight's mamba_full) still resume.
run_fingerprint =
  {opts[:rollouts], opts[:expert], backbone, opts[:prev_action_dropout],
   opts[:transition_weight], max_epochs}
  |> then(fn base ->
    if cw = opts[:conversion_weight],
      do: :erlang.append_element(base, {:conversion_weight, cw}),
      else: base
  end)
  |> then(fn base ->
    if bc_paths != [],
      do: :erlang.append_element(base, {:bc, length(bc_paths), length(bc_frames)}),
      else: base
  end)
  |> then(fn base ->
    if probe_reg > 0,
      do: :erlang.append_element(base, {:probe_reg, probe_reg, probe_reg_every}),
      else: base
  end)
  |> :erlang.phash2()

{trainer, start_epoch} =
  cond do
    opts[:resume] && File.exists?(trainer_ckpt) ->
      {:ok, raw} = ExPhil.Training.Checkpoint.load(trainer_ckpt, warn_on_mismatch: false)
      meta = Map.get(raw, :meta) || %{}

      if meta[:fingerprint] != run_fingerprint do
        Output.error(
          "--resume refused: #{trainer_ckpt} was written by a DIFFERENT run " <>
            "(fingerprint #{inspect(meta[:fingerprint])} vs #{run_fingerprint}: " <>
            "pool/expert/backbone/dropout/transition/epochs changed). " <>
            "Delete the snapshot to retrain fresh."
        )

        System.halt(4)
      end

      {:ok, restored} = Imitation.load_checkpoint(trainer, trainer_ckpt)
      Output.success("Resuming from trainer snapshot: epoch #{meta[:epoch]}, step #{restored.step}")
      {restored, meta[:epoch] + 1}

    opts[:resume] ->
      Output.warning("--resume set but no snapshot at #{trainer_ckpt} — training fresh")
      {trainer, 1}

    true ->
      {trainer, 1}
  end

# --nan-forensics: per-batch loss finiteness checks plus a trail of numeric
# vitals (param max/norm, adam mu/nu extremes, nu zero-fraction) sampled
# every 100 optimizer steps. On the first non-finite loss: dump the trail,
# halt(2). Discriminates the two live NaN mechanisms:
#   optimizer-spike (pure-bf16 stale second moment) -> param_max modest,
#     nu_zero_frac high/stale, then a single-step blowup;
#   forward overflow -> param_max grinding upward over thousands of steps.
nan_forensics = opts[:nan_forensics] || false

# Float tensors only: ModelState.data also carries integer RNG-key tensors
# (observed u32 ~2.46e9) that poison max/norm stats and wrap on squaring.
flatten_tensors = fn data ->
  walk = fn walk, v ->
    cond do
      is_struct(v, Nx.Tensor) ->
        case Nx.type(v) do
          {:f, _} -> [v]
          {:bf, _} -> [v]
          _ -> []
        end

      is_struct(v) -> []
      is_map(v) -> v |> Map.values() |> Enum.flat_map(&walk.(walk, &1))
      is_tuple(v) -> v |> Tuple.to_list() |> Enum.flat_map(&walk.(walk, &1))
      true -> []
    end
  end

  walk.(walk, data)
end

# Adam(W) state lives somewhere inside the composed optimizer state tuple —
# find the first map holding :mu and :nu rather than hardcoding the nesting.
find_adam_state = fn state ->
  find = fn find, v ->
    cond do
      is_map(v) and not is_struct(v) and Map.has_key?(v, :mu) and Map.has_key?(v, :nu) -> v
      is_tuple(v) -> v |> Tuple.to_list() |> Enum.find_value(&find.(find, &1))
      is_map(v) and not is_struct(v) -> v |> Map.values() |> Enum.find_value(&find.(find, &1))
      true -> nil
    end
  end

  find.(find, state)
end

safe_max = fn nums ->
  if nums != [] and Enum.all?(nums, &is_number/1), do: Enum.max(nums), else: :nan
end

numeric_stats = fn tr_now ->
  params = flatten_tensors.(tr_now.policy_params.data)
  param_maxes = Enum.map(params, fn t -> Nx.abs(t) |> Nx.reduce_max() |> Nx.to_number() end)
  param_sq = Enum.map(params, fn t -> Nx.multiply(t, t) |> Nx.sum() |> Nx.to_number() end)

  {mu_max, nu_max, nu_zero_frac} =
    case find_adam_state.(tr_now.optimizer_state) do
      %{mu: mu, nu: nu} ->
        mus = flatten_tensors.(mu)
        nus = flatten_tensors.(nu)
        nu_total = nus |> Enum.map(&Nx.size/1) |> Enum.sum()

        nu_zeros =
          nus
          |> Enum.map(fn t -> Nx.equal(t, 0.0) |> Nx.sum() |> Nx.to_number() end)
          |> Enum.sum()

        {safe_max.(Enum.map(mus, fn t -> Nx.abs(t) |> Nx.reduce_max() |> Nx.to_number() end)),
         safe_max.(Enum.map(nus, fn t -> Nx.reduce_max(t) |> Nx.to_number() end)),
         Float.round(nu_zeros / max(nu_total, 1), 4)}

      _ ->
        {nil, nil, nil}
    end

  %{
    param_max: safe_max.(param_maxes),
    param_norm:
      if(Enum.all?(param_sq, &is_number/1),
        do: Float.round(:math.sqrt(Enum.sum(param_sq)), 3),
        else: :nan
      ),
    mu_max: mu_max,
    nu_max: nu_max,
    nu_zero_frac: nu_zero_frac
  }
end

# Track the best epoch so a late divergence still exports a usable policy.
# On NaN, restore the best params and CONTINUE (up to 5 restores) instead of
# halting: every LR >= 1.5e-4 run NaN'd mid-run (5-for-5 at 2e-4, 2026-07-13)
# and halting forfeits the rest of the epoch budget. The shuffle seed varies
# by epoch so a restored retry sees a different batch order rather than
# deterministically re-diverging (grad clipping 0.5 is already on; these
# NaNs are forward-pass overflows, so a fresh trajectory is the only out).
{best, final_loss, epochs_used} =
  Enum.reduce_while(start_epoch..max_epochs, {trainer, nil, 0, [], nil, 0}, fn epoch,
                                                                     {tr, _, _, history, best,
                                                                      restores} ->
    {tr, epoch_loss} =
      dataset
      |> Data.batched_sequences(
        batch_size: 64,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: true,
        drop_last: false,
        seed: 42 + epoch,
        transition_weight: opts[:transition_weight],
        sampling_weights: sampling_weights
      )
      |> then(fn batches ->
        if nan_forensics do
          batches
          |> Enum.reduce_while({tr, nil, []}, fn batch, {tr_acc, _, trail} ->
            {tr_next, metrics} = Imitation.train_step(tr_acc, batch, loss_fn)
            loss_num = Nx.to_number(metrics.loss)
            step = tr_next.step

            trail =
              if rem(step, 100) == 0 do
                stats = numeric_stats.(tr_next)
                # The ~150 eager EXLA ops per sample leave device buffers
                # behind until BEAM GC frees the refs — without this the GPU
                # OOMs by step ~3000 (observed 2026-07-13).
                :erlang.garbage_collect()
                Enum.take([{step, loss_num, stats} | trail], 30)
              else
                trail
              end

            # Cliff checkpoints: full trainer state (params + optimizer)
            # every 25k steps once inside the failure zone, keep last 2 —
            # later experiments resume from the cliff instead of re-paying
            # the 40-90 min approach ("build the time-savers first").
            if rem(step, 25_000) == 0 and step >= 250_000 do
              File.mkdir_p!("checkpoints/cliffs")
              path = "checkpoints/cliffs/cliff_step#{step}.ckpt"
              Output.puts("saving cliff checkpoint: #{path}")
              Imitation.save_checkpoint(tr_next, path)

              "checkpoints/cliffs/cliff_step*.ckpt"
              |> Path.wildcard()
              |> Enum.sort_by(fn p ->
                Regex.run(~r/step(\d+)/, p) |> List.last() |> String.to_integer()
              end)
              |> Enum.drop(-2)
              |> Enum.each(&File.rm/1)
            end

            if rem(step, 1000) == 0 and trail != [] do
              {_, _, st} = hd(trail)
              Output.puts("forensics step #{step}: loss=#{inspect(loss_num)} #{inspect(st)}")
            end

            if is_number(loss_num) do
              {:cont, {tr_next, metrics.loss, trail}}
            else
              Output.error(
                "FORENSICS: first non-finite loss (#{inspect(loss_num)}) " <>
                  "at optimizer step #{step}, epoch #{epoch}"
              )

              Output.puts("post-mortem (already-poisoned) stats: #{inspect(numeric_stats.(tr_next))}")
              Output.puts("trail (newest first, sampled every 100 steps):")

              Enum.each(trail, fn {s, l, st} ->
                Output.puts("  step #{s}: loss=#{inspect(l)} #{inspect(st)}")
              end)

              System.halt(2)
            end
          end)
          |> then(fn {tr2, l, _} -> {tr2, l} end)
        else
          Enum.reduce(batches, {tr, nil}, fn batch, {tr_acc, _} ->
            {tr_next, metrics} = Imitation.train_step(tr_acc, batch, loss_fn)
            {tr_next, metrics.loss}
          end)
        end
      end)

    loss = Nx.to_number(epoch_loss)

    # Per-epoch heartbeat (2026-07-22): previously only rem(epoch,5)==0
    # logged, so the drill went SILENT for 4 of every 5 epochs. With
    # ~10-min mamba_2 epochs that's ~40 min of no log output during
    # healthy training — long enough that a log-mtime heartbeat monitor
    # false-alarms (observed while watching the r16 relaunch). One line
    # every epoch keeps mtime advancing, so "pid alive AND log advancing"
    # is a sound liveness signal again without needing to poll GPU util.
    rss = MemoryLedger.process_memory().rss_bytes

    Output.puts(
      "epoch #{epoch}/#{max_epochs}: loss=#{inspect(loss)} rss=#{MemoryLedger.format_bytes(rss)}"
    )

    # Probe-as-regularizer refit (r15): re-derive the shield-lock direction
    # from the network's CURRENT activations. Skipped on NaN epochs — the
    # restore path below rewinds params, and a direction fit on poisoned
    # activations would ride along.
    tr =
      if probe_reg > 0 and is_number(loss) and rem(epoch, probe_reg_every) == 0 do
        {tr2, rstats} = ProbeRegularizer.refit(tr, dataset, probe_frame_labels, window_size: window)
        Output.puts("probe-reg refit @ epoch #{epoch}: #{inspect(rstats)}")
        tr2
      else
        tr
      end

    # Mid-training publish: a play-able snapshot every 10 epochs, so the
    # run can be eye-tested while it trains (from a separate worktree —
    # NO-MIX). Write-then-rename is atomic on the same filesystem, so a
    # reader can never observe a torn file. Warm-restart caveat: mid-cycle
    # snapshots can look worse than the final export.
    #
    # Snapshot runs BEFORE probe-eval (r16 lesson, 2026-07-22): both fire
    # on the same epochs, and when probe-eval crashed at epoch 10 the run
    # died before its first save — 10 epochs of work gone. Risky
    # instrumentation goes last so a death there costs 0 epochs.
    if rem(epoch, 10) == 0 and is_number(loss) do
      latest = Path.rootname(out_path) <> "_latest.bin"
      tmp = latest <> ".tmp"

      case Imitation.export_policy(tr, tmp) do
        :ok -> File.rename!(tmp, latest)
        _ -> File.rm(tmp)
      end

      # Trainer snapshot for --resume (full state incl. Adam moments;
      # atomic inside save_checkpoint). ~11MB sync write every 10 epochs
      # — negligible next to a ~110s epoch. Deleted on successful export.
      Imitation.save_checkpoint(tr, trainer_ckpt,
        meta: %{epoch: epoch, fingerprint: run_fingerprint}
      )
    end

    # Feature-formation curves (--probe-eval-every)
    if probe_eval_every > 0 and is_number(loss) and rem(epoch, probe_eval_every) == 0 do
      bas = ExPhil.Interp.TrainingProbes.eval(tr, probe_eval_trunk_fn, dataset, probe_eval_labels, window_size: window)
      Output.puts("probe-eval @ epoch #{epoch}: #{ExPhil.Interp.TrainingProbes.format(bas)}")

      File.write!(
        probe_curves_path,
        Jason.encode!(%{epoch: epoch, loss: loss, probes: bas}) <> "\n",
        [:append]
      )
    end

    best =
      case best do
        nil -> if is_number(loss), do: {tr, loss}, else: nil
        {_, best_loss} -> if is_number(loss) and loss < best_loss, do: {tr, loss}, else: best
      end

    history = if is_number(loss), do: Enum.take([loss | history], 100), else: history

    plateaued? =
      length(history) == 100 and is_number(loss) and
        List.last(history) - Enum.min(history) < 1.0e-3

    cond do
      not is_number(loss) and best != nil and restores < 5 ->
        {best_tr, best_loss} = best

        Output.warning(
          "NaN at epoch #{epoch} — restored best params " <>
            "(loss=#{Float.round(best_loss * 1.0, 5)}), continuing " <>
            "(restore #{restores + 1}/5)"
        )

        {:cont, {best_tr, best_loss, epoch, history, best, restores + 1}}

      not is_number(loss) -> {:halt, {tr, loss, epoch, history, best, restores}}
      loss < memorized_loss -> {:halt, {tr, loss, epoch, history, best, restores}}
      plateaued? -> {:halt, {tr, loss, epoch, history, best, restores}}
      true -> {:cont, {tr, loss, epoch, history, best, restores}}
    end
  end)
  |> then(fn {_tr, loss, epoch, _, best, _} -> {best, loss, epoch} end)

case {is_number(final_loss), best} do
  {_, nil} ->
    Output.error("Training diverged with no usable epoch — no policy exported")
    System.halt(1)

  {false, {_best_tr, best_loss}} ->
    Output.warning(
      "Training diverged (#{inspect(final_loss)}) at epoch #{epochs_used} — " <>
        "exporting best epoch instead (loss=#{Float.round(best_loss * 1.0, 5)})"
    )

  {true, {_best_tr, best_loss}} ->
    Output.puts("Converged: loss=#{Float.round(best_loss * 1.0, 5)} after #{epochs_used} epochs")
end

{best_trainer, _best_loss} = best

case Imitation.export_policy(best_trainer, out_path) do
  :ok ->
    Output.success("Policy exported: #{out_path}")

    final_mem = record_ledger.("train")

    Output.puts(
      "memory: run peak rss #{MemoryLedger.format_bytes(final_mem.hwm_bytes)} " <>
        "over #{pool_frames} frames — ledger updated (#{MemoryLedger.default_path()})"
    )

    # The run finished — the resume snapshot is a cliff we no longer need
    File.rm(trainer_ckpt)

  {:error, reason} ->
    Output.error("Export failed: #{inspect(reason)}")
    System.halt(1)
end
