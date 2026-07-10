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

alias ExPhil.Training.{Data, Imitation, Output}
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
      backbone: :string
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
         "test/fixtures/replays/mewtwo_approach_fair.slp",
       "checkpoints/mewtwo_fair_dagger_policy.bin", 16}

    "fox_recovery" ->
      # Rules-only (recovery is pure geometry): no fixture; every ordinary
      # replay with offstage moments is a rollout to relabel
      {ExPhil.Agents.FoxRecoveryExpert, nil, "checkpoints/fox_recovery_dagger_policy.bin", 1}

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
window = 16

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
  {:ok, replay} = Peppi.parse(path)
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

# Expert table from ALL fixture recordings combined
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

all_frames = List.flatten(fixture_frame_lists ++ rollout_frame_lists)
Output.puts("Aggregate: #{length(all_frames)} frames (#{length(fixture_frames)} fixture)")

# Shift per source replay — never across concat boundaries
shifted_frames =
  (fixture_frame_lists ++ rollout_frame_lists)
  |> Enum.flat_map(&Data.shift_actions(&1, action_delay))

dataset =
  shifted_frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings(
    use_prev_action: prev_action,
    prev_action_dropout: prev_action_dropout
  )

embed_size = Embeddings.embedding_size(dataset.embed_config)

# Cosine-decay the LR: constant LR diverged to NaN late in run after run
# (multishine at 90k frames, mewtwo at 31k) — more steps per epoch means more
# chances for one step to blow up. Decay over ~200 epochs' worth of steps
# (400 left LR too high too long: a 90k-frame pool still NaN'd at epoch 168);
# healthy runs hit the loss bar or plateau well before the floor.
steps_per_epoch = div(dataset.size, 64) + 1

# --backbone tests whether drill conclusions transfer across architectures
# (drills default to GRU for iteration speed; E-series models are mamba —
# e.g. is the under-confidence that hysteresis exposed GRU-specific?)
backbone = String.to_atom(opts[:backbone] || "gru")

trainer =
  Imitation.new(
    embed_config: dataset.embed_config,
    use_prev_action: prev_action,
    embed_size: embed_size,
    temporal: true,
    backbone: backbone,
    window_size: window,
    hidden_size: 256,
    num_layers: 1,
    learning_rate: learning_rate,
    lr_schedule: :cosine,
    warmup_steps: steps_per_epoch,
    decay_steps: steps_per_epoch * 200,
    max_grad_norm: 0.5,
    label_smoothing: 0.0,
    dropout: 0.0
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

memorized_loss = 2.0e-3
max_epochs = 1000

# Track the best epoch so a late divergence still exports a usable policy
{best, final_loss, epochs_used} =
  Enum.reduce_while(1..max_epochs, {trainer, nil, 0, [], nil}, fn epoch,
                                                                  {tr, _, _, history, best} ->
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

    best =
      case best do
        nil -> if is_number(loss), do: {tr, loss}, else: nil
        {_, best_loss} -> if is_number(loss) and loss < best_loss, do: {tr, loss}, else: best
      end

    history = Enum.take([loss | history], 100)

    plateaued? =
      length(history) == 100 and is_number(loss) and
        List.last(history) - Enum.min(history) < 1.0e-3

    cond do
      not is_number(loss) -> {:halt, {tr, loss, epoch, history, best}}
      loss < memorized_loss -> {:halt, {tr, loss, epoch, history, best}}
      plateaued? -> {:halt, {tr, loss, epoch, history, best}}
      true -> {:cont, {tr, loss, epoch, history, best}}
    end
  end)
  |> then(fn {_tr, loss, epoch, _, best} -> {best, loss, epoch} end)

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
  :ok -> Output.success("Policy exported: #{out_path}")
  {:error, reason} ->
    Output.error("Export failed: #{inspect(reason)}")
    System.halt(1)
end
