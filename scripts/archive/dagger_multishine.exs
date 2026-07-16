# DAgger loop for the multishine probe: aggregate the canonical fixture with
# live rollout replays relabeled by the scripted expert, and retrain.
#
# The insight that makes this cheap: Slippi replays of live play sessions ARE
# on-policy DAgger rollouts — they record every state the policy actually
# visited (including off-distribution freezes the fixture never covers). So an
# iteration is just: play (play_dolphin_async.exs) → grab the replay from
# ~/Slippi → run this script → play the new policy.
#
#   mix run scripts/dagger_multishine.exs \
#     --rollouts "~/Slippi/Game_20260708T083538.slp,~/Slippi/Game_20260708T081923.slp" \
#     --out checkpoints/multishine_dagger1_policy.bin
#
# Relabeling: every rollout frame's controller is REPLACED by the expert's
# correction (MultishineExpert — fixture table + recovery rules); dead/respawn
# frames are dropped. Note the prev-action channel is derived from the same
# (relabeled, delay-shifted) controller field, so it shows the expert's prev
# input rather than what the policy actually pressed — prev-action dropout
# (default 0.1 here) keeps the model from leaning on that too hard.
#
# Defaults are purpose-built for the delay-2 era probes: --prev-action on,
# --action-delay 2. Pass --no-prev-action / --action-delay N to override.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.{Data, Imitation, Output}
alias ExPhil.Data.Peppi
alias ExPhil.Agents.MultishineExpert
alias ExPhil.Embeddings

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      rollouts: :string,
      fixture: :string,
      out: :string,
      prev_action: :boolean,
      action_delay: :integer,
      prev_action_dropout: :float,
      port: :integer,
      lr: :float
    ]
  )

fixture_path = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
out_path = opts[:out] || "checkpoints/multishine_dagger_policy.bin"
prev_action = Keyword.get(opts, :prev_action, true)
prev_action_dropout = opts[:prev_action_dropout] || 0.1
action_delay = Keyword.get(opts, :action_delay, 2)
port = opts[:port] || 1
# 2e-4 (not the probe trainer's 5e-4): the aggregate grows each DAgger
# iteration, and more optimizer steps per epoch at high LR eventually
# blew up (loss 0.0084 at epoch 150 → NaN at 178 on a 73k-frame pool)
learning_rate = opts[:lr] || 2.0e-4
window = 16

rollout_paths =
  (opts[:rollouts] || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

if rollout_paths == [] do
  Output.error("No rollout replays given — pass --rollouts \"path1.slp,glob*.slp\"")
  System.halt(1)
end

Output.banner("Multishine DAgger Trainer")

Output.config([
  {"Fixture", fixture_path},
  {"Rollouts", length(rollout_paths)},
  {"Prev-action", prev_action},
  {"Prev-action dropout", prev_action_dropout},
  {"Action delay", action_delay},
  {"Out", out_path}
])

expert = MultishineExpert.from_fixture(fixture_path, player_port: port)

load_frames = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: port, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
end

# Fixture: human = expert; recorded controllers are the labels (SD tail
# filtered by its input signature, same as train_multishine_policy.exs)
fixture_frames =
  load_frames.(fixture_path)
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
      not c.button_b and not c.button_x
  end)

Output.puts("Fixture frames: #{length(fixture_frames)}")

# Rollouts: keep the visited states, replace the (policy's own) controllers
# with expert corrections; drop dead/respawn frames. The frame-number gaps
# left by dropped frames are respected by shift_actions' contiguity check.
#
# :prev_controller = what the policy ACTUALLY pressed landing at decision
# time (frame t+delay-1), NOT the expert's correction — the live agent's
# prev-action channel shows its own (often wrong) last output, and training
# must cover exactly those (state, own-mistake) → correction pairs. Deriving
# prev from relabeled neighbors taught "(reflector, prev=X) → X" while live
# the model saw (reflector, prev=B) and fell back to B (the DAgger-2 B-spam).
relabel = fn frames, recorded ->
  Enum.flat_map(frames, fn frame ->
    # The input landing at decision time — fed to the expert (so recovery
    # taps alternate against what was really pressed) AND to the prev-action
    # channel (so the model sees the same signal the label was keyed on)
    prev = recorded[frame.game_state.frame + action_delay - 1]

    case MultishineExpert.label(expert, frame.game_state.players[port], prev) do
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

button_sig = fn c -> {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r} end

rollout_frame_lists =
  Enum.map(rollout_paths, fn path ->
    raw = load_frames.(path)

    # Actual per-frame inputs, kept for the prev-action channel and the
    # disagreement diagnostic
    recorded = Map.new(raw, fn f -> {f.game_state.frame, f.controller} end)

    relabeled = relabel.(raw, recorded)

    # Disagreement = fraction of kept frames where the expert's buttons differ
    # from what the policy pressed — a measure of how off-expert the rollout was
    disagreements =
      Enum.count(relabeled, fn f ->
        button_sig.(f.controller) != button_sig.(recorded[f.game_state.frame])
      end)

    pct = Float.round(100.0 * disagreements / max(length(relabeled), 1), 1)
    Output.puts("  #{Path.basename(path)}: #{length(relabeled)} frames, #{pct}% corrected")

    relabeled
  end)

all_frames = List.flatten([fixture_frames | rollout_frame_lists])
Output.puts("Aggregate: #{length(all_frames)} frames (#{length(fixture_frames)} fixture)")

# Shift per source replay — never across the fixture/rollout concat boundary
# (overlapping frame numbers could fool the contiguity check into stitching
# two different games together)
shifted_frames =
  [fixture_frames | rollout_frame_lists]
  |> Enum.flat_map(&Data.shift_actions(&1, action_delay))

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
    learning_rate: learning_rate,
    max_grad_norm: 0.5,
    label_smoothing: 0.0,
    dropout: 0.0
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

# Same convergence bar as the probe trainer, with a plateau stop as fallback
# (aggregated data is more diverse than the pure fixture, so the absolute bar
# may take longer or stall slightly above 2e-3)
memorized_loss = 2.0e-3
max_epochs = 1000

# Track the best epoch's trainer so a late divergence (loss → NaN after
# converging most of the way) still exports a usable policy — essential for
# unattended dagger_loop.sh runs. Holding one extra trainer reference costs
# one extra param set on the GPU.
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
    Output.puts(
      "Converged: loss=#{Float.round(best_loss * 1.0, 5)} after #{epochs_used} epochs"
    )
end

{best_trainer, _best_loss} = best

case Imitation.export_policy(best_trainer, out_path) do
  :ok -> Output.success("Policy exported: #{out_path}")
  {:error, reason} ->
    Output.error("Export failed: #{inspect(reason)}")
    System.halt(1)
end
