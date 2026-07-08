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
      lr: :float
    ]
  )

{expert_mod, default_fixture, default_out} =
  case opts[:expert] || "multishine" do
    "multishine" ->
      {ExPhil.Agents.MultishineExpert, "test/fixtures/replays/fox_multishine_closed.slp",
       "checkpoints/multishine_dagger_policy.bin"}

    "mewtwo_fair" ->
      {ExPhil.Agents.MewtwoFairExpert, "test/fixtures/replays/mewtwo_fair_chains.slp",
       "checkpoints/mewtwo_fair_dagger_policy.bin"}

    other ->
      Output.error("Unknown expert #{inspect(other)} (multishine | mewtwo_fair)")
      System.halt(1)
  end

fixture_path = opts[:fixture] || default_fixture
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
  Output.error("No rollout replays given — pass --rollouts \"path1.slp,glob*.slp\"")
  System.halt(1)
end

Output.banner("Drill DAgger Trainer")

Output.config([
  {"Expert", inspect(expert_mod)},
  {"Fixture", fixture_path},
  {"Rollouts", length(rollout_paths)},
  {"Prev-action", prev_action},
  {"Prev-action dropout", prev_action_dropout},
  {"Action delay", action_delay},
  {"Out", out_path}
])

expert = expert_mod.from_fixture(fixture_path, player_port: port)

load_frames = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: port, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
end

# Fixture: human = expert; recorded controllers are the labels
fixture_frames = load_frames.(fixture_path)
Output.puts("Fixture frames: #{length(fixture_frames)}")

relabel = fn frames, recorded ->
  Enum.flat_map(frames, fn frame ->
    prev = recorded[frame.game_state.frame + action_delay - 1]

    case expert_mod.label(expert, frame.game_state.players[port], prev) do
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
    raw = load_frames.(path)
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

all_frames = List.flatten([fixture_frames | rollout_frame_lists])
Output.puts("Aggregate: #{length(all_frames)} frames (#{length(fixture_frames)} fixture)")

# Shift per source replay — never across concat boundaries
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
