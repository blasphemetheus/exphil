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
    strict: [replays: :string, out: :string, robust: :boolean]
  )

# --replays accepts a dir or glob of .slp files; default = the single
# canonical fixture (pipeline-gate mode). --robust enables regularization
# for the generalization experiment (task #20) instead of pure memorization.
replay_glob = opts[:replays] || "test/fixtures/replays/fox_multishine_closed.slp"
out_path = opts[:out] || "checkpoints/multishine_probe_policy.bin"
robust = opts[:robust] || false
window = 16

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
    # game) — filter by that exact input signature rather than a fixed count
    |> Enum.reject(fn %{controller: c} ->
      c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
        not c.button_b and not c.button_x
    end)
  end)

Output.puts("Training frames: #{length(frames)}")

dataset =
  frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings()

embed_size = Embeddings.embedding_size(dataset.embed_config)

trainer =
  Imitation.new(
    embed_config: dataset.embed_config,
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
    dropout: if(robust, do: 0.1, else: 0.0)
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

# Robust mode can't use an absolute loss bar: label smoothing imposes a
# FLOOR (≈0.33/categorical head at ε=0.05 → ~1.62 total) that no model can
# go below. Stop on plateau instead: <1e-3 improvement over 100 epochs.
memorized_loss = if robust, do: :plateau, else: 2.0e-3
max_epochs = if robust, do: 800, else: 2000

{trainer, final_loss, epochs_used} =
  Enum.reduce_while(1..max_epochs, {trainer, nil, 0, []}, fn epoch, {tr, _, _, history} ->
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

    history = Enum.take([loss | history], 100)

    plateaued? =
      length(history) == 100 and is_number(loss) and
        List.last(history) - Enum.min(history) < 1.0e-3

    cond do
      not is_number(loss) -> {:halt, {tr, loss, epoch, history}}
      memorized_loss == :plateau and plateaued? -> {:halt, {tr, loss, epoch, history}}
      is_number(memorized_loss) and loss < memorized_loss -> {:halt, {tr, loss, epoch, history}}
      true -> {:cont, {tr, loss, epoch, history}}
    end
  end)
  |> then(fn {tr, loss, epoch, _} -> {tr, loss, epoch} end)

if not is_number(final_loss) do
  Output.error("Training diverged (#{inspect(final_loss)}) — no policy exported")
  System.halt(1)
end

Output.puts("Memorized: loss=#{Float.round(final_loss * 1.0, 5)} after #{epochs_used} epochs")

case Imitation.export_policy(trainer, out_path) do
  :ok -> Output.success("Policy exported: #{out_path}")
  {:error, reason} ->
    Output.error("Export failed: #{inspect(reason)}")
    System.halt(1)
end
