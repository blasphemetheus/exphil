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

fixture = "test/fixtures/replays/fox_multishine_closed.slp"
out_path = "checkpoints/multishine_probe_policy.bin"
window = 16

Output.banner("Multishine Probe Policy Trainer")

{:ok, replay} = Peppi.parse(fixture)

frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(800)

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
    label_smoothing: 0.0,
    dropout: 0.0
  )

{_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

memorized_loss = 2.0e-3
max_epochs = 2000

{trainer, final_loss, epochs_used} =
  Enum.reduce_while(1..max_epochs, {trainer, nil, 0}, fn epoch, {tr, _, _} ->
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

    cond do
      not is_number(loss) -> {:halt, {tr, loss, epoch}}
      loss < memorized_loss -> {:halt, {tr, loss, epoch}}
      true -> {:cont, {tr, loss, epoch}}
    end
  end)

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
