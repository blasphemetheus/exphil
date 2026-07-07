defmodule ExPhil.Training.OverfitReplicationSlpTest do
  @moduledoc """
  Tier-2 overfit-replication gate (Hunt #3): memorize-and-replicate on the
  REAL recorded replay `test/fixtures/replays/fox_multishine.slp`, ingested
  through the Peppi parse path.

  What this adds over the synthetic gate: parser port mapping, the parsed
  stick range convention (flat main_stick_x/y, 0..1) flowing into
  controller_to_action discretization, and button field mapping.

  Design notes learned the hard way:
  - Countdown frames (game frame < 0) are sliced off: the game state is
    FROZEN while the scripted inputs kept cycling — identical states with
    conflicting targets are irreducible for any model.
  - This gate is TEMPORAL (GRU, 16-frame windows). A real replay's single
    frames are not guaranteed phase-distinguishable (unlike the synthetic
    fixture, which bakes phase into the state by construction), so the
    single-frame version of this task is ill-posed: loss floors around 0.04
    and replication fails for representational reasons, not pipeline bugs.
  - The pass bar is "≥40% of expected shines matched positionally (±1f)",
    NOT full :periodic. The recorder is OPEN-LOOP (inputs keyed to wall
    clock, not game state), so during whiffed cycles Fox stands identically
    still while the script's phase advances — state-identical windows with
    different targets, irreducible even at memorization loss (observed:
    loss 0.0018 with ~28% shine dropout clustered in idle segments). A
    parse-side convention bug (port swap, stick range, button mapping)
    produces ~0% positional matches; ambiguity produces partial dropout —
    40% separates them by a wide margin (observed healthy runs: 55-63%, convention bugs: ~0%). A future CLOSED-LOOP recorder
    (react to Fox's action state) would make :exact achievable here.

  The recording: scripted 12-frame shine cycle on port 1 (Fox), human G&W on
  port 2 attacking (hitstun adds state variety while inputs stay periodic).
  See scripts/record_multishine.exs.

  Run with: mix test test/exphil/training/overfit_replication_slp_test.exs --include slow
  """
  use ExUnit.Case, async: false

  @moduletag :slow
  @moduletag timeout: 600_000

  alias ExPhil.Test.ReplicationCheck
  alias ExPhil.Training.{Data, Imitation}
  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings
  alias ExPhil.Data.Peppi

  @fixture "test/fixtures/replays/fox_multishine.slp"
  # Footage region: shine loop after the countdown; excludes the deliberate
  # SD phase at the end (hold-left until game end, only there to finalize
  # the file).
  @footage_frames 800
  @window 16
  @epochs 2000
  @batch_size 64

  test "a memorized multishine from a real .slp is replicated (Peppi + temporal)" do
    if not File.exists?(@fixture) do
      # Tier-2 fixture is optional per the harness design — regenerate with
      # scripts/record_multishine.exs if absent.
      IO.puts("[overfit-slp] fixture missing, skipping")
      assert true
    else
      {:ok, replay} = Peppi.parse(@fixture)

      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.take(@footage_frames)

      assert length(frames) == @footage_frames

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
          window_size: @window,
          hidden_size: 256,
          num_layers: 1,
          learning_rate: 5.0e-4,
          max_grad_norm: 0.5,
          label_smoothing: 0.0,
          dropout: 0.0
        )

      {predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      # Real-game data is messier than the synthetic fixture; this bar still
      # requires near-total memorization.
      memorized_loss = 2.0e-3

      {trainer, final_loss} =
        Enum.reduce_while(1..@epochs, {trainer, nil}, fn _epoch, {tr, _} ->
          {tr, epoch_loss} =
            dataset
            |> Data.batched_sequences(
              batch_size: @batch_size,
              window_size: @window,
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

          cond do
            not is_number(loss) -> {:halt, {tr, loss}}
            loss < memorized_loss -> {:halt, {tr, loss}}
            true -> {:cont, {tr, loss}}
          end
        end)

      if not is_number(final_loss) do
        flunk("Training diverged (loss=#{inspect(final_loss)}) — not a replication verdict.")
      end

      embedded_frames =
        Enum.map(frames, fn %{game_state: gs} ->
          Embeddings.Game.embed(gs, nil, 1, config: dataset.embed_config)
        end)

      emitted =
        for i <- 0..(@footage_frames - 1) do
          window =
            if i + 1 >= @window do
              Enum.slice(embedded_frames, i - @window + 1, @window)
            else
              pad = List.duplicate(hd(embedded_frames), @window - (i + 1))
              pad ++ Enum.slice(embedded_frames, 0, i + 1)
            end

          batch = window |> Nx.stack() |> Nx.reshape({1, @window, embed_size})

          trainer.policy_params
          |> Policy.sample(predict_fn, batch, deterministic: true)
          |> Policy.to_controller_state(axis_buckets: 16)
        end

      expected = Enum.map(frames, & &1.controller)

      steady = @window - 1
      expected_steady = Enum.drop(expected, steady)
      emitted_steady = Enum.drop(emitted, steady)

      expected_shines = ReplicationCheck.shine_indices(expected_steady)
      actual_shines = ReplicationCheck.shine_indices(emitted_steady)

      matched =
        Enum.count(expected_shines, fn e ->
          Enum.any?(actual_shines, fn a -> abs(a - e) <= 1 end)
        end)

      match_fraction = matched / max(length(expected_shines), 1)

      IO.puts(
        "\n[overfit-slp] loss=#{Float.round(final_loss * 1.0, 4)} " <>
          "shines matched: #{matched}/#{length(expected_shines)} " <>
          "(#{Float.round(match_fraction * 100, 1)}%), emitted #{length(actual_shines)}"
      )

      if match_fraction < 0.4 do
        flunk("""
        Real-replay multishine positional match #{Float.round(match_fraction * 100, 1)}% < 40% —
        this is convention-bug territory, not open-loop ambiguity.
        Final training loss: #{Float.round(final_loss * 1.0, 4)}
        Expected shine positions: #{inspect(Enum.take(expected_shines, 15))}
        Actual shine positions:   #{inspect(Enum.take(actual_shines, 15))}
        Suspects unique to this path: build_controller_state field mapping,
        parsed stick 0..1 convention vs controller_to_action discretization,
        port selection in to_training_frames.
        """)
      end
    end
  end
end
