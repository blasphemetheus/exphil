defmodule ExPhil.Training.OverfitReplicationTemporalTest do
  @moduledoc """
  Temporal variant of the overfit-replication gate (Hunt #2 of the end-to-end
  failure sweep; see overfit_replication_test.exs for the single-frame gate).

  The Mewtwo ship run uses a temporal backbone, but the single-frame gate only
  certifies the non-temporal pipeline. The temporal path has its own classic
  silent failure: an off-by-one between the state window and the target frame
  trains a model to predict the PAST (or future) — invisible to val_loss,
  fatal live. This gate memorizes the multishine through the REAL temporal
  seams (precompute_frame_embeddings → lazy sequence slicing →
  batched_sequences targets) and decodes with the agent-style padded window.

  Run with: mix test test/exphil/training/overfit_replication_temporal_test.exs --include slow
  """
  use ExUnit.Case, async: false

  @moduletag :slow
  @moduletag timeout: 600_000

  alias ExPhil.Test.{ReplayFixtures, ReplicationCheck}
  alias ExPhil.Training.{Data, Imitation}
  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings

  @frames 128
  @period 8
  @window 16
  @epochs 500
  @batch_size 16

  test "a memorized multishine is replicated through the temporal pipeline" do
    raw = ReplayFixtures.tech_fixture(:multishine, frames: @frames, period: @period)

    frames =
      for {gs, cs} <- raw do
        swapped = %{gs | players: %{1 => gs.players[2], 2 => gs.players[1]}}
        %{game_state: swapped, controller: cs}
      end

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
        hidden_size: 64,
        num_layers: 1,
        learning_rate: 5.0e-4,
        max_grad_norm: 0.5,
        label_smoothing: 0.0,
        dropout: 0.0
      )

    {predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

    memorized_loss = 5.0e-3

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
      flunk(
        "Temporal training diverged (loss=#{inspect(final_loss)}) — not a replication verdict."
      )
    end

    # Decode exactly like Agent.compute_temporal_action: sliding window over
    # per-frame embeddings, padded by repeating the FIRST frame until full.
    embedded_frames =
      Enum.map(frames, fn %{game_state: gs} ->
        Embeddings.Game.embed(gs, nil, 1, config: dataset.embed_config)
      end)

    emitted =
      for i <- 0..(@frames - 1) do
        window =
          if i + 1 >= @window do
            Enum.slice(embedded_frames, i - @window + 1, @window)
          else
            pad = List.duplicate(hd(embedded_frames), @window - (i + 1))
            pad ++ Enum.slice(embedded_frames, 0, i + 1)
          end

        batch = window |> Nx.stack() |> Nx.reshape({1, @window, Nx.size(hd(window))})

        trainer.policy_params
        |> Policy.sample(predict_fn, batch, deterministic: true)
        |> Policy.to_controller_state(axis_buckets: 16)
      end

    expected = Enum.map(raw, fn {_gs, cs} -> cs end)

    # Early frames sit on padded (out-of-distribution) windows — judge only
    # the steady-state region, like a real game after warmup.
    steady = @window - 1
    expected_steady = Enum.drop(expected, steady)
    emitted_steady = Enum.drop(emitted, steady)

    case ReplicationCheck.check(expected_steady, emitted_steady, strictness: :periodic) do
      {:ok, diagnostics} ->
        exact =
          case ReplicationCheck.check(expected_steady, emitted_steady, strictness: :exact) do
            {:ok, _} -> "exact: PASS"
            {:error, _} -> "exact: no"
          end

        IO.puts(
          "\n[overfit-temporal] REPLICATED — loss=#{Float.round(final_loss * 1.0, 4)} " <>
            "#{inspect(Map.take(diagnostics, [:expected_shines, :actual_shines]))} #{exact}"
        )

      {:error, diagnostics} ->
        flunk("""
        Memorized multishine NOT replicated through the temporal pipeline.
        Final training loss: #{Float.round(final_loss * 1.0, 4)}
        Diagnostics: #{inspect(diagnostics, pretty: true, limit: 20)}
        Prime suspect: window/target alignment in sequence building — if the
        target is offset from the window's last frame, the model learned to
        predict the past/future and emitted shines will be phase-shifted by
        the offset (compare expected vs actual shine positions above).
        """)
    end
  end
end
