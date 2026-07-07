defmodule ExPhil.Training.OverfitReplicationTest do
  @moduledoc """
  Piece 3 of the overfit-replication harness (docs/planning/HANDOFF.md).

  Vlad Firoiu's pipeline-correctness gate: train a small model to MEMORIZE a
  short scripted behavior (Fox multishine), decode its per-frame predictions
  back to `%ControllerState{}`, and check the behavior is reproduced. A
  memorized behavior a correct pipeline can always replicate — failure here is
  categorically a pipeline bug (embedding / discretization / decode /
  sampling), not underfitting.

  Exercises the real seams end-to-end:
  fixture → Data.from_frames → Data.batched (controller→target discretization)
  → Imitation.train_step → Policy.sample(deterministic) →
  Policy.to_controller_state → ReplicationCheck.

  Tagged :slow — needs a working Nx backend and ~a minute of training.
  Run with: mix test test/exphil/training/overfit_replication_test.exs --include slow
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
  @epochs 500
  @batch_size 32

  test "a memorized multishine is replicated (pipeline correctness gate)" do
    raw = ReplayFixtures.tech_fixture(:multishine, frames: @frames, period: @period)

    # The training embed path takes port 1's perspective (see
    # Data embed_states_fast calls); the fixture puts Fox on port 2 — swap so
    # the acting player is the embedded "self".
    frames =
      for {gs, cs} <- raw do
        swapped = %{gs | players: %{1 => gs.players[2], 2 => gs.players[1]}}
        %{game_state: swapped, controller: cs}
      end

    dataset = Data.from_frames(frames)
    embed_size = Embeddings.embedding_size(dataset.embed_config)

    trainer =
      Imitation.new(
        embed_config: dataset.embed_config,
        embed_size: embed_size,
        hidden_sizes: [128, 128],
        # 1e-3 without clipping diverged to NaN on ~half of param inits
        learning_rate: 5.0e-4,
        max_grad_norm: 0.5,
        # Memorization run: no regularization that fights fitting the data
        label_smoothing: 0.0,
        dropout: 0.0
      )

    {predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

    # Train until memorized (loss < threshold), NOT for a fixed large epoch
    # count: once loss ≈ 0 the logits keep growing toward ±inf (pos_weight
    # never stops pushing) and eventually produce NaN — grinding past
    # memorization is what made this test flaky at fixed 300-500 epochs.
    memorized_loss = 5.0e-3

    {trainer, final_loss} =
      Enum.reduce_while(1..@epochs, {trainer, nil}, fn _epoch, {tr, _loss} ->
        {tr, epoch_loss} =
          dataset
          |> Data.batched(batch_size: @batch_size, shuffle: true, drop_last: false, seed: 42)
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

    # Divergence is a DIFFERENT failure than non-replication: NaN weights say
    # nothing about pipeline correctness. Fail loud and separately.
    # (Nx.to_number returns :nan/:infinity atoms for non-finite floats.)
    if not is_number(final_loss) do
      flunk(
        "Training diverged (loss=#{inspect(final_loss)}) — not a replication verdict. " <>
          "Lower the learning rate / check grad clipping before reading anything into this."
      )
    end

    # Decode: deterministic per-frame predictions back to controller states —
    # the same path the live agent uses (Policy.sample + to_controller_state).
    emitted =
      for %{game_state: gs} <- frames do
        embedded = Embeddings.Game.embed(gs, nil, 1, config: dataset.embed_config)
        batch = Nx.new_axis(embedded, 0)

        trainer.policy_params
        |> Policy.sample(predict_fn, batch, deterministic: true)
        |> Policy.to_controller_state(axis_buckets: 16)
      end

    expected = Enum.map(raw, fn {_gs, cs} -> cs end)

    periodic = ReplicationCheck.check(expected, emitted, strictness: :periodic)
    exact = ReplicationCheck.check(expected, emitted, strictness: :exact)

    case periodic do
      {:ok, diagnostics} ->
        # :exact is aspirational — report but don't gate on it.
        exact_note =
          case exact do
            {:ok, _} -> "exact: PASS"
            {:error, d} -> "exact: no (#{inspect(Map.take(d, [:reason]))})"
          end

        IO.puts(
          "\n[overfit] REPLICATED — loss=#{Float.round(final_loss * 1.0, 4)} " <>
            "#{inspect(Map.take(diagnostics, [:expected_shines, :actual_shines]))} #{exact_note}"
        )

      {:error, diagnostics} ->
        flunk("""
        Memorized multishine NOT replicated — this is a pipeline bug, not underfitting.
        Final training loss: #{Float.round(final_loss * 1.0, 4)}
        Diagnostics: #{inspect(diagnostics, pretty: true, limit: 20)}
        Check, in order: controller→target discretization (Data.batched),
        embedding determinism, Policy.sample argmax decode, to_controller_state
        bucket→analog mapping (GOTCHAS #51/#52 class bugs live here).
        """)
    end
  end
end
