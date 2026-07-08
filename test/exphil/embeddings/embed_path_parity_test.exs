defmodule ExPhil.Embeddings.EmbedPathParityTest do
  @moduledoc """
  Train/inference embedding-skew hunt (see the overfit-replication gate).

  Training embeds via the batched `Embeddings.Game.embed_states_fast/3`
  (data.ex); the live agent embeds via per-frame `Embeddings.Game.embed/4`
  (agents/agent.ex `embed_game_state/3`). If these ever drift, the deployed
  model receives different inputs than it was trained on — a silent BC killer
  invisible to val_loss. This test pins them to bitwise-identical outputs
  across diverse game states.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings
  alias ExPhil.Test.ReplayFixtures

  defp diverse_states do
    multishine =
      ReplayFixtures.tech_fixture(:multishine, frames: 16, period: 8)
      |> Enum.map(fn {gs, _cs} -> gs end)

    named = [
      ReplayFixtures.neutral_game_fixture(:mewtwo_vs_fox),
      ReplayFixtures.neutral_game_fixture(:marth_vs_sheik)
    ]

    multishine ++ named
  rescue
    # Not all named fixtures may exist; the multishine sequence alone is
    # already phase-diverse (grounded/airborne, action states, y positions).
    _ ->
      ReplayFixtures.tech_fixture(:multishine, frames: 16, period: 8)
      |> Enum.map(fn {gs, _cs} -> gs end)
  end

  test "embed_states_fast (training) == embed (live agent) for every state and port" do
    config = Embeddings.config()
    states = diverse_states()

    for port <- [1, 2] do
      batched = Embeddings.Game.embed_states_fast(states, port, config: config)

      per_frame =
        states
        |> Enum.map(&Embeddings.Game.embed(&1, nil, port, config: config))
        |> Nx.stack()

      assert Nx.shape(batched) == Nx.shape(per_frame)

      diff =
        batched
        |> Nx.subtract(per_frame)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      # Tolerance: scalar-vs-batched float ops round differently by ~1 ULP
      # (observed 3e-8). Structural/layout bugs produce diffs of ~1.0 — this
      # threshold separates them by 6 orders of magnitude.
      if diff > 1.0e-6 do
        # Locate the first mismatching state + dimension for the report
        mismatch =
          batched
          |> Nx.subtract(per_frame)
          |> Nx.abs()
          |> Nx.greater(1.0e-6)
          |> Nx.reduce_max(axes: [1])
          |> Nx.to_flat_list()
          |> Enum.find_index(&(&1 == 1))

        flunk("""
        TRAIN/INFERENCE EMBEDDING SKEW (port #{port}): max |diff| = #{diff}
        First mismatching state index: #{inspect(mismatch)}
        The model trains on embed_states_fast but plays on embed — these MUST
        be identical or the deployed policy sees inputs it never trained on.
        """)
      end
    end
  end

  test "prev-action channel: batched prev_controllers == per-frame prev_action" do
    # Prev-action conditioning (task #21): training embeds frame i with frame
    # i-1's controller via embed_states_fast(prev_controllers:); the live
    # agent embeds with its own last emitted controller via embed(gs, prev,
    # port). Same parity requirement as the base test — plus the channel must
    # actually CHANGE the embedding (guard against silently ignored opts).
    config = Embeddings.config()

    frames = ReplayFixtures.tech_fixture(:multishine, frames: 16, period: 8)
    states = Enum.map(frames, fn {gs, _cs} -> gs end)
    # Shifted controllers: frame i's prev = frame i-1's controller, nil first
    prevs = [nil | Enum.map(frames, fn {_gs, cs} -> cs end)] |> Enum.take(length(states))

    batched =
      Embeddings.Game.embed_states_fast(states, 1, config: config, prev_controllers: prevs)

    per_frame =
      Enum.zip(states, prevs)
      |> Enum.map(fn {gs, prev} -> Embeddings.Game.embed(gs, prev, 1, config: config) end)
      |> Nx.stack()

    assert Nx.shape(batched) == Nx.shape(per_frame)

    diff =
      batched |> Nx.subtract(per_frame) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff <= 1.0e-6,
           "PREV-ACTION EMBEDDING SKEW: max |diff| = #{diff} — training " <>
             "(prev_controllers) and live (embed prev arg) disagree"

    # The channel must be live: real prev controllers ≠ zeroed slot
    zeroed = Embeddings.Game.embed_states_fast(states, 1, config: config)

    channel_delta =
      batched |> Nx.subtract(zeroed) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert channel_delta > 1.0e-3,
           "prev_controllers appears to be IGNORED (embedding identical to zeros)"
  end
end
