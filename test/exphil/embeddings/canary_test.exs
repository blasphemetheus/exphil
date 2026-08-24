defmodule ExPhil.Embeddings.CanaryTest do
  # Train-vs-live embedding fingerprint (GUARDS_BACKLOG #1).
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Canary
  alias ExPhil.Embeddings.Game.Config

  defp config(overrides \\ []) do
    Enum.reduce(overrides, Config.default(), fn {k, v}, acc -> Map.put(acc, k, v) end)
  end

  test "batched and live fingerprints agree for the default config" do
    assert Canary.compare(
             Canary.fingerprint_batched(config()),
             Canary.fingerprint_live(config())
           ) == :ok
  end

  test "agreement holds for the custom-layout configs (queue/delay/stage)" do
    for overrides <- [
          [queue_depth: 4, with_delay_id: true],
          [stage_internals: true],
          [queue_depth: 4, with_delay_id: true, stage_internals: true]
        ] do
      cfg = config(overrides)

      assert Canary.compare(
               Canary.fingerprint_batched(cfg),
               Canary.fingerprint_live(cfg)
             ) == :ok,
             "paths diverge for #{inspect(overrides)}"
    end
  end

  test "a config divergence is DETECTED (the guard actually guards)" do
    stored = Canary.fingerprint_batched(config())
    live = Canary.fingerprint_live(config(stage_internals: true))

    assert {:error, {:size_mismatch, _, _}} = Canary.compare(stored, live)
  end

  test "a value-level divergence names the dims" do
    stored = Canary.fingerprint_batched(config())
    [h | t] = stored
    corrupted = [h + 1.0 | t]

    assert {:error, {:divergent_dims, 1, [{0, _, _}]}} = Canary.compare(stored, corrupted)
  end

  test "the canary state exercises the stage-internals family" do
    # FoD with heights set: flag flips the embedding (regression pin
    # for the exact 08-24 bug class the canary exists to catch).
    on = Canary.fingerprint_live(config(stage_internals: true))
    off = Canary.fingerprint_live(config())
    assert length(on) != length(off)
  end
end
