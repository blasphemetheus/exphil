defmodule ExPhil.Harness.AfConventionDefaultTest do
  @moduledoc """
  Pins `af_convention: :parsed` as the default, permanently.

  GOTCHAS #81 SETTLED (2026-07-26): the per-action delta table underlying
  `af_convention: :live` is **invalid**. The dual-port experiment showed the
  live-vs-parsed `action_frame` delta varies *per occurrence* — the same
  action, same run, same port, sometimes 0 and sometimes 1 — so no key
  (action, character, or run) makes the table right. It is a majority vote
  that is wrong for the minority.

  Why this test exists: `test/exphil/data/action_frame_convention_test.exs`
  and `test/exphil/eval/state_stream_diff_test.exs` still pin that table's
  CONTENTS, so they pass whether or not the feature is enabled. A
  well-meaning future change flipping the default to `:live` would pass the
  entire suite while silently corrupting live embeddings for the minority
  of frames. This is the guard that catches it.

  If you are here because this test failed: read GOTCHAS #81's SETTLED
  section before changing it. Re-enabling the feature requires conditioning
  on entry context (not action id), not a default flip.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent

  describe "af_convention default (GOTCHAS #81 SETTLED)" do
    test "an Agent started without the option uses :parsed" do
      {:ok, agent} = Agent.start_link([])

      on_exit(fn -> if Process.alive?(agent), do: GenServer.stop(agent) end)

      assert :sys.get_state(agent).af_convention == :parsed,
             "af_convention must default to :parsed — the :live delta table is INVALID " <>
               "(GOTCHAS #81 SETTLED: the delta varies per OCCURRENCE, so no per-action " <>
               "key can be correct). Re-enabling needs entry-context conditioning, not a " <>
               "default flip."
    end

    test "the --live-af CLI flag still defaults to false" do
      opts = ExPhil.CLI.parse_args([], flags: [:dolphin])

      refute opts[:live_af],
             "--live-af must default to false; it enables the invalid per-action delta table."
    end
  end
end
