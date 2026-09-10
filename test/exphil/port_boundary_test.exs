defmodule ExPhil.PortBoundaryTest do
  @moduledoc """
  INVARIANTS.md item 5: ports exist only at the parse/bridge boundary.

  Downstream code resolves the subject and the opponent through
  `ExPhil.Bridge.GameState.subject/opponent(_port)` (live) or works on
  training frames where the subject is port 1 BY CONSTRUCTION (Peppi
  remap). Absolute-port reads elsewhere are the bug class behind the
  projectile-owner feature, the runner's stock/SD logic reading the human
  under Slippi Online, and the all-zero opponent on non-1+2 seating.

  The live decision path (agent, embeddings, runner) is at ZERO. The
  remaining reads are offline instruments and training internals that
  operate on remapped frames (subject == port 1 by construction) — listed
  with a RATCHET: counts may only fall, new files start at 0.
  """
  use ExUnit.Case, async: true

  @pattern ~r/players\[[12]\]|\bown_port == [12]\b|if [a-z_]+ == 1, do: 2, else: 1|player_port == 1\b/

  # The boundary: ports legitimately live here.
  @boundary ~w(lib/exphil/data/ lib/exphil_bridge/types.ex lib/exphil_bridge/melee_port.ex)

  # Offline instruments / training internals on REMAPPED frames (subject is
  # port 1 by construction). Ratchet — migrate to Labels/GameState helpers
  # and lower the number; never raise it.
  @legacy_max %{
    "lib/exphil/interp/cycle_sim.ex" => 8,
    "lib/exphil/training/opener_sampling.ex" => 4,
    "lib/exphil/eval/scenario_scan.ex" => 3,
    "lib/exphil/training/probe_regularizer.ex" => 2,
    "lib/exphil/interp/basin_rollout.ex" => 2,
    "lib/exphil/interp/absorber_entry.ex" => 2,
    "lib/exphil/training/streaming.ex" => 1,
    "lib/exphil/training/margin_sampling.ex" => 1,
    "lib/exphil/training/data.ex" => 1,
    "lib/exphil/training/augmentation.ex" => 1,
    "lib/exphil/interp/loop_stats.ex" => 1,
    "lib/exphil/interp/activations.ex" => 1,
    "lib/exphil/evaluation/metrics.ex" => 1,
    "lib/exphil/agents/dummies/policy_opponent.ex" => 1
  }

  # These MUST stay at zero: the live decision path.
  @live_path ~w(lib/exphil/agents/agent.ex lib/exphil/embeddings/game.ex lib/exphil/embeddings/game/projectiles.ex lib/exphil/bridge/async_runner.ex)

  defp count(path) do
    path
    |> File.read!()
    |> String.split("\n")
    |> Enum.count(fn line ->
      t = String.trim(line)
      not String.starts_with?(t, "#") and Regex.match?(@pattern, line)
    end)
  end

  test "the live decision path has no absolute-port reads" do
    for f <- @live_path, do: assert(count(f) == 0, "#{f} has #{count(f)} absolute-port read(s)")
  end

  test "absolute-port reads outside the boundary only ratchet down" do
    files = Path.wildcard("lib/**/*.ex") |> Enum.reject(fn f -> Enum.any?(@boundary, &String.starts_with?(f, &1)) end)

    offenders =
      for f <- files, n = count(f), n > Map.get(@legacy_max, f, 0), do: "#{f}: #{n} (max #{Map.get(@legacy_max, f, 0)})"

    assert offenders == [], "absolute-port reads (new or grown):\n" <> Enum.join(offenders, "\n")
  end

  test "GameState resolves subject/opponent from the stamp and occupied ports" do
    alias ExPhil.Bridge.GameState
    p = fn -> %{} end
    gs = %GameState{players: %{2 => p.(), 3 => p.()}, own_port: 2}
    assert GameState.subject_port(gs) == 2
    assert GameState.opponent_port(gs) == 3
    # no stamp: fallback is the configured/remapped port
    gs2 = %GameState{players: %{1 => p.(), 4 => p.()}}
    assert GameState.subject_port(gs2, 1) == 1
    assert GameState.opponent_port(gs2, 1) == 4
  end
end
