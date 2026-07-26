# Does a policy agree with its expert on the states IT ACTUALLY VISITS?
#
#   mix run scripts/eval_policy_on_rollout.exs --policy checkpoints/X.bin \
#     --rollout ~/Slippi/Game_Y.slp [--expert multishine] [--fixture F.slp]
#
# The closed-loop counterpart to eval_policy_on_fixture.exs, which feeds
# FIXTURE states in fixture order. That is teacher forcing on the state
# channel: it can only ever ask "on the trajectory you memorized, do you
# reproduce it?" — so a policy that is perfect on-manifold and catastrophic
# one frame off scores ~99% and looks healthy. Measured 2026-07-26: the
# multishine probe policy scored 99.9% there while freezing solid live.
#
# This script instead replays a ROLLOUT (a Slippi replay of the policy
# actually playing), labels every visited frame with the scripted expert, and
# asks whether the policy would do what the expert says THERE.
#
# The headline number is the share of visited frames that are OFF-MANIFOLD —
# {action, action_frame} pairs the training fixture never contains. Measured
# 2026-07-26 on the multishine probe policy: 79% of its own rollout was states
# it had never been trained on. That alone is the exposure-bias diagnosis.
#
# READ THE TWO AGREEMENT COLUMNS WITH CARE — two confounds, both real:
#
#   1. Membership is POINTWISE, the policy is TEMPORAL (GRU, window 16). A
#      frame can be pointwise on-manifold yet reached through a 16-frame
#      history the model never saw, so its hidden state is off-distribution
#      anyway. On-manifold agreement is therefore an OPTIMISTIC label on a
#      pessimistic reality, not a clean control.
#   2. Agreement is measured against the EXPERT (table + recovery rules), not
#      against the fixture's recorded inputs. Many recovery rules emit "start
#      a shine" (press B) — which is exactly what a policy STUCK holding B is
#      already doing, so the two agree by coincidence. This inflates
#      off-manifold agreement specifically in the failure mode we care about.
#
# Observed consequence: the probe policy scored 12.7% on-manifold and 23.8%
# off-manifold — inverted from what a naive reading predicts, for confound 2.
# Treat the columns as a comparison BETWEEN policies on the same rollout
# (before vs after DAgger), not as absolute quality.
#
# Exit code 1 if off-manifold agreement is below --min-agreement (default 0.5)
# so a loop can gate on it.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.{Agent, MultishineExpert}
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      rollout: :string,
      fixture: :string,
      port: :integer,
      limit: :integer,
      min_agreement: :float,
      stochastic: :boolean
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
rollout_path = opts[:rollout] || raise "--rollout required"
fixture_path = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
port = opts[:port] || 1
min_agreement = opts[:min_agreement] || 0.5

Output.banner("Policy vs expert on its OWN rollout (closed-loop)")
Output.puts("policy:  #{Path.basename(policy_path)}")
Output.puts("rollout: #{Path.basename(rollout_path)}")

frames_of = fn path ->
  case Peppi.parse(path) do
    {:ok, replay} ->
      Peppi.to_training_frames(replay, player_port: port, opponent_port: 3 - port)
      |> Enum.reject(&(&1.game_state.frame < 0))

    {:error, reason} ->
      # A rollout killed mid-game is truncated and peppi refuses it. Say so
      # plainly — this is the single most common reason this script fails.
      Output.error("Could not parse #{Path.basename(path)}: #{inspect(reason)}")

      Output.error(
        "If the run was killed mid-game the .slp is truncated; use --seconds so it SDs to a clean end."
      )

      System.halt(1)
  end
end

fixture_frames = frames_of.(fixture_path)
rollout_frames = frames_of.(rollout_path)

rollout_frames =
  if opts[:limit], do: Enum.take(rollout_frames, opts[:limit]), else: rollout_frames

# The training manifold: every {action, action_frame} the fixture contains.
manifold =
  fixture_frames
  |> Enum.map(fn f ->
    p = f.game_state.players[port]
    {trunc(p.action), trunc(p.action_frame)}
  end)
  |> MapSet.new()

Output.puts("fixture manifold: #{MapSet.size(manifold)} distinct {action, af} pairs")
Output.puts("rollout frames:   #{length(rollout_frames)}")

expert = MultishineExpert.from_fixture(fixture_path)

# Deterministic by default: we are measuring what the policy BELIEVES, not
# what a sample happens to draw. Sampling dilutes agreement uniformly and
# would hide the on/off-manifold gap this script exists to expose.
{:ok, agent} =
  Agent.start_link(
    policy_path: policy_path,
    deterministic: not (opts[:stochastic] || false),
    deterministic_buttons: not (opts[:stochastic] || false)
  )

# Walk the rollout, asking the policy for an action at each visited state and
# comparing to the expert's label for that same state.
{on_tot, on_ok, off_tot, off_ok, off_states} =
  Enum.reduce(rollout_frames, {0, 0, 0, 0, %{}}, fn f, {ont, ono, offt, offo, seen} ->
    player = f.game_state.players[port]
    key = {trunc(player.action), trunc(player.action_frame)}

    case MultishineExpert.label(expert, player) do
      :skip ->
        {ont, ono, offt, offo, seen}

      {:ok, want} ->
        {:ok, got} = Agent.get_action(agent, f.game_state, player_port: port)

        agree? =
          Map.get(got, :button_b, false) == Map.get(want, :button_b, false) and
            Map.get(got, :button_x, false) == Map.get(want, :button_x, false)

        if MapSet.member?(manifold, key) do
          {ont + 1, ono + if(agree?, do: 1, else: 0), offt, offo, seen}
        else
          {ont, ono, offt + 1, offo + if(agree?, do: 1, else: 0),
           Map.update(seen, key, 1, &(&1 + 1))}
        end
    end
  end)

pct = fn ok, tot -> if tot == 0, do: 0.0, else: Float.round(ok * 100 / tot, 1) end
total = on_tot + off_tot

Output.puts("")
Output.puts("states ON  the training manifold: #{on_tot} (#{pct.(on_tot, total)}% of rollout)")
Output.puts("  expert agreement: #{pct.(on_ok, on_tot)}%")
Output.puts("states OFF the training manifold: #{off_tot} (#{pct.(off_tot, total)}% of rollout)")
Output.puts("  expert agreement: #{pct.(off_ok, off_tot)}%")

if off_states != %{} do
  Output.puts("")
  Output.puts("most-visited UNSEEN {action, af} states:")

  off_states
  |> Enum.sort_by(&(-elem(&1, 1)))
  |> Enum.take(8)
  |> Enum.each(fn {{a, af}, n} -> Output.puts("  action #{a} af #{af}: #{n} frames") end)
end

off_ratio = if off_tot == 0, do: 1.0, else: off_ok / off_tot

Output.puts("")

cond do
  off_tot == 0 ->
    Output.success("Rollout never left the training manifold — no exposure-bias evidence here.")

  off_ratio >= min_agreement ->
    Output.success(
      "Off-manifold agreement #{pct.(off_ok, off_tot)}% — the policy generalizes off-trajectory."
    )

  true ->
    Output.warning(
      "Off-manifold agreement #{pct.(off_ok, off_tot)}% (< #{min_agreement * 100}%) — EXPOSURE BIAS. " <>
        "The policy is fine where it was trained and wrong where it actually goes; more epochs on the " <>
        "same fixture cannot fix this. Aggregate these states with dagger_drill.exs."
    )

    System.halt(1)
end
