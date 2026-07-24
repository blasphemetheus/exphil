# Does a trained policy reproduce its OWN fixture, offline?
#
#   mix run scripts/eval_policy_on_fixture.exs --policy checkpoints/X.bin
#     [--fixture test/fixtures/replays/fox_multishine_closed.slp]
#
# The cheap (no-Dolphin, ~10s) discriminator between the two reasons a
# drill policy fails live:
#
#   HIGH agreement here + fails live  -> the live state stream differs from
#     the parsed one (GOTCHAS #81: Peppi and libmelee disagree on
#     action_frame), or the action_delay convention is off. The policy
#     learned the right function of the WRONG features.
#
#   LOW agreement here                -> it never learned the mapping at
#     all; the training loss was measured on something else (weighting,
#     shifted labels, a head you are not comparing).
#
# Feeds states in fixture order and, when the policy uses the prev-action
# channel, feeds back the policy's OWN previous emission — the same
# closed-loop convention as live inference, so a policy that only works
# under teacher forcing is not flattered.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [policy: :string, fixture: :string, port: :integer, limit: :integer])

policy_path = opts[:policy] || raise "--policy required"
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
port = opts[:port] || 1

{:ok, replay} = ExPhil.Data.Peppi.parse(Path.expand(fixture))

frames =
  replay
  |> ExPhil.Data.Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn f ->
    c = f.controller
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)
  |> then(fn fs -> if opts[:limit], do: Enum.take(fs, opts[:limit]), else: fs end)

Output.banner("Policy vs its own fixture (offline)")
Output.puts("policy:  #{Path.basename(policy_path)}")
Output.puts("fixture: #{Path.basename(fixture)} (#{length(frames)} frames)")

{:ok, agent} = Agent.start_link(policy_path: policy_path, deterministic: true)
Agent.warmup(agent)

# Compare against the input recorded on the SAME frame (delay 0) and on the
# NEXT frame (delay 1). Whichever scores higher tells you which convention
# the policy actually learned — worth knowing before blaming the bridge.
results =
  frames
  |> Enum.map(fn f ->
    case Agent.get_controller(agent, f.game_state, player_port: port) do
      {:ok, c} -> {c.button_b, c.button_x, f.controller.button_b, f.controller.button_x}
      _ -> nil
    end
  end)
  |> Enum.reject(&is_nil/1)

n = length(results)

same_frame =
  Enum.count(results, fn {pb, px, ab, ax} -> pb == ab and px == ax end)

next_frame =
  results
  |> Enum.zip(tl(results) ++ [nil])
  |> Enum.reject(fn {_a, b} -> is_nil(b) end)
  |> Enum.count(fn {{pb, px, _, _}, {_, _, ab, ax}} -> pb == ab and px == ax end)

Output.puts("")
Output.puts("B/X agreement vs controller_N   (delay 0): #{Float.round(100.0 * same_frame / max(n, 1), 1)}%")
Output.puts("B/X agreement vs controller_N+1 (delay 1): #{Float.round(100.0 * next_frame / max(n - 1, 1), 1)}%")

# What the policy is actually pressing — a policy stuck on "B always" shows
# up here instantly, and no agreement number explains that as clearly.
pb_rate = Enum.count(results, fn {pb, _, _, _} -> pb end) / max(n, 1)
px_rate = Enum.count(results, fn {_, px, _, _} -> px end) / max(n, 1)
ab_rate = Enum.count(results, fn {_, _, ab, _} -> ab end) / max(n, 1)
ax_rate = Enum.count(results, fn {_, _, _, ax} -> ax end) / max(n, 1)

Output.puts("")
Output.puts("press rates   policy: B=#{Float.round(pb_rate * 100, 1)}% X=#{Float.round(px_rate * 100, 1)}%")
Output.puts("            fixture: B=#{Float.round(ab_rate * 100, 1)}% X=#{Float.round(ax_rate * 100, 1)}%")

best = max(same_frame / max(n, 1), next_frame / max(n - 1, 1))

cond do
  best > 0.95 ->
    Output.success(
      "Policy reproduces the fixture offline. A live failure is then a STATE-STREAM " <>
        "problem (parsed vs live action_frame, GOTCHAS #81) or an action_delay mismatch — " <>
        "not a learning failure."
    )

  abs(pb_rate - ab_rate) > 0.2 or abs(px_rate - ax_rate) > 0.2 ->
    Output.warning(
      "Press rates do not match the fixture — the policy collapsed to a constant " <>
        "(the classic 'hold the button' failure), so it never learned the alternation."
    )

  true ->
    Output.warning("Policy does not reproduce its own fixture offline — suspect the training setup, not the bridge.")
end
