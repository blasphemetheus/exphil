# JIT residual attribution (JIT_WARMUP.md step: read the warmup stage lines).
# Boots the Agent with the production-candidate checkpoint and runs :warmup
# standalone — no Dolphin. The Agent's stage instrumentation (embed / sample1 /
# sample2 / confidence) does the attribution.
Logger.configure(level: :info)

alias ExPhil.Agents.Agent

policy = System.get_env("WARMUP_POLICY", "checkpoints/ms_g19_ep4.bin")
IO.puts("=== warmup attribution: #{policy} ===")

t0 = System.monotonic_time(:millisecond)
stateful = System.get_env("WARMUP_STATEFUL") == "1"
IO.puts("[attrib] stateful_step: #{stateful}")
{:ok, agent} = Agent.start_link(policy_path: policy, stateful_step: stateful)
t1 = System.monotonic_time(:millisecond)
IO.puts("[attrib] Agent.start_link (checkpoint load + build): #{t1 - t0}ms")

{:ok, elapsed} = GenServer.call(agent, :warmup, 600_000)
IO.puts("[attrib] warmup total: #{elapsed}ms")

# A second warmup call in the SAME process should be ~instant (in-memory JIT
# cache) — confirms the stages measure compile, not steady-state inference.
{:ok, elapsed2} = GenServer.call(agent, :warmup, 600_000)
IO.puts("[attrib] warmup again (same process): #{elapsed2}ms")
