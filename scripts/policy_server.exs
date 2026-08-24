# Resident policy server (JIT_WARMUP option 3; POLICY_SERVER_DESIGN.md).
# Boots a named node, starts ExPhil.Agents.PolicyServer, pre-warms the
# given checkpoint(s), and serves Agents to sessions launched with
# --policy-server. Runs until killed.
#
#   mix run scripts/policy_server.exs --policy checkpoints/ms_g19_ep4.bin
#   mix run scripts/policy_server.exs --policy a.bin,b.bin --stateful-step
#
# LAWS while this runs: it is a live exphil beam — NO MIX in any
# EXLA-sharing repo (same rule as training). Distribution is loopback
# only (ERL_AFLAGS set below must not be overridden).
alias ExPhil.Agents.PolicyServer
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, stateful_step: :boolean, cookie: :string]
  )

policies = String.split(opts[:policy] || "checkpoints/ms_g19_ep4.bin", ",")

Output.banner("ExPhil Policy Server")

# Named node, loopback only. epmd auto-starts via Node.start.
{:ok, _} = Node.start(:"exphil_policy@127.0.0.1", :longnames)
Node.set_cookie(String.to_atom(opts[:cookie] || "exphil_policy_local"))

{:ok, _pid} = PolicyServer.start_link()

agent_opts = if opts[:stateful_step], do: [stateful_step: true], else: []

for policy <- policies do
  case PolicyServer.preload(policy, agent_opts) do
    {:ok, ms} -> Output.success("preloaded #{policy} (warmup #{ms}ms)")
    {:error, reason} -> Output.error("preload #{policy} failed: #{inspect(reason)}")
  end
end

status = PolicyServer.status()
Output.config([{"Node", Node.self()}, {"Rev", status.rev}, {"Warmed", inspect(status.warmed)}])
Output.puts("Serving. Sessions: add --policy-server to play_dolphin_async. Ctrl-C to stop.")

Process.sleep(:infinity)
