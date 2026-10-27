# AI-generated; attribution and poem: docs/guides/RESYNC_PHASES.md.
# Invoke through scripts/run_recurrent_validation.exs with RECURRENT_CLIENT=host.
filters =
  if "--phases-only" in System.argv(),
    do: [exclude: [:test], include: [:resync_phase]],
    else: [exclude: [:slow, :integration, :external]]

ExUnit.start(filters ++ [timeout: 180_000, max_cases: 1])
Code.compile_file("lib/exphil/cli.ex")
Code.require_file("test/exphil/agents/agent_stateful_min_gru_test.exs")
Code.require_file("test/exphil/agents/agent_stateful_step_test.exs")
Code.require_file("test/exphil/cli_test.exs")
