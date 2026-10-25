# Bare-VM runner: reuse compiled dependencies without rebuilding a shared NIF.
# devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs SCRIPT [ARGS]
Application.ensure_all_started(:crypto)
Application.ensure_all_started(:exphil)
client = if System.get_env("RECURRENT_CLIENT") == "host", do: :host, else: :cuda

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, memory_fraction: 0.18, preallocate: false]
)

Application.put_env(:exla, :default_client, client)
Nx.global_default_backend({EXLA.Backend, client: client})
Nx.Defn.global_default_options(compiler: EXLA, client: client, precision: :highest)
Logger.configure(level: :warning)
Code.compile_file(Path.expand("../edifice/lib/edifice/stateful/replay.ex"))
Code.compile_file(Path.expand("../edifice/lib/edifice/stateful.ex"))
Code.compile_file("lib/exphil/agents/agent.ex")
[script | args] = System.argv()
System.argv(args)
Code.require_file(script)
