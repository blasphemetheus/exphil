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
Code.compile_file(Path.expand("../edifice/lib/edifice/utils/sampling_random.ex"))

if System.get_env("SAMPLING_REFERENCE") == "1" do
  # Validation-only baseline: identical sampler with rolled Nx uniform draws.
  File.read!("lib/exphil/networks/policy/sampling.ex")
  |> String.replace("SamplingRandom.uniform(", "Nx.Random.uniform(")
  |> String.replace(":unrolled_threefry_v1", ":rolled_reference")
  |> Code.compile_string("sampling_rolled_reference.ex")
else
  Code.compile_file("lib/exphil/networks/policy/sampling.ex")
end

Code.compile_file(Path.expand("../edifice/lib/edifice/stateful/replay.ex"))
Code.compile_file(Path.expand("../edifice/lib/edifice/stateful.ex"))
Code.compile_file("lib/exphil/agents/agent.ex")
Code.compile_file("lib/exphil/sim/action_delay.ex")
[script | args] = System.argv()
System.argv(args)
Code.require_file(script)
