# AI-generated. See docs/guides/MAMBA_BATCH_PARITY.md.
# elixir scripts/run_framework_validation.exs STACK_LIB SCRIPT [SCRIPT_ARGS...]
# Reuses isolated compiled dependencies. Never invokes Mix or rebuilds a NIF.
[stack, script | args] = System.argv()
base = Path.expand("_build/dev/lib")
stack = Path.expand(stack)

paths = fn root ->
  Path.wildcard(Path.join(root, "*/ebin"))
  |> Map.new(fn path -> {path |> Path.dirname() |> Path.basename(), path} end)
end

base_paths = paths.(base)
overrides = paths.(stack)

for app <- ["nx", "exla", "xla", "axon", "polaris"] do
  unless Map.has_key?(overrides, app), do: raise("missing compiled #{app} in #{stack}")
end

selected = Map.merge(base_paths, overrides)

argv =
  Enum.flat_map(Enum.sort(selected), fn {_, path} -> ["-pa", path] end) ++
    ["scripts/run_recurrent_validation.exs", script | args]

{_, status} = System.cmd("elixir", argv, into: IO.stream(:stdio, :line), stderr_to_stdout: true)
System.halt(status)
