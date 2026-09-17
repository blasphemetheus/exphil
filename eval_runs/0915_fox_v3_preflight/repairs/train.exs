[path] = System.argv()
args = path |> File.read!() |> JSON.decode!()
{_, status} = System.cmd("mix", args, into: IO.stream(), stderr_to_stdout: true,
  env: [{"EXLA_TARGET", "cuda"}, {"EXPHIL_GPU_MEMORY_FRACTION", "0.70"}])
System.halt(status)
