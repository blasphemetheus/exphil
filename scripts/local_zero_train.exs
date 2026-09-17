# Tiny Elixir launcher keeps the exact training arguments in the experiment.
path = List.first(System.argv()) || "eval_runs/0915_local_zero/bootstrap/train_args.json"
args = path |> File.read!() |> JSON.decode!()

{_, status} =
  System.cmd("mix", args,
    into: IO.stream(),
    stderr_to_stdout: true,
    env: [{"EXLA_TARGET", "cuda"}, {"EXPHIL_GPU_MEMORY_FRACTION", "0.15"}]
  )

System.halt(status)
