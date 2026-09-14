{opts, [], []} = OptionParser.parse(System.argv(), strict: [report: :string, out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
source = Keyword.fetch!(opts, :report)
report = source |> File.read!() |> Jason.decode!()
policy = Map.fetch!(report, "policy")
digest = :crypto.hash(:sha256, File.read!(policy)) |> Base.encode16(case: :lower)

unless digest == report["policy_sha256"],
  do: raise("fit report does not match the current checkpoint")

{:ok, export} =
  Nx.with_default_backend(Nx.BinaryBackend, fn -> ExPhil.Training.Checkpoint.load_policy(policy) end)
ExPhil.Networks.Policy.ExecutionContract.verify_report!(export.config, report)

result =
  ExPhil.Eval.EarlyTeacherGate.check(report)
  |> Map.merge(%{source_report: source, policy: policy, policy_sha256: digest})

File.write!(out, Jason.encode!(result, pretty: true), [:exclusive])
IO.inspect(result)
unless result.ready, do: System.halt(1)
