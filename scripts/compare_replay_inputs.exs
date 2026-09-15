# Exact exposed input/state audit, inclusive bounds (default first = -39).
# mix run --no-start scripts/compare_replay_inputs.exs SOURCE RERUN LAST [FIRST]
# For handoff frame 344, use LAST=343; subsequent policy inputs intentionally differ.
# Missing data or any mismatch exits 1.
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ReplayPrefixAudit

{source, rerun, first, last} =
  case System.argv() do
    [a, b, last] -> {a, b, -39, String.to_integer(last)}
    [a, b, last, first] -> {a, b, String.to_integer(first), String.to_integer(last)}
    _ -> raise ArgumentError, "usage: compare_replay_inputs.exs SOURCE RERUN LAST [FIRST]"
  end

{:ok, a} = Peppi.parse(source)
{:ok, b} = Peppi.parse(rerun)
report = ReplayPrefixAudit.compare(a.frames, b.frames, first, last)

for port <- report.ports do
  IO.puts("port #{port.port}: compared #{port.compared}/#{report.expected_frames} frames")

  for kind <- [:missing, :input, :state] do
    IO.puts("  first #{kind} mismatch: #{inspect(port[kind], limit: :infinity)}")
  end
end

IO.puts("exact exposed inputs/state equal: #{report.valid}")
unless report.valid, do: System.halt(1)
