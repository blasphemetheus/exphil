# Tier-1 report card (task #27): mechanics-health gates for a policy replay.
# Gate math lives in ExPhil.Interp.ReportCard (unit-tested; loops can call
# ReportCard.score/1 programmatically). This script is the CLI printer.
#
#   mix run scripts/report_card.exs <replay.slp> [more.slp ...]
#
# Port 1 is assumed to be the policy.

alias ExPhil.Interp.ReportCard

for path <- System.argv() do
  try do
    data = ExPhil.Interp.ReplayStats.load(path)
    result = ReportCard.evaluate(data)

    IO.puts("\n== REPORT CARD: #{Path.basename(path)} (#{data.n} frames)")

    for g <- result.gates do
      status = if g.pass, do: "PASS", else: "FAIL"
      vs = if is_float(g.value), do: Float.round(g.value, 2), else: g.value
      IO.puts("  [#{status}] #{String.pad_trailing(g.name, 26)} #{inspect(vs)}  (target #{g.target})")
    end

    IO.puts("  SCORE: #{result.passed}/#{result.total} gates" <>
              if(result.passed == result.total, do: "  ✓ ALL PASS", else: ""))
  rescue
    e ->
      # Truncated/corrupt replays (killed Dolphin, GOTCHA #58 orphans)
      # must not sink the whole batch — score the rest.
      IO.puts("\n== REPORT CARD: #{Path.basename(path)} — UNPARSEABLE, skipped (#{Exception.message(e)})")
  end
end
