# Export a policy + replay to a rewind-viewer session JSON.
#
#   mix run scripts/export_rewind.exs \
#     --policy checkpoints/fox_il_v2_edgeB_20260810_060518_best_policy.bin \
#     --replay eval_runs/0810_edgeB_pool/r1/Game_x.slp \
#     --out eval_runs/session.json [--port 1] [--delay-id N]
#
# Open priv/viewer/rewind_viewer.html in a browser and load the JSON.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Inspect
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replay: :string, out: :string, port: :integer, delay_id: :integer]
  )

policy = opts[:policy] || raise("--policy is required")
replay = opts[:replay] || raise("--replay is required")
out = opts[:out] || raise("--out is required")

open_opts = [player_port: opts[:port] || 1]
open_opts = if opts[:delay_id], do: open_opts ++ [delay_id: opts[:delay_id]], else: open_opts

{:ok, session} = Inspect.open(policy, replay, open_opts)
:ok = Inspect.export_session(session, out)

size_mb = Float.round(File.stat!(out).size / 1.0e6, 1)
Output.success("#{session.total} frames -> #{out} (#{size_mb} MB)")
Output.puts("  open priv/viewer/rewind_viewer.html and load it")
