defmodule ExPhil.Training.Inhibitor do
  @moduledoc """
  Idle-suspend inhibitor for long-running scripts (training, evals,
  corpus builds).

  2026-08-10: the desktop idle-suspended mid-run TWICE — it froze the
  convC1 training for ~25 minutes (survived by luck: CUDA context came
  back on resume) and is the prime suspect for the 15:20 PS-recording
  CUDA wedge ("Delay kernel timed out"). Background compute does not
  count as user activity, so any long GPU job needs an explicit
  inhibitor — the RC-hardening lesson (07-31 postmortem) extended to
  every script.

  Mechanism: spawns `systemd-inhibit --what=sleep:idle` around a tiny
  `read`-on-stdin child through a Port. Ports close their pipes when the
  owning BEAM exits — including crashes and SIGKILL — so the `read`
  returns, systemd-inhibit exits, and the lock is released with the
  script's lifetime. No cleanup path required.

  No-ops (with a note at :debug) when systemd-inhibit is unavailable
  (CI, containers, non-systemd hosts).
  """

  require Logger

  @doc """
  Hold a sleep:idle inhibitor for the life of this BEAM. Returns the
  port (or nil when unavailable). Call once near script startup:

      ExPhil.Training.Inhibitor.hold("fox_il training")
  """
  @spec hold(String.t()) :: port() | nil
  def hold(why) do
    case System.find_executable("systemd-inhibit") do
      nil ->
        Logger.debug("[Inhibitor] systemd-inhibit not found — suspend inhibition skipped")
        nil

      exe ->
        sh = System.find_executable("sh") || "/bin/sh"

        port =
          Port.open({:spawn_executable, sh}, [
            :binary,
            args: [
              "-c",
              # `read` blocks on the port's stdin; when the BEAM dies the
              # pipe closes, read returns, the inhibitor exits
              "exec '#{exe}' --what=sleep:idle --who=exphil --why='#{String.replace(why, "'", "")}' sh -c 'read _line'"
            ]
          ])

        Logger.debug("[Inhibitor] holding sleep:idle (#{why})")
        port
    end
  end
end
