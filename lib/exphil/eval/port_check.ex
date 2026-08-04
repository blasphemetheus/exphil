defmodule ExPhil.Eval.PortCheck do
  @moduledoc """
  Decide whether a replay's port setup matches what the run ASKED for.

  Extracted from `scripts/check_replay_ports.exs` (2026-08-03) so the
  decision is unit-testable. It gates every CPU-dummy eval block, and until
  now had no tests at all — while the bug it guards silently invalidated an
  entire era of recordings.

  The bug (GOTCHAS #57 / #57b): Slippi records each port's TYPE in the
  game-start block (0 HUMAN, 1 CPU, 2 DEMO, 3 empty), but Peppi's
  `PlayerMeta` does not expose it and libmelee never logged the achieved
  level. A dummy that silently came up HUMAN was therefore invisible — it
  just looked like a very passive CPU. On 2026-07-26, five of six recordings
  made with `--dummy cpu --dummy-cpu-level 9` produced a HUMAN port 2 (Wait
  76% of frames, never jumped, drifted onto the ledge); the tell was
  "level-9 CPUs don't ledge plank". Cause: an autostart race at character
  select. A fresh CPU also defaults to level 1, so a level mismatch is the
  same race firing before the slider drag finished.

  Port maps use string keys because they arrive as decoded JSON from the
  py-slippi reader: `%{"port" => 2, "type" => 1, "cpu_level" => 9}`.
  """

  @type port_info :: %{optional(String.t()) => term()}
  @type verdict ::
          :ok
          | {:error, :absent, String.t()}
          | {:error, :not_cpu, String.t()}
          | {:error, :wrong_level, String.t()}

  @human 0
  @cpu 1
  @demo 2
  @empty 3

  @doc "Human-readable name for Slippi's port type code."
  @spec type_name(integer() | nil) :: String.t()
  def type_name(@human), do: "HUMAN"
  def type_name(@cpu), do: "CPU"
  def type_name(@demo), do: "DEMO"
  def type_name(@empty), do: "empty"
  def type_name(other), do: "type=#{inspect(other)}"

  @doc """
  Verify `ports` against expectations.

  Options:
    * `:expect_cpu` — port number that must be a CPU (nil = no check)
    * `:expect_level` — required CPU level (only checked when the replay
      reports one; some builds report nil)

  Returns `:ok` or `{:error, reason_atom, message}`.
  """
  @spec verify([port_info()], keyword()) :: verdict()
  def verify(ports, opts \\ []) do
    want_port = Keyword.get(opts, :expect_cpu)
    want_level = Keyword.get(opts, :expect_level)

    cond do
      is_nil(want_port) ->
        :ok

      true ->
        port = Enum.find(ports, &(&1["port"] == want_port))
        check_port(port, want_port, want_level)
    end
  end

  defp check_port(nil, want_port, _want_level) do
    {:error, :absent, "expected a CPU on port #{want_port}, but that port is absent"}
  end

  defp check_port(port, want_port, want_level) do
    level = port["cpu_level"]

    cond do
      port["type"] != @cpu ->
        {:error, :not_cpu,
         "port #{want_port} is #{type_name(port["type"])}, NOT a CPU — the dummy never " <>
           "finished character-select setup (GOTCHAS #57)"}

      want_level && level && level != want_level ->
        {:error, :wrong_level,
         "port #{want_port} is a CPU but level #{level}, expected #{want_level} — Melee " <>
           "defaults a fresh CPU to 1, so this is the autostart race firing before the " <>
           "slider drag finished"}

      true ->
        :ok
    end
  end
end
