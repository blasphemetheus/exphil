defmodule ExPhil.Harness.ScriptLintTest do
  @moduledoc """
  Static guards over `scripts/` and `eval_runs/` for failures that cost real
  time and are invisible to every other test (they live in shell, not
  Elixir).

  ## What is deliberately NOT linted here, and why

  Two candidates from the 2026-08-03 plan were investigated and dropped
  rather than shipped as theatre — recorded so nobody "helpfully" adds them:

    * **bare `/nix/store` paths** (GOTCHA #83, cost a whole eval sweep):
      vacuous in this repo. The path that got garbage-collected lives in an
      EXTERNAL wrapper (`~/.config/Slippi Launcher/.../*.AppImage`), which a
      repo test cannot see. Note that `scripts/overnight_newera8.sh:154` and
      `scripts/checkpoint_ladder.sh:53` already implement the correct guard
      — they verify the wrapper's store roots still exist before launching —
      and newer eval scripts simply never adopted it. Reuse that, don't lint.

    * **`Axon.build` without `:seed`** (GOTCHA #76): every occurrence in
      `lib/` today is inside a `@moduledoc` example; the one real path
      (`Training.Utils.build_compiled/2`) forwards opts. Forcing a default
      seed would be actively HARMFUL: seed-farm diversity comes precisely
      from Axon's system-time seeding, so a fixed default would make every
      "different seed" identical.
  """
  use ExUnit.Case, async: true

  @shell_globs ["scripts/*.sh", "eval_runs/*.sh"]

  # A pattern is self-matching when it mentions one of OUR script names — the
  # calling wrapper's own /proc cmdline contains that string, so `pgrep -f`
  # finds itself. GOTCHA #63/#75 — cost two launches and one exit-144 shell
  # death. Basenames are DERIVED from the tree (v1 of this lint only matched
  # patterns carrying a file extension and therefore missed
  # `pgrep -f "overnight_newera8|dagger_drill"`, the exact shape of the bug).
  @pgrep_line_re ~r/(pgrep|pkill)\s+-[a-z]*f[a-z]*\s+(.*)$/

  # Mitigations that make a self-matching pattern safe.
  @mitigations [
    # exclude our own pid
    "grep -v $$",
    # the bracket trick: [d]agger matches the process but not this cmdline
    "[",
    # explicit opt-out for a reviewed case
    "lint:allow-self-match"
  ]

  defp shell_scripts do
    @shell_globs
    |> Enum.flat_map(&Path.wildcard/1)
    |> Enum.sort()
  end

  # Every script basename we ship, without extension — the tokens that can
  # appear in a wrapper's own cmdline.
  defp script_basenames do
    ["scripts/*.exs", "scripts/*.sh", "eval_runs/*.sh"]
    |> Enum.flat_map(&Path.wildcard/1)
    |> Enum.map(&(&1 |> Path.basename() |> Path.rootname()))
    |> Enum.uniq()
  end

  describe "pgrep/pkill self-match (GOTCHA #63/#75)" do
    test "every -f pattern naming a script has a mitigation on the same line" do
      names = script_basenames()

      offenders =
        for path <- shell_scripts(),
            {line, idx} <- Enum.with_index(String.split(File.read!(path), "\n"), 1),
            not String.starts_with?(String.trim(line), "#"),
            [_, _, pattern] = Regex.run(@pgrep_line_re, line) || [nil, nil, nil],
            pattern != nil,
            Enum.any?(names, &String.contains?(pattern, &1)),
            not Enum.any?(@mitigations, &String.contains?(line, &1)),
            do: "#{path}:#{idx}: #{String.trim(line)}"

      assert offenders == [],
             """
             `pgrep -f` / `pkill -f` with a pattern naming one of our scripts
             matches the CALLING script's own cmdline (GOTCHA #63/#75: a
             busy-guard that refuses to run because it found itself; a pkill
             that killed its own shell with exit 144).

             Add one of: `| grep -v $$`, the bracket trick (`"[d]agger_drill"`),
             or `# lint:allow-self-match` if the case is reviewed and safe.

             #{Enum.join(offenders, "\n")}
             """
    end
  end

  describe "shell scripts are syntactically valid" do
    test "bash -n parses every script" do
      broken =
        for path <- shell_scripts(),
            {out, code} = System.cmd("bash", ["-n", path], stderr_to_stdout: true),
            code != 0,
            do: "#{path}: #{String.trim(out)}"

      assert broken == [], "shell syntax errors:\n#{Enum.join(broken, "\n")}"
    end
  end
end
