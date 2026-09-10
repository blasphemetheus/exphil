defmodule ExPhil.Interp.LabelSourceTest do
  @moduledoc """
  INVARIANTS.md item 9, structural half: no instrument may compute its own
  "what did the player do from this frame" label. Every probe/scan reads
  the successor-aligned input through `ExPhil.Interp.Labels`. This greps
  the instrument scripts for same-frame controller reads used as labels.
  """
  use ExUnit.Case, async: true

  @instruments Path.wildcard("scripts/probe_*.exs") ++ Path.wildcard("scripts/*_scan.exs")

  # A same-frame controller read: `:array.get(i, arr).controller`,
  # `f.controller`, `pf.controller` — the leaked label shape.
  @forbidden ~r/(\:array\.get\([^)]*\)|\bp?f)\.controller\b/

  # Legacy multishine-era instruments that read the same-frame controller.
  # Some are DESCRIPTIVE (what input was in effect on this frame — fine);
  # some are calibration labels (leaked — probe_cal_drift compares model
  # output for frame f against f.controller). RATCHET: counts may only
  # fall; migrate to Labels (issued_input for targets, producing_input +
  # a `# same-frame-descriptive` marker for descriptive reads).
  @legacy_max %{
    "scripts/probe_cycle_margins.exs" => 4,
    "scripts/probe_cal_drift.exs" => 3,
    "scripts/probe_absorber_entry.exs" => 3,
    "scripts/probe_replay_basin.exs" => 1
  }

  test "instruments read labels through ExPhil.Interp.Labels, never frames[i].controller" do
    assert @instruments != []

    per_file =
      for path <- @instruments do
        n =
          path
          |> File.read!()
          |> String.split("\n")
          |> Enum.count(fn line ->
            t = String.trim(line)

            not String.starts_with?(t, "#") and
              not String.contains?(line, "Labels.") and
              not String.contains?(line, "same-frame-descriptive") and
              Regex.match?(@forbidden, line)
          end)

        {path, n}
      end

    offenders =
      for {path, n} <- per_file, n > Map.get(@legacy_max, path, 0), do: "#{path}: #{n} (max #{Map.get(@legacy_max, path, 0)})"

    assert offenders == [],
           "same-frame controller reads used as labels (new or grown):\n" <> Enum.join(offenders, "\n")
  end
end
