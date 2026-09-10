defmodule ExPhil.Training.Config.FlagDocsTest do
  @moduledoc "INVARIANTS.md item 2 phase C: the TRAINING.md flag reference is generated, never hand-edited."
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config.FlagDocs

  # RATCHET: table flags without a description. 31 at generation (2026-09-09).
  # Add descriptions to FlagDocs.@docs; never raise this number.
  @undocumented_max 31

  test "the committed reference section equals a fresh render (drift fails)" do
    committed = FlagDocs.committed()
    assert committed != nil, "TRAINING.md has no generated flag-reference section; run FlagDocs.write!()"

    assert committed == FlagDocs.render(),
           "docs/guides/TRAINING.md flag reference is stale — run: mix run -e 'ExPhil.Training.Config.FlagDocs.write!()'"
  end

  test "every parser flag appears in the rendered table exactly once" do
    rendered = FlagDocs.render()

    for flag <- ExPhil.Training.Config.Parser.flags() do
      n = rendered |> String.split("| `#{flag}` |") |> length() |> Kernel.-(1)
      assert n == 1, "#{flag} appears #{n} times in the rendered table"
    end
  end

  test "undocumented flags only ratchet down" do
    u = FlagDocs.undocumented()
    assert length(u) <= @undocumented_max, "undocumented flags grew to #{length(u)}: #{inspect(u)}"
  end
end
