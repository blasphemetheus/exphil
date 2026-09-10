defmodule ExPhil.Networks.BackboneTableParityTest do
  @moduledoc """
  INVARIANTS.md item 3: an architecture needs a CLI entry, a dispatcher
  clause, defaults, and an output-size rule — four hand-maintained tables
  today. This pins them against each other until per-backbone specs make
  the missing-row class unrepresentable.

  2026-09-09 audit: 97 dispatchable, 75 with NO defaults clause (silent
  `[]` — the xlstm/:mamba3/:retnet bug class), 27 dispatchable but
  rejected at the CLI door, 52 without an output-size clause (loud
  CaseClauseError, not silent). The `[]` fallback is now a warned
  baseline; the 27 are listed; this test ratchets the rest down.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config

  @backbone_src "lib/exphil/networks/policy/backbone.ex"

  # Dispatchable = a bespoke clause in the source (not introspectable; read
  # the source, same trade as flag_parity_test) OR a `build:` recipe on the
  # spec row (item 3 phase C: the generic `other ->` clause builds those).
  defp dispatch_atoms do
    src = File.read!(@backbone_src)
    [_, block] = Regex.run(~r/def build_temporal_backbone\((.*?)\n  end\n/s, src)

    bespoke =
      Regex.scan(~r/^\s+((?::[a-z][a-z0-9_]*\s*,?\s*)+)->/m, block)
      |> Enum.flat_map(fn [_, g] -> Regex.scan(~r/:([a-z][a-z0-9_]*)/, g) |> Enum.map(fn [_, a] -> String.to_atom(a) end) end)

    Enum.uniq(bespoke ++ Config.spec_built_backbones())
  end

  test "a backbone is EITHER a bespoke clause OR a spec recipe, never both" do
    src = File.read!(@backbone_src)
    [_, block] = Regex.run(~r/def build_temporal_backbone\((.*?)\n  end\n/s, src)

    bespoke =
      Regex.scan(~r/^\s+((?::[a-z][a-z0-9_]*\s*,?\s*)+)->/m, block)
      |> Enum.flat_map(fn [_, g] -> Regex.scan(~r/:([a-z][a-z0-9_]*)/, g) |> Enum.map(fn [_, a] -> String.to_atom(a) end) end)

    both = Enum.filter(bespoke, &(&1 in Config.spec_built_backbones()))
    assert both == [], "shadowed recipes (clause wins silently): #{inspect(both)}"
  end

  # Known exceptions, each with a reason. Shrink, never grow, without a note.
  @dispatch_not_valid [
    # dispatcher spells it :mamba_2 while the CLI/defaults use :mamba2 — a
    # rename the review flagged (dead :mamba_2 defaults clause). Unify.
    :mamba_2
  ]
  @valid_not_dispatch [
    # listed for years; no build_temporal_backbone clause. Either wire or drop.
    :hybrid
  ]

  # RATCHET: number of dispatchable backbones still relying on the generic
  # baseline. May only go DOWN. 74 after phase B (spec map; xlstm family had an explicit clause).
  @baseline_reliant_max 74

  test "every dispatchable backbone is a valid CLI backbone" do
    valid = MapSet.new(Config.valid_backbones())
    missing = dispatch_atoms() |> Enum.reject(&MapSet.member?(valid, &1)) |> Enum.reject(&(&1 in @dispatch_not_valid))
    assert missing == [], "dispatchable but rejected at the CLI door: #{inspect(missing)}"
  end

  test "every valid CLI backbone is dispatchable" do
    dispatch = MapSet.new(dispatch_atoms())
    orphans = Config.valid_backbones() |> Enum.reject(&MapSet.member?(dispatch, &1)) |> Enum.reject(&(&1 in @valid_not_dispatch))
    assert orphans == [], "valid but no dispatcher clause (would crash at build): #{inspect(orphans)}"
  end

  test "no backbone gets empty defaults; baseline reliance only ratchets down" do
    reliant =
      for b <- dispatch_atoms(), Config.backbone_defaults_baseline?(b), do: b

    for b <- dispatch_atoms() do
      d = Config.backbone_defaults(b)
      assert d != [], "#{inspect(b)} returned [] defaults"
      assert Keyword.has_key?(d, :temporal) or b == :mlp, "#{inspect(b)} defaults lack :temporal"
      assert Keyword.has_key?(d, :precision), "#{inspect(b)} defaults lack :precision"
    end

    assert length(reliant) <= @baseline_reliant_max,
           "baseline-reliant backbones grew to #{length(reliant)} (max #{@baseline_reliant_max}); add clauses, don't raise the ratchet"
  end
end
