defmodule ExPhil.Networks.EdificeRegistryLinkTest do
  @moduledoc """
  INVARIANTS.md item 3 phase C (cross-repo half).

  The exphil dispatcher (`Policy.Backbone.build_temporal_backbone/2`) wires
  edifice modules by direct alias, not through `Edifice.build/2`, so an
  edifice rename or deregistration would only surface as an UndefinedFunctionError
  at build time. Edifice's registry is now derived from one family-grouped
  list (`Edifice.list_families/0` == its `@registry_by_family`); this test
  pins every module the dispatcher aliases to that registry.
  """
  use ExUnit.Case, async: true

  @backbone_src "lib/exphil/networks/policy/backbone.ex"

  # Modules aliased for helpers only (not registry names) go here WITH a reason.
  # Empty as of 2026-09-09: every aliased module is a registered architecture.
  @not_registered []

  # Bespoke clauses alias their module in the source; spec-built backbones
  # (item 3 phase C) name theirs in the `build:` recipe on the Config row.
  defp aliased_modules do
    from_source =
      File.read!(@backbone_src)
      |> then(&Regex.scan(~r/^\s*alias (Edifice(?:\.[A-Z][A-Za-z0-9]*)+)/m, &1))
      |> Enum.map(fn [_, m] -> Module.concat([m]) end)

    from_recipes =
      for b <- ExPhil.Training.Config.spec_built_backbones(),
          {mod, _embed_key, _params} = ExPhil.Training.Config.backbone_recipe(b),
          do: mod

    Enum.uniq(from_source ++ from_recipes)
  end

  test "every edifice module the dispatcher aliases is a registered edifice architecture" do
    registered =
      Edifice.list_architectures()
      |> Enum.map(&Edifice.module_for/1)
      |> MapSet.new()

    unregistered =
      aliased_modules()
      |> Enum.reject(&MapSet.member?(registered, &1))
      |> Enum.reject(&(&1 in @not_registered))

    assert unregistered == [],
           "dispatcher aliases edifice modules missing from Edifice's registry: #{inspect(unregistered)}"
  end

  test "every aliased edifice module loads" do
    missing = aliased_modules() |> Enum.reject(&Code.ensure_loaded?/1)
    assert missing == [], "dispatcher aliases edifice modules that do not exist: #{inspect(missing)}"
  end
end
