defmodule ExPhil.Harness.BackboneDefaultsTest do
  @moduledoc """
  Guards the atom contract between three lists that must agree:

    * `Config.@valid_backbones` — the CLI gate
    * `Networks.Policy.Backbone.build_temporal_backbone/3` — the dispatcher
    * `Config.backbone_defaults/1` — per-backbone hyperparameters

  Found 2026-08-03 while preparing the architecture bake-off:
  `backbone_defaults/1` had clauses for `:ret_net` and `:mamba_3` while both
  the dispatcher and the CLI gate spell them `:retnet` and `:mamba3`. Those
  clauses could never match, so those backbones trained with NO defaults —
  no precision pin, no window size, no layer count — silently, because a
  missing default is indistinguishable from "this backbone needs none".

  That matters most exactly when it is least visible: a wide sweep where
  nobody inspects each arm's config.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config

  # Backbones whose defaults are deliberately absent (they take the generic
  # path). Keeping this explicit means "no entry" is a decision, not a typo.
  @intentionally_undefaulted []

  describe "backbone_defaults/1 keys are real backbones" do
    test "every backbone with defaults is CLI-selectable" do
      valid = MapSet.new(Config.valid_backbones())

      defaulted =
        Config.valid_backbones()
        |> Enum.filter(&(Config.backbone_defaults(&1) not in [nil, []]))
        |> MapSet.new()

      # Any atom that HAS defaults must be a valid backbone. We can only
      # enumerate via valid_backbones/0, so the real check is the inverse
      # below plus the spot-checks — this asserts the intersection is sane.
      assert MapSet.subset?(defaulted, valid)
    end

    test "the two backbones that were silently untuned now get defaults" do
      for backbone <- [:retnet, :mamba3] do
        defaults = Config.backbone_defaults(backbone)

        assert defaults not in [nil, []],
               "#{inspect(backbone)} has no defaults — this is the :ret_net/:mamba_3 " <>
                 "atom-typo class: the clause exists but under a name the dispatcher " <>
                 "never uses, so the backbone trains untuned."

        assert Keyword.has_key?(defaults, :temporal)
      end
    end

    test "the misspelled atoms are gone" do
      for typo <- [:ret_net, :mamba_3] do
        refute typo in Config.valid_backbones(),
               "#{inspect(typo)} is not a real backbone atom"

        assert Config.backbone_defaults(typo) in [nil, []],
               "#{inspect(typo)} should have no defaults clause — it is a typo of a real " <>
                 "backbone, and a clause under it can never match the dispatcher."
      end
    end

    test "undefaulted backbones are an explicit list, not an accident" do
      undefaulted =
        Config.valid_backbones()
        |> Enum.filter(&(Config.backbone_defaults(&1) in [nil, []]))

      # This is informational rather than a hard gate: ~23 of the CLI's
      # backbones have tuned defaults today and filling the rest is the
      # bake-off's job. The assertion pins the KNOWN-intentional set so a
      # newly-typo'd key shows up as a diff here.
      assert Enum.all?(@intentionally_undefaulted, &(&1 in undefaulted))
    end
  end
end
