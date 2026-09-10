defmodule ExPhil.Networks.BackboneSpecBuildTest do
  @moduledoc """
  INVARIANTS.md item 3 phase C (middle road): every backbone whose
  construction is a `build:` recipe on its `Config.@backbone_specs` row
  must actually construct through the generic clause. Axon graph
  construction only (no init / no forward), the same bar as edifice's
  registry_integrity_test — a bad module name, a param the module rejects,
  or a `{:ref, key}` pointing at nothing fails HERE instead of at a user's
  training launch.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy.Backbone
  alias ExPhil.Training.Config

  @embed 32
  @opts [window_size: 8, hidden_size: 64, num_layers: 1]

  test "there are spec-built backbones (the recipe path is live)" do
    assert length(Config.spec_built_backbones()) >= 60
  end

  for b <- Config.spec_built_backbones() do
    test "#{b} builds from its spec recipe" do
      b = unquote(b)
      {mod, embed_key, params} = Config.backbone_recipe(b)
      assert Code.ensure_loaded?(mod) and function_exported?(mod, :build, 1), "#{inspect(mod)} has no build/1"
      assert is_atom(embed_key)
      assert Keyword.keyword?(params)

      result = Backbone.build_temporal_backbone(@embed, b, @opts)
      assert is_struct(result, Axon), "#{b}: expected %Axon{}, got #{inspect(result, limit: 3)}"
    end
  end

  test "output rules resolve for every spec-built backbone that has one; the rest raise loudly" do
    for b <- Config.spec_built_backbones() do
      case Config.backbone_output_rule(b) do
        {key, default} when is_atom(key) and is_integer(default) ->
          assert Backbone.temporal_backbone_output_size(b, []) == default
          assert Backbone.temporal_backbone_output_size(b, [{key, 7}]) == 7

        nil ->
          assert_raise ArgumentError, ~r/no output-size rule/, fn ->
            Backbone.temporal_backbone_output_size(b, [])
          end
      end
    end
  end

  test "recipe params resolve in order: seq_len follows window_size unless given" do
    # retnet's recipe: seq_len defaults to {:ref, :window_size}
    {mod, _, params} = Config.backbone_recipe(:retnet)
    assert mod == Edifice.Attention.RetNet
    assert params[:seq_len] == {:ref, :window_size}
    # building with only window_size must not raise (seq_len derived)
    assert %Axon{} = Backbone.build_temporal_backbone(@embed, :retnet, window_size: 12, num_layers: 1)
  end

  test "spec-only keys never leak into training defaults" do
    for b <- Config.valid_backbones() do
      d = Config.backbone_defaults(b)
      refute Keyword.has_key?(d, :build), "#{b}: build recipe leaked into defaults"
      refute Keyword.has_key?(d, :output), "#{b}: output rule leaked into defaults"
    end
  end
end
