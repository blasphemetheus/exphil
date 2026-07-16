defmodule ExPhil.Training.PolicyManifestTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Checkpoint

  @moduletag :tmp_dir

  defp params do
    %{
      "dense_1" => %{
        "kernel" => Nx.iota({4, 8}, type: :f32, backend: Nx.BinaryBackend),
        "bias" => Nx.broadcast(Nx.tensor(0.5, backend: Nx.BinaryBackend), {8})
      }
    }
  end

  defp config do
    %{
      embed_size: 288,
      backbone: :gru,
      temporal: true,
      window_size: 60,
      hidden_size: 256,
      num_layers: 2,
      axis_buckets: 17,
      shoulder_buckets: 5,
      use_prev_action: false
    }
  end

  test "edifice-manifest policies round-trip with config and spec", %{tmp_dir: dir} do
    path = Path.join(dir, "manifest_policy.bin")
    spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config()), external: true)

    :ok = Edifice.Checkpoint.save(params(), path, spec: spec, metadata: %{config: config()})

    assert {:ok, export} = Checkpoint.load_policy(path)
    assert export.config == config()
    assert export.spec["arch"] == "exphil_policy" or export.spec[:arch] == :exphil_policy
    assert Nx.to_flat_list(export.params["dense_1"]["bias"]) == List.duplicate(0.5, 8)
    assert Nx.shape(export.params["dense_1"]["kernel"]) == {4, 8}
  end

  test "legacy term_to_binary policies still load (r1-r10 compat)", %{tmp_dir: dir} do
    path = Path.join(dir, "legacy_policy.bin")
    :ok = File.write(path, :erlang.term_to_binary(%{params: params(), config: config()}))

    assert {:ok, export} = Checkpoint.load_policy(path)
    assert export.config == config()
    assert Nx.shape(export.params["dense_1"]["kernel"]) == {4, 8}
  end

  test "unknown formats are rejected loudly, not misparsed", %{tmp_dir: dir} do
    path = Path.join(dir, "garbage.bin")
    :ok = File.write(path, <<7, 7, 7, "not a policy">>)

    assert {:error, {:unknown_policy_format, ^path}} = Checkpoint.load_policy(path)
  end

  test "embed-size validation still applies to manifest-format loads", %{tmp_dir: dir} do
    path = Path.join(dir, "mismatch_policy.bin")
    spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config()), external: true)
    :ok = Edifice.Checkpoint.save(params(), path, spec: spec, metadata: %{config: config()})

    assert {:error, {:embed_mismatch, _}} =
             Checkpoint.load_policy(path, current_embed_size: 1208, error_on_mismatch: true)
  end
end
