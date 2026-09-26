defmodule ExPhil.Training.CheckpointConfigBackendTest do
  use ExUnit.Case, async: false
  alias ExPhil.Training.Imitation

  @tag :gpu
  test "saved configuration contains portable tensors, including nested values" do
    clients = Application.fetch_env!(:exla, :clients)

    Application.put_env(
      :exla,
      :clients,
      Keyword.put(clients, :cuda, platform: :cuda, memory_fraction: 0.05)
    )

    on_exit(fn -> Application.put_env(:exla, :clients, clients) end)

    path =
      Path.join(System.tmp_dir!(), "config_backend_#{System.unique_integer([:positive])}.axon")

    on_exit(fn -> File.rm(path) end)
    trainer = Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false, precision: :f32)
    weights = Nx.tensor([1.0, 2.0], backend: {EXLA.Backend, client: :cuda})

    config =
      Map.merge(trainer.config, %{button_pos_weight: weights, nested: [pair: {weights, :kept}]})

    assert :ok = Imitation.save_checkpoint(%{trainer | config: config}, path)
    saved = path |> File.read!() |> :erlang.binary_to_term()
    assert %Nx.BinaryBackend{} = saved.config.button_pos_weight.data
    assert %Nx.BinaryBackend{} = elem(saved.config.nested[:pair], 0).data
    assert Nx.to_flat_list(saved.config.button_pos_weight) == [1.0, 2.0]
    assert elem(saved.config.nested[:pair], 1) == :kept
  end
end
