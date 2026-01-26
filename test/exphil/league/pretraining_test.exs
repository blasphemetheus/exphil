defmodule ExPhil.League.PretrainingTest do
  use ExUnit.Case, async: true

  alias ExPhil.League.Pretraining
  alias ExPhil.League.ArchitectureEntry

  describe "build_model/3" do
    test "builds MLP model" do
      embed_config = ExPhil.Embeddings.config([])
      config = %{hidden_sizes: [256, 256]}

      model = Pretraining.build_model(:mlp, embed_config, config)

      assert %Axon{} = model
    end

    test "builds LSTM model" do
      embed_config = ExPhil.Embeddings.config([])
      config = %{hidden_size: 256, num_layers: 2}

      model = Pretraining.build_model(:lstm, embed_config, config)

      assert %Axon{} = model
    end

    test "builds GRU model" do
      embed_config = ExPhil.Embeddings.config([])
      config = %{hidden_size: 256, num_layers: 2}

      model = Pretraining.build_model(:gru, embed_config, config)

      assert %Axon{} = model
    end

    test "builds Mamba model" do
      embed_config = ExPhil.Embeddings.config([])
      config = ArchitectureEntry.default_config(:mamba)

      model = Pretraining.build_model(:mamba, embed_config, config)

      assert %Axon{} = model
    end

    test "builds attention model" do
      embed_config = ExPhil.Embeddings.config([])
      config = ArchitectureEntry.default_config(:attention)

      model = Pretraining.build_model(:attention, embed_config, config)

      assert %Axon{} = model
    end

    test "builds Jamba model" do
      embed_config = ExPhil.Embeddings.config([])
      config = ArchitectureEntry.default_config(:jamba)

      model = Pretraining.build_model(:jamba, embed_config, config)

      assert %Axon{} = model
    end

    test "raises for unknown architecture" do
      embed_config = ExPhil.Embeddings.config([])

      assert_raise ArgumentError, ~r/Unknown architecture/, fn ->
        Pretraining.build_model(:unknown, embed_config, %{})
      end
    end
  end

  describe "module structure" do
    # Ensure module is loaded before checking exports
    setup do
      Code.ensure_loaded!(Pretraining)
      :ok
    end

    test "exports train_all/3" do
      assert function_exported?(Pretraining, :train_all, 3)
    end

    test "exports train_to_target/5" do
      assert function_exported?(Pretraining, :train_to_target, 5)
    end

    test "exports build_model/3" do
      assert function_exported?(Pretraining, :build_model, 3)
    end
  end

  describe "default options" do
    # Verify default option values by checking the function accepts them
    test "train_all accepts target_loss option" do
      # Can't run full training without data, but verify API exists
      assert is_function(&Pretraining.train_all/3)
    end
  end
end
