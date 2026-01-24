defmodule ExPhil.League.ArchitectureEntryTest do
  use ExUnit.Case, async: true

  alias ExPhil.League.ArchitectureEntry

  describe "new/1" do
    test "creates entry with required fields" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :mamba_mewtwo,
        architecture: :mamba
      )

      assert entry.id == :mamba_mewtwo
      assert entry.architecture == :mamba
      assert entry.character == :mewtwo  # default
      assert entry.generation == 0
      assert entry.elo == 1000.0
      assert entry.lineage == []
      assert entry.stats.wins == 0
      assert entry.stats.losses == 0
      assert entry.stats.draws == 0
    end

    test "accepts custom character" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :lstm_ganon,
        architecture: :lstm,
        character: :ganondorf
      )

      assert entry.character == :ganondorf
    end

    test "accepts custom config" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :mlp_mewtwo,
        architecture: :mlp,
        config: %{hidden_sizes: [512, 512]}
      )

      assert entry.config.hidden_sizes == [512, 512]
    end

    test "accepts custom initial Elo" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :attention_mewtwo,
        architecture: :attention,
        elo: 1500
      )

      assert entry.elo == 1500.0
    end

    test "rejects unsupported architecture" do
      {:error, {:unsupported_architecture, :unknown, _}} =
        ArchitectureEntry.new(id: :bad, architecture: :unknown)
    end
  end

  describe "new!/1" do
    test "returns entry on success" do
      entry = ArchitectureEntry.new!(id: :mamba_mewtwo, architecture: :mamba)
      assert entry.id == :mamba_mewtwo
    end

    test "raises on failure" do
      assert_raise ArgumentError, fn ->
        ArchitectureEntry.new!(id: :bad, architecture: :unknown)
      end
    end
  end

  describe "update_from_training/2" do
    test "increments generation and adds to lineage" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.update_from_training(entry, %{layer1: %{}})

      assert updated.generation == 1
      assert updated.params == %{layer1: %{}}
      assert updated.lineage == ["mamba_mewtwo_v0"]
    end

    test "accumulates lineage across generations" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      entry = ArchitectureEntry.update_from_training(entry, %{v: 1})
      entry = ArchitectureEntry.update_from_training(entry, %{v: 2})
      entry = ArchitectureEntry.update_from_training(entry, %{v: 3})

      assert entry.generation == 3
      assert length(entry.lineage) == 3
      assert entry.lineage == ["mamba_mewtwo_v0", "mamba_mewtwo_v1", "mamba_mewtwo_v2"]
    end
  end

  describe "update_elo/2" do
    test "updates Elo rating" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.update_elo(entry, 1050.5)

      assert updated.elo == 1050.5
    end

    test "converts integer to float" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.update_elo(entry, 1100)

      assert updated.elo == 1100.0
    end
  end

  describe "record_result/3" do
    test "records win" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.record_result(entry, :win, 4500)

      assert updated.stats.wins == 1
      assert updated.stats.losses == 0
      assert updated.stats.draws == 0
      assert updated.stats.total_frames == 4500
    end

    test "records loss" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.record_result(entry, :loss, 3000)

      assert updated.stats.wins == 0
      assert updated.stats.losses == 1
      assert updated.stats.draws == 0
    end

    test "records draw" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      updated = ArchitectureEntry.record_result(entry, :draw)

      assert updated.stats.draws == 1
    end

    test "accumulates stats" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      entry = ArchitectureEntry.record_result(entry, :win, 4500)
      entry = ArchitectureEntry.record_result(entry, :win, 4000)
      entry = ArchitectureEntry.record_result(entry, :loss, 5000)

      assert entry.stats.wins == 2
      assert entry.stats.losses == 1
      assert entry.stats.total_frames == 13500
    end
  end

  describe "win_rate/1" do
    test "returns 0.5 with no games" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      assert ArchitectureEntry.win_rate(entry) == 0.5
    end

    test "calculates correct win rate" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      entry = ArchitectureEntry.record_result(entry, :win)
      entry = ArchitectureEntry.record_result(entry, :win)
      entry = ArchitectureEntry.record_result(entry, :loss)
      entry = ArchitectureEntry.record_result(entry, :draw)

      # 2 wins out of 4 games
      assert_in_delta ArchitectureEntry.win_rate(entry), 0.5, 0.001
    end
  end

  describe "games_played/1" do
    test "counts all games" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      entry = ArchitectureEntry.record_result(entry, :win)
      entry = ArchitectureEntry.record_result(entry, :loss)
      entry = ArchitectureEntry.record_result(entry, :draw)

      assert ArchitectureEntry.games_played(entry) == 3
    end
  end

  describe "trained?/1" do
    test "returns false with empty params" do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      refute ArchitectureEntry.trained?(entry)
    end

    test "returns true with params" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :mamba_mewtwo,
        architecture: :mamba,
        params: %{layer1: %{weight: :tensor}}
      )

      assert ArchitectureEntry.trained?(entry)
    end
  end

  describe "default_config/1" do
    test "returns config for mlp" do
      config = ArchitectureEntry.default_config(:mlp)

      assert config.hidden_sizes == [256, 256]
      assert config.dropout == 0.1
    end

    test "returns config for lstm" do
      config = ArchitectureEntry.default_config(:lstm)

      assert config.hidden_size == 256
      assert config.num_layers == 2
      assert config.window_size == 30
    end

    test "returns config for mamba" do
      config = ArchitectureEntry.default_config(:mamba)

      assert config.hidden_size == 256
      assert config.state_size == 16
      assert config.expand_factor == 2
    end

    test "returns config for attention" do
      config = ArchitectureEntry.default_config(:attention)

      assert config.num_heads == 4
      assert config.head_dim == 64
    end

    test "returns config for jamba" do
      config = ArchitectureEntry.default_config(:jamba)

      assert config.attention_every == 3
      assert config.num_layers == 4
    end
  end

  describe "supported_architectures/0" do
    test "returns all supported types" do
      archs = ArchitectureEntry.supported_architectures()

      assert :mlp in archs
      assert :lstm in archs
      assert :gru in archs
      assert :mamba in archs
      assert :attention in archs
      assert :jamba in archs
    end
  end

  describe "to_metadata/1 and from_metadata/1" do
    test "round-trips metadata" do
      {:ok, entry} = ArchitectureEntry.new(
        id: :mamba_mewtwo,
        architecture: :mamba,
        config: %{hidden_size: 512}
      )

      entry = entry
      |> ArchitectureEntry.update_elo(1200)
      |> ArchitectureEntry.record_result(:win, 4500)
      |> ArchitectureEntry.update_from_training(%{})

      metadata = ArchitectureEntry.to_metadata(entry)
      {:ok, restored} = ArchitectureEntry.from_metadata(metadata)

      assert restored.id == entry.id
      assert restored.architecture == entry.architecture
      assert restored.elo == entry.elo
      assert restored.generation == entry.generation
      assert restored.stats.wins == entry.stats.wins
    end

    test "handles string keys from JSON" do
      metadata = %{
        "id" => "mamba_mewtwo",
        "architecture" => "mamba",
        "character" => "mewtwo",
        "generation" => 5,
        "elo" => 1450,
        "config" => %{"hidden_size" => 256},
        "lineage" => ["v0", "v1"],
        "stats" => %{"wins" => 10, "losses" => 5, "draws" => 1, "total_frames" => 50000}
      }

      {:ok, entry} = ArchitectureEntry.from_metadata(metadata)

      assert entry.id == :mamba_mewtwo
      assert entry.architecture == :mamba
      assert entry.generation == 5
      assert entry.elo == 1450.0
    end
  end
end
