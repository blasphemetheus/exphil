defmodule ExPhil.Training.RegistryTest do
  use ExUnit.Case, async: false

  alias ExPhil.Training.Registry

  # Use a test-specific registry file
  @test_registry "test/fixtures/test_registry.json"

  setup do
    # Clean up test registry before each test
    File.rm(@test_registry)

    # Temporarily override the registry path
    original_path = Application.get_env(:exphil, :registry_path)
    Application.put_env(:exphil, :registry_path, @test_registry)

    on_exit(fn ->
      File.rm(@test_registry)
      if original_path do
        Application.put_env(:exphil, :registry_path, original_path)
      else
        Application.delete_env(:exphil, :registry_path)
      end
    end)

    :ok
  end

  describe "register/1" do
    test "registers a model with required fields" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{epochs: 10, backbone: :mamba}
      })

      assert entry.checkpoint_path == "checkpoints/test.axon"
      assert entry.training_config.epochs == 10
      assert entry.training_config.backbone == :mamba
      assert is_binary(entry.id)
      assert is_binary(entry.name)
      assert entry.tags == []
      assert entry.parent_id == nil
    end

    test "registers with optional fields" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        policy_path: "checkpoints/test_policy.bin",
        config_path: "checkpoints/test_config.json",
        training_config: %{epochs: 10},
        metrics: %{final_loss: 1.234},
        tags: ["mewtwo", "production"],
        name: "custom_name"
      })

      assert entry.policy_path == "checkpoints/test_policy.bin"
      assert entry.config_path == "checkpoints/test_config.json"
      assert entry.metrics.final_loss == 1.234
      assert entry.tags == ["mewtwo", "production"]
      assert entry.name == "custom_name"
    end

    test "returns error without checkpoint_path" do
      assert {:error, :missing_required_field} = Registry.register(%{
        training_config: %{epochs: 10}
      })
    end

    test "returns error without training_config" do
      assert {:error, :missing_required_field} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon"
      })
    end

    test "generates unique IDs" do
      {:ok, entry1} = Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{}
      })

      {:ok, entry2} = Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{}
      })

      assert entry1.id != entry2.id
    end

    test "sanitizes config by removing functions" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{
          epochs: 10,
          callback: fn -> :ok end
        }
      })

      assert entry.training_config[:epochs] == 10
      refute Map.has_key?(entry.training_config, :callback)
    end
  end

  describe "list/1" do
    test "returns empty list when no models" do
      assert {:ok, []} = Registry.list()
    end

    test "returns all registered models" do
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{}
      })
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{}
      })

      {:ok, models} = Registry.list()
      assert length(models) == 2
    end

    test "filters by tags" do
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{},
        tags: ["mewtwo"]
      })
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{},
        tags: ["ganondorf"]
      })

      {:ok, models} = Registry.list(tags: ["mewtwo"])
      assert length(models) == 1
      assert hd(models).tags == ["mewtwo"]
    end

    test "filters by backbone" do
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{backbone: :mamba}
      })
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{backbone: :mlp}
      })

      {:ok, models} = Registry.list(backbone: :mamba)
      assert length(models) == 1
      # JSON stores as string, so convert for comparison
      assert to_string(hd(models).training_config.backbone) == "mamba"
    end

    test "limits results" do
      for i <- 1..5 do
        Registry.register(%{
          checkpoint_path: "checkpoints/test#{i}.axon",
          training_config: %{}
        })
      end

      {:ok, models} = Registry.list(limit: 3)
      assert length(models) == 3
    end

    test "sorts by created_at descending by default" do
      {:ok, first} = Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{}
      })
      Process.sleep(10)  # Ensure different timestamp
      {:ok, second} = Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{}
      })

      {:ok, models} = Registry.list()
      assert hd(models).id == second.id
    end
  end

  describe "get/1" do
    test "returns model by ID" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{}
      })

      {:ok, found} = Registry.get(entry.id)
      assert found.id == entry.id
    end

    test "returns model by name" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{},
        name: "my_model"
      })

      {:ok, found} = Registry.get("my_model")
      assert found.name == "my_model"
    end

    test "returns error for unknown model" do
      assert {:error, :not_found} = Registry.get("nonexistent")
    end
  end

  describe "tag/2 and untag/2" do
    test "adds tags to a model" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{}
      })

      :ok = Registry.tag(entry.id, ["production", "v1"])

      {:ok, updated} = Registry.get(entry.id)
      assert "production" in updated.tags
      assert "v1" in updated.tags
    end

    test "removes tags from a model" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{},
        tags: ["a", "b", "c"]
      })

      :ok = Registry.untag(entry.id, ["b"])

      {:ok, updated} = Registry.get(entry.id)
      assert updated.tags == ["a", "c"]
    end

    test "deduplicates tags" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{},
        tags: ["a"]
      })

      :ok = Registry.tag(entry.id, ["a", "b"])

      {:ok, updated} = Registry.get(entry.id)
      assert Enum.sort(updated.tags) == ["a", "b"]
    end
  end

  describe "delete/2" do
    test "removes model from registry" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{}
      })

      :ok = Registry.delete(entry.id)

      assert {:error, :not_found} = Registry.get(entry.id)
    end

    test "returns error for unknown model" do
      assert {:error, :not_found} = Registry.delete("nonexistent")
    end
  end

  describe "best/1" do
    test "returns model with lowest loss" do
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/high_loss.axon",
        training_config: %{},
        metrics: %{final_loss: 5.0}
      })
      {:ok, best_entry} = Registry.register(%{
        checkpoint_path: "checkpoints/low_loss.axon",
        training_config: %{},
        metrics: %{final_loss: 1.0}
      })

      {:ok, best} = Registry.best()
      assert best.id == best_entry.id
    end

    test "returns error when no models have metric" do
      {:ok, _} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{},
        metrics: %{}
      })

      assert {:error, :no_models_with_metric} = Registry.best()
    end
  end

  describe "exists?/1" do
    test "returns true for existing model" do
      {:ok, entry} = Registry.register(%{
        checkpoint_path: "checkpoints/test.axon",
        training_config: %{}
      })

      assert Registry.exists?(entry.id)
      assert Registry.exists?(entry.name)
    end

    test "returns false for nonexistent model" do
      refute Registry.exists?("nonexistent")
    end
  end

  describe "count/1" do
    test "returns total count" do
      for i <- 1..3 do
        Registry.register(%{
          checkpoint_path: "checkpoints/test#{i}.axon",
          training_config: %{}
        })
      end

      assert {:ok, 3} = Registry.count()
    end

    test "respects filters" do
      Registry.register(%{
        checkpoint_path: "checkpoints/test1.axon",
        training_config: %{},
        tags: ["production"]
      })
      Registry.register(%{
        checkpoint_path: "checkpoints/test2.axon",
        training_config: %{},
        tags: ["dev"]
      })

      assert {:ok, 1} = Registry.count(tags: ["production"])
    end
  end

  describe "lineage/1" do
    test "returns model lineage chain" do
      {:ok, parent} = Registry.register(%{
        checkpoint_path: "checkpoints/parent.axon",
        training_config: %{}
      })

      {:ok, child} = Registry.register(%{
        checkpoint_path: "checkpoints/child.axon",
        training_config: %{},
        parent_id: parent.id
      })

      {:ok, grandchild} = Registry.register(%{
        checkpoint_path: "checkpoints/grandchild.axon",
        training_config: %{},
        parent_id: child.id
      })

      {:ok, lineage} = Registry.lineage(grandchild.id)

      assert length(lineage) == 3
      assert Enum.at(lineage, 0).id == parent.id
      assert Enum.at(lineage, 1).id == child.id
      assert Enum.at(lineage, 2).id == grandchild.id
    end
  end
end
