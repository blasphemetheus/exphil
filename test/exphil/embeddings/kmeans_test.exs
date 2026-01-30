defmodule ExPhil.Embeddings.KMeansTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.KMeans

  describe "fit/2" do
    test "fits k clusters to 1D data" do
      # Create data with 3 clear clusters
      data =
        Nx.concatenate([
          Nx.broadcast(0.1, {100}),
          Nx.broadcast(0.5, {100}),
          Nx.broadcast(0.9, {100})
        ])

      centers = KMeans.fit(data, k: 3)

      assert Nx.axis_size(centers, 0) == 3
      # Centers should be sorted
      centers_list = Nx.to_flat_list(centers)
      assert centers_list == Enum.sort(centers_list)
    end

    test "centers are approximately at cluster means" do
      # Generate well-separated clusters
      key = Nx.Random.key(42)
      {c1, key} = Nx.Random.normal(key, 0.1, 0.01, shape: {50})
      {c2, key} = Nx.Random.normal(key, 0.5, 0.01, shape: {50})
      {c3, _key} = Nx.Random.normal(key, 0.9, 0.01, shape: {50})
      data = Nx.concatenate([c1, c2, c3]) |> Nx.clip(0.0, 1.0)

      centers = KMeans.fit(data, k: 3, max_iters: 50)
      centers_list = Nx.to_flat_list(centers)

      # Centers should be close to 0.1, 0.5, 0.9
      assert_in_delta Enum.at(centers_list, 0), 0.1, 0.05
      assert_in_delta Enum.at(centers_list, 1), 0.5, 0.05
      assert_in_delta Enum.at(centers_list, 2), 0.9, 0.05
    end

    test "handles uniform data" do
      # Uniform distribution should still produce k centers
      key = Nx.Random.key(123)
      {data, _key} = Nx.Random.uniform(key, shape: {500})

      centers = KMeans.fit(data, k: 5)

      assert Nx.axis_size(centers, 0) == 5
      # All centers should be in [0, 1]
      assert Nx.to_number(Nx.reduce_min(centers)) >= 0.0
      assert Nx.to_number(Nx.reduce_max(centers)) <= 1.0
    end

    test "respects seed for reproducibility" do
      data = Nx.linspace(0.0, 1.0, n: 100)

      centers1 = KMeans.fit(data, k: 5, seed: 42)
      centers2 = KMeans.fit(data, k: 5, seed: 42)

      assert Nx.to_flat_list(centers1) == Nx.to_flat_list(centers2)
    end
  end

  describe "kmeans_plusplus_init/3" do
    test "initializes k distinct centers" do
      data = Nx.linspace(0.0, 1.0, n: 100)

      centers = KMeans.kmeans_plusplus_init(data, 5, 42)

      assert Nx.axis_size(centers, 0) == 5
      # All centers should be in [0, 1]
      assert Nx.to_number(Nx.reduce_min(centers)) >= 0.0
      assert Nx.to_number(Nx.reduce_max(centers)) <= 1.0
    end

    test "centers are spread out" do
      # With uniform data, K-means++ should spread centers
      data = Nx.linspace(0.0, 1.0, n: 100)

      centers = KMeans.kmeans_plusplus_init(data, 5, 42)
      centers_list = Nx.to_flat_list(centers) |> Enum.sort()

      # Check that centers span most of the range
      range = List.last(centers_list) - List.first(centers_list)
      assert range > 0.5
    end
  end

  describe "assign_to_clusters/2" do
    test "assigns points to nearest center" do
      data = Nx.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
      centers = Nx.tensor([0.0, 0.5, 1.0])

      assignments = KMeans.assign_to_clusters(data, centers)
      assignments_list = Nx.to_flat_list(assignments)

      # 0.0 -> center 0 (dist 0)
      # 0.3 -> center 1 (dist 0.2, closer than center 0 dist 0.3)
      # 0.5 -> center 1 (dist 0)
      # 0.7 -> center 1 (dist 0.2, closer than center 2 dist 0.3)
      # 1.0 -> center 2 (dist 0)
      assert assignments_list == [0, 1, 1, 1, 2]
    end

    test "handles edge cases at midpoints" do
      data = Nx.tensor([0.25, 0.75])
      centers = Nx.tensor([0.0, 0.5, 1.0])

      assignments = KMeans.assign_to_clusters(data, centers)
      assignments_list = Nx.to_flat_list(assignments)

      # 0.25 is equidistant from 0.0 and 0.5 (dist 0.25 each), argmin picks first
      # 0.75 is equidistant from 0.5 and 1.0 (dist 0.25 each), argmin picks first
      assert assignments_list == [0, 1]
    end
  end

  describe "discretize/2" do
    test "discretizes to nearest cluster" do
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

      assert KMeans.discretize(0.1, centers) == 0
      assert KMeans.discretize(0.3, centers) == 1
      assert KMeans.discretize(0.5, centers) == 2
      assert KMeans.discretize(0.6, centers) == 2
      assert KMeans.discretize(0.9, centers) == 4
    end

    test "clamps out-of-range values" do
      centers = Nx.tensor([0.0, 0.5, 1.0])

      assert KMeans.discretize(-0.5, centers) == 0
      assert KMeans.discretize(1.5, centers) == 2
    end
  end

  describe "discretize_batch/2" do
    test "discretizes multiple values" do
      centers = Nx.tensor([0.0, 0.5, 1.0])
      values = Nx.tensor([0.1, 0.4, 0.6, 0.9])

      indices = KMeans.discretize_batch(values, centers)
      indices_list = Nx.to_flat_list(indices)

      assert indices_list == [0, 1, 1, 2]
    end
  end

  describe "undiscretize/2" do
    test "returns cluster center" do
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

      assert KMeans.undiscretize(0, centers) == 0.0
      assert KMeans.undiscretize(2, centers) == 0.5
      assert KMeans.undiscretize(4, centers) == 1.0
    end

    test "round-trip preserves nearest cluster" do
      centers = Nx.tensor([0.0, 0.3, 0.5, 0.7, 1.0])

      # Discretize and undiscretize
      original = 0.35
      idx = KMeans.discretize(original, centers)
      recovered = KMeans.undiscretize(idx, centers)

      # Should be the nearest center
      assert_in_delta recovered, 0.3, 0.01
    end
  end

  describe "k/1" do
    test "returns number of clusters" do
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
      assert KMeans.k(centers) == 5
    end
  end

  describe "default_centers/1" do
    test "generates uniform centers" do
      centers = KMeans.default_centers(5)
      centers_list = Nx.to_flat_list(centers)

      assert length(centers_list) == 5
      assert_in_delta Enum.at(centers_list, 0), 0.0, 0.001
      assert_in_delta Enum.at(centers_list, 2), 0.5, 0.001
      assert_in_delta Enum.at(centers_list, 4), 1.0, 0.001
    end

    test "matches uniform bucket behavior" do
      # Default centers should match uniform bucket discretization
      centers = KMeans.default_centers(17)
      centers_list = Nx.to_flat_list(centers)

      # Should be same as i/16 for i in 0..16
      expected = Enum.map(0..16, fn i -> i / 16 end)

      Enum.zip(centers_list, expected)
      |> Enum.each(fn {actual, exp} ->
        assert_in_delta actual, exp, 0.001
      end)
    end
  end

  describe "save/2 and load/1" do
    @tag :tmp_dir
    test "round-trips centers through file", %{tmp_dir: tmp_dir} do
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
      path = Path.join(tmp_dir, "centers.nx")

      assert :ok = KMeans.save(centers, path)
      assert {:ok, loaded} = KMeans.load(path)

      assert Nx.to_flat_list(loaded) == Nx.to_flat_list(centers)
    end

    test "load returns error for missing file" do
      assert {:error, :enoent} = KMeans.load("/nonexistent/path/centers.nx")
    end

    @tag :tmp_dir
    test "load!/1 returns nil for missing file" do
      assert nil == KMeans.load!("/nonexistent/path/centers.nx")
    end
  end

  describe "load_from_config/2" do
    @tag :tmp_dir
    test "loads centers when path exists in config", %{tmp_dir: tmp_dir} do
      centers = Nx.tensor([0.0, 0.5, 1.0])
      path = Path.join(tmp_dir, "centers.nx")
      KMeans.save(centers, path)

      config = %{kmeans_centers: path}
      assert {:ok, loaded} = KMeans.load_from_config(config)
      assert Nx.to_flat_list(loaded) == Nx.to_flat_list(centers)
    end

    test "returns :not_configured when no path in config" do
      assert :not_configured = KMeans.load_from_config(%{})
      assert :not_configured = KMeans.load_from_config([])
      assert :not_configured = KMeans.load_from_config(%{kmeans_centers: nil})
    end

    test "returns error when path configured but file missing" do
      config = %{kmeans_centers: "/nonexistent/path.nx"}
      assert {:error, :file_not_found, "/nonexistent/path.nx"} = KMeans.load_from_config(config)
    end

    test "supports string keys in map" do
      config = %{"kmeans_centers" => "/nonexistent/path.nx"}
      assert {:error, :file_not_found, "/nonexistent/path.nx"} = KMeans.load_from_config(config)
    end

    test "supports keyword list config" do
      config = [kmeans_centers: "/nonexistent/path.nx"]
      assert {:error, :file_not_found, "/nonexistent/path.nx"} = KMeans.load_from_config(config)
    end
  end

  describe "load_from_config!/2" do
    @tag :tmp_dir
    test "returns tensor when found", %{tmp_dir: tmp_dir} do
      centers = Nx.tensor([0.0, 0.5, 1.0])
      path = Path.join(tmp_dir, "centers.nx")
      KMeans.save(centers, path)

      config = %{kmeans_centers: path}
      loaded = KMeans.load_from_config!(config)
      assert Nx.to_flat_list(loaded) == Nx.to_flat_list(centers)
    end

    test "returns nil when not configured" do
      assert nil == KMeans.load_from_config!(%{})
    end

    test "returns nil when file missing" do
      config = %{kmeans_centers: "/nonexistent/path.nx"}
      assert nil == KMeans.load_from_config!(config)
    end
  end

  describe "info_string/1" do
    test "returns summary of centers" do
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
      info = KMeans.info_string(centers)

      assert info =~ "5 clusters"
      assert info =~ "0.0"
      assert info =~ "1.0"
    end

    test "formats range correctly" do
      centers = Nx.tensor([0.003, 0.5, 0.996])
      info = KMeans.info_string(centers)

      assert info == "3 clusters, range [0.003, 0.996]"
    end
  end
end
