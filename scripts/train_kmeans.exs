#!/usr/bin/env elixir
# Train K-means cluster centers from replay stick data
#
# Usage:
#   mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx
#
# Options:
#   --replays DIR    Directory containing .slp replay files (default: ./replays)
#   --k NUM          Number of clusters (default: 21)
#   --max-files NUM  Maximum replay files to process (default: all)
#   --output PATH    Output path for centers (default: priv/kmeans_centers.nx)
#   --max-iters NUM  Maximum K-means iterations (default: 100)

defmodule TrainKMeans do
  alias ExPhil.Data.Peppi
  alias ExPhil.Embeddings.KMeans

  def run(args) do
    # Parse args
    replays_dir = get_arg_value(args, "--replays") || "./replays"
    k = String.to_integer(get_arg_value(args, "--k") || "21")
    max_files = case get_arg_value(args, "--max-files") do
      nil -> nil
      v -> String.to_integer(v)
    end
    output_path = get_arg_value(args, "--output") || "priv/kmeans_centers.nx"
    max_iters = String.to_integer(get_arg_value(args, "--max-iters") || "100")

    IO.puts("""
    \n========================================
      K-Means Stick Discretization Training
    ========================================
    Replays:    #{replays_dir}
    Clusters:   #{k}
    Max files:  #{max_files || "all"}
    Output:     #{output_path}
    Max iters:  #{max_iters}
    """)

    # Step 1: Find replay files
    IO.puts("\n[1/4] Finding replay files...")
    all_files = Path.wildcard(Path.join(replays_dir, "**/*.slp"))
    replay_files = if max_files, do: Enum.take(all_files, max_files), else: all_files

    IO.puts("  Found #{length(replay_files)} replay files")

    if length(replay_files) == 0 do
      IO.puts("  ERROR: No replay files found in #{replays_dir}")
      System.halt(1)
    end

    # Step 2: Extract stick positions
    IO.puts("\n[2/4] Extracting stick positions from replays...")

    total = length(replay_files)
    stick_data = replay_files
    |> Enum.with_index()
    |> Enum.flat_map(fn {file, idx} ->
      if rem(idx, 100) == 0 do
        IO.write("\r  Processing file #{idx + 1}/#{total}...")
      end

      extract_sticks_from_file(file)
    end)

    IO.puts("\n  Extracted #{length(stick_data)} stick positions")

    if length(stick_data) < k * 10 do
      IO.puts("  WARNING: Not enough data points (#{length(stick_data)}) for #{k} clusters")
      IO.puts("  Need at least #{k * 10} points for reliable clustering")
    end

    # Step 3: Run K-means
    IO.puts("\n[3/4] Running K-means clustering (k=#{k}, max_iters=#{max_iters})...")

    # Convert to tensor
    stick_tensor = Nx.tensor(stick_data, type: :f32)

    # Fit K-means
    centers = KMeans.fit(stick_tensor, k: k, max_iters: max_iters)

    IO.puts("  Cluster centers:")
    centers_list = Nx.to_flat_list(centers)
    Enum.with_index(centers_list)
    |> Enum.each(fn {center, idx} ->
      IO.puts("    #{idx}: #{Float.round(center, 4)}")
    end)

    # Step 4: Save centers
    IO.puts("\n[4/4] Saving cluster centers to #{output_path}...")

    # Ensure directory exists
    Path.dirname(output_path) |> File.mkdir_p!()

    case KMeans.save(centers, output_path) do
      :ok ->
        IO.puts("  ✓ Saved #{k} cluster centers")

        # Also save as human-readable JSON
        json_path = String.replace(output_path, ".nx", ".json")
        json = Jason.encode!(%{
          k: k,
          centers: centers_list,
          source_files: length(replay_files),
          source_samples: length(stick_data),
          timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
        }, pretty: true)
        File.write!(json_path, json)
        IO.puts("  ✓ Saved metadata to #{json_path}")

      {:error, reason} ->
        IO.puts("  ✗ Error saving: #{inspect(reason)}")
        System.halt(1)
    end

    # Show comparison with uniform buckets
    IO.puts("\n========================================")
    IO.puts("  Comparison with Uniform Buckets")
    IO.puts("========================================")
    IO.puts("\nUniform 17 buckets (0-16):")
    uniform = Enum.map(0..16, fn i -> Float.round(i / 16, 4) end)
    IO.puts("  #{inspect(uniform)}")

    IO.puts("\nK-means #{k} clusters:")
    IO.puts("  #{inspect(Enum.map(centers_list, &Float.round(&1, 4)))}")

    IO.puts("\nDone! Use --kmeans-centers #{output_path} to use these centers for training.")
  end

  defp get_arg_value(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> nil
      idx -> Enum.at(args, idx + 1)
    end
  end

  defp extract_sticks_from_file(file) do
    case Peppi.parse(file) do
      {:ok, replay} ->
        frames = Map.get(replay, :frames, [])
        Enum.flat_map(frames, fn frame ->
          p1_sticks = extract_sticks(frame[:p1_controller])
          p2_sticks = extract_sticks(frame[:p2_controller])
          p1_sticks ++ p2_sticks
        end)

      {:error, _reason} ->
        []
    end
  end

  defp extract_sticks(nil), do: []
  defp extract_sticks(%{main_stick: main, c_stick: c}) do
    main_values = case main do
      %{x: x, y: y} when is_number(x) and is_number(y) -> [x, y]
      _ -> []
    end
    c_values = case c do
      %{x: x, y: y} when is_number(x) and is_number(y) -> [x, y]
      _ -> []
    end
    main_values ++ c_values
  end
  defp extract_sticks(_), do: []
end

TrainKMeans.run(System.argv())
