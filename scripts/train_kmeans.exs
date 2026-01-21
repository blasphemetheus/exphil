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

alias ExPhil.Training.Output

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

    Output.banner("K-Means Stick Discretization Training")
    Output.config([
      {"Replays", replays_dir},
      {"Clusters", k},
      {"Max files", max_files || "all"},
      {"Output", output_path},
      {"Max iters", max_iters}
    ])

    # Step 1: Find replay files
    Output.step(1, 4, "Finding replay files")
    all_files = Path.wildcard(Path.join(replays_dir, "**/*.slp"))
    replay_files = if max_files, do: Enum.take(all_files, max_files), else: all_files

    Output.puts("  Found #{length(replay_files)} replay files")

    if length(replay_files) == 0 do
      Output.error("No replay files found in #{replays_dir}")
      System.halt(1)
    end

    # Step 2: Extract stick positions
    Output.step(2, 4, "Extracting stick positions from replays")

    total = length(replay_files)
    stick_data = replay_files
    |> Enum.with_index()
    |> Enum.flat_map(fn {file, idx} ->
      if rem(idx, 100) == 0 do
        Output.progress_bar(idx + 1, total, label: "Processing")
      end

      extract_sticks_from_file(file)
    end)
    Output.progress_done()

    Output.puts("  Extracted #{length(stick_data)} stick positions")

    if length(stick_data) < k * 10 do
      Output.warning("Not enough data points (#{length(stick_data)}) for #{k} clusters")
      Output.puts("  Need at least #{k * 10} points for reliable clustering")
    end

    # Step 3: Run K-means
    Output.step(3, 4, "Running K-means clustering (k=#{k}, max_iters=#{max_iters})")

    # Convert to tensor
    stick_tensor = Nx.tensor(stick_data, type: :f32)

    # Fit K-means
    centers = KMeans.fit(stick_tensor, k: k, max_iters: max_iters)

    Output.puts("  Cluster centers:")
    centers_list = Nx.to_flat_list(centers)
    Enum.with_index(centers_list)
    |> Enum.each(fn {center, idx} ->
      Output.puts("    #{idx}: #{Float.round(center, 4)}")
    end)

    # Step 4: Save centers
    Output.step(4, 4, "Saving cluster centers to #{output_path}")

    # Ensure directory exists
    Path.dirname(output_path) |> File.mkdir_p!()

    case KMeans.save(centers, output_path) do
      :ok ->
        Output.success("Saved #{k} cluster centers")

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
        Output.success("Saved metadata to #{json_path}")

      {:error, reason} ->
        Output.error("Error saving: #{inspect(reason)}")
        System.halt(1)
    end

    # Show comparison with uniform buckets
    Output.divider()
    Output.section("Comparison with Uniform Buckets")
    Output.puts("Uniform 17 buckets (0-16):")
    uniform = Enum.map(0..16, fn i -> Float.round(i / 16, 4) end)
    Output.puts("  #{inspect(uniform)}")

    Output.puts("")
    Output.puts("K-means #{k} clusters:")
    Output.puts("  #{inspect(Enum.map(centers_list, &Float.round(&1, 4)))}")

    Output.puts("")
    Output.success("Done! Use --kmeans-centers #{output_path} to use these centers for training.")
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
