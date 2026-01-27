#!/usr/bin/env elixir
# Inspect embedding cache files to see their contents/metadata

alias ExPhil.Training.EmbeddingCache

IO.puts("\n=== Embedding Cache Contents ===\n")

cache_dir = System.get_env("EXPHIL_CACHE_DIR") || "cache/embeddings"

if File.exists?(cache_dir) do
  cache_dir
  |> File.ls!()
  |> Enum.filter(&String.ends_with?(&1, ".emb"))
  |> Enum.each(fn filename ->
    path = Path.join(cache_dir, filename)
    stat = File.stat!(path)
    key = String.replace_suffix(filename, ".emb", "")

    IO.puts("📦 #{filename}")
    IO.puts("   Size: #{Float.round(stat.size / 1_000_000, 2)} MB")

    # Try to peek at the structure
    case File.read(path) do
      {:ok, binary} ->
        data = :erlang.binary_to_term(binary)

        case data do
          {:stacked_frame_embeddings, %{shape: shape, type: type}} ->
            IO.puts("   Type: Stacked frame embeddings (new format)")
            IO.puts("   Shape: #{inspect(shape)}")
            IO.puts("   Dtype: #{inspect(type)}")
            IO.puts("   Frames: #{elem(shape, 0)}")

          {:frame_embeddings, %{shape: shape, type: type, size: size}} ->
            IO.puts("   Type: Frame embeddings (legacy format)")
            IO.puts("   Shape: #{inspect(shape)}")
            IO.puts("   Dtype: #{inspect(type)}")
            IO.puts("   Count: #{size}")

          {:sequence_embeddings, %{shape: shape, type: type, size: size, embed_config: config}} ->
            IO.puts("   Type: Sequence embeddings")
            IO.puts("   Shape: #{inspect(shape)} (window_size × embed_dim)")
            IO.puts("   Dtype: #{inspect(type)}")
            IO.puts("   Count: #{size} sequences")
            if config do
              IO.puts("   Embed config:")
              config
              |> Map.from_struct()
              |> Enum.filter(fn {_k, v} -> v != nil end)
              |> Enum.each(fn {k, v} ->
                IO.puts("     #{k}: #{inspect(v)}")
              end)
            end

          {:raw, _} ->
            IO.puts("   Type: Raw data (unknown format)")

          other ->
            IO.puts("   Type: Unknown - #{inspect(elem(other, 0))}")
        end

      {:error, reason} ->
        IO.puts("   Error reading: #{inspect(reason)}")
    end

    IO.puts("")
  end)
else
  IO.puts("Cache directory not found: #{cache_dir}")
end
