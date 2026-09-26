# CPU-only inventory using existing compiled modules. No Mix compilation.
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output
defmodule MewtwoHeader do
  # Peppi 2.1.2 io/slippi/de.rs: payload-size event follows the 15-byte
  # UBJSON header; GameStart player records start at event offset 0x65,
  # stride 36. CSS Mewtwo = 10. Unknown formats go to the full parser.
  def possible?(path) do
    File.open!(path, [:read, :binary], fn io ->
      data = IO.binread(io, 1024)
      case data do
        <<"{U", 3, "raw[$U#l", _size::32, 0x35, count, _::binary>> when rem(count, 3) == 1 ->
          start = 15 + 1 + count
          if byte_size(data) > start + 0x65 + 3 * 36 + 1 and :binary.at(data, start) == 0x36 do
            Enum.any?(0..3, fn port ->
              offset = start + 0x65 + port * 36
              :binary.at(data, offset) == 10 and :binary.at(data, offset + 1) in [0, 1]
            end)
          else
            true
          end
        _ -> true
      end
    end)
  rescue
    _ -> true
  end
end
[out | roots] = System.argv()
File.mkdir_p!(out)
Output.banner("Mewtwo replay inventory")
files = roots |> Enum.flat_map(&Path.wildcard(Path.join(&1, "**/*.slp"))) |> Enum.uniq() |> Enum.sort()
Output.puts("Inspecting #{length(files)} replay paths on CPU")
File.open!(Path.join(out, "metadata.jsonl"), [:write], fn io ->
  files
  |> Task.async_stream(fn path ->
    case if(MewtwoHeader.possible?(path), do: Peppi.metadata(path), else: :skip) do
      :skip -> nil
      {:ok, meta} ->
        if Enum.any?(meta.players, &(String.downcase(&1.character_name || "") == "mewtwo")) do
          %{path: Path.expand(path), stage: meta.stage, frames: meta.duration_frames,
            started_at: meta.started_at, random_seed: meta.random_seed,
            players: Enum.map(meta.players, &Map.from_struct/1),
            sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower)}
        end
      {:error, reason} -> %{path: path, error: inspect(reason)}
    end
  end, max_concurrency: 8, ordered: false, timeout: :infinity)
  |> Enum.with_index(1)
  |> Enum.each(fn {{:ok, row}, i} ->
    if row, do: IO.binwrite(io, Jason.encode!(row) <> "\n")
    if rem(i, 1000) == 0, do: Output.puts("Inspected #{i}/#{length(files)}")
  end)
end)
Output.success("Inventory complete: #{out}/metadata.jsonl")
