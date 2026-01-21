defmodule Mix.Tasks.Exphil.List do
  @moduledoc """
  List all ExPhil checkpoints with metadata.

  ## Usage

      mix exphil.list [options]

  ## Options

    * `--dir PATH` - Directory to search (default: ./checkpoints)
    * `--sort FIELD` - Sort by: date, size, name (default: date)
    * `--reverse` - Reverse sort order

  ## Examples

      mix exphil.list
      mix exphil.list --dir ./my_checkpoints
      mix exphil.list --sort size --reverse

  """
  use Mix.Task

  alias ExPhil.Training.Output

  @shortdoc "List all ExPhil checkpoints"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [dir: :string, sort: :string, reverse: :boolean]
    )

    dir = Keyword.get(opts, :dir, "./checkpoints")
    sort_by = Keyword.get(opts, :sort, "date")
    reverse = Keyword.get(opts, :reverse, false)

    unless File.dir?(dir) do
      Mix.shell().error("Directory not found: #{dir}")
      System.halt(1)
    end

    # Find all checkpoint files
    axon_files = Path.wildcard(Path.join(dir, "**/*.axon"))
    bin_files = Path.wildcard(Path.join(dir, "**/*.bin"))
    all_files = axon_files ++ bin_files

    if all_files == [] do
      Mix.shell().info("No checkpoints found in #{dir}")
      System.halt(0)
    end

    # Collect file info
    checkpoints = all_files
    |> Enum.map(fn path ->
      stat = File.stat!(path)
      %{
        path: path,
        name: Path.basename(path),
        size: stat.size,
        mtime: stat.mtime,
        type: if(String.ends_with?(path, ".axon"), do: :checkpoint, else: :policy)
      }
    end)

    # Sort
    checkpoints = case sort_by do
      "size" -> Enum.sort_by(checkpoints, & &1.size)
      "name" -> Enum.sort_by(checkpoints, & &1.name)
      _ -> Enum.sort_by(checkpoints, & &1.mtime)
    end

    checkpoints = if reverse, do: checkpoints, else: Enum.reverse(checkpoints)

    # Display
    Output.section("ExPhil Checkpoints")
    Output.kv("Directory", dir)
    Output.kv("Found", "#{length(checkpoints)} checkpoint(s)")
    Output.puts_raw("")

    Enum.each(checkpoints, fn cp ->
      type_indicator = if cp.type == :checkpoint, do: "[CKPT]", else: "[POL] "
      size_str = Output.format_bytes(cp.size)
      date_str = format_datetime(cp.mtime)

      Output.puts_raw("  #{Output.colorize(type_indicator, :cyan)} #{cp.name}")
      Output.puts_raw("       Size: #{size_str} | Modified: #{date_str}")
      Output.puts_raw("       Path: #{cp.path}")
      Output.puts_raw("")
    end)

    Output.divider()
  end

  defp format_datetime({{year, month, day}, {hour, min, _sec}}) do
    "#{year}-#{pad(month)}-#{pad(day)} #{pad(hour)}:#{pad(min)}"
  end

  defp pad(n), do: String.pad_leading(to_string(n), 2, "0")
end
