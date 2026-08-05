defmodule Mix.Tasks.Exphil.StampReplays do
  @shortdoc "Tag bot .slp replays with a name so they're identifiable in a corpus"

  @moduledoc """
  Stamp `.slp` replays produced by the bot with a player name.

  Local games record no names, so bot replays are otherwise
  indistinguishable from any other offline game in a corpus.

      mix exphil.stamp_replays eval_runs/0805_g12_ys
      mix exphil.stamp_replays eval_runs/0805_g12_ys --tag exph --port 1

  Options:

    * `--tag` - name to write (default: "exph")
    * `--port` - controller port the bot played on, 1-4 (default: 1)

  Already-named players are never overwritten.
  """

  use Mix.Task

  alias ExPhil.Replay.Stamp

  @impl true
  def run(argv) do
    {opts, paths, _} =
      OptionParser.parse(argv, strict: [tag: :string, port: :integer])

    if paths == [] do
      Mix.raise("usage: mix exphil.stamp_replays <dir-or-file> [--tag exph] [--port 1]")
    end

    tag = Keyword.get(opts, :tag, "exph")
    port = Keyword.get(opts, :port, 1)
    tags = %{port - 1 => tag}

    for path <- paths do
      cond do
        File.dir?(path) ->
          {stamped, skipped} = Stamp.stamp_dir(path, tags)
          Mix.shell().info("#{path}: stamped #{stamped}, skipped #{skipped}")

        File.exists?(path) ->
          case Stamp.stamp_file(path, tags) do
            {:ok, n} -> Mix.shell().info("#{path}: stamped #{n} player(s)")
            {:error, reason} -> Mix.shell().error("#{path}: #{inspect(reason)}")
          end

        true ->
          Mix.shell().error("#{path}: not found")
      end
    end
  end
end
