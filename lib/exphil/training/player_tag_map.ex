defmodule ExPhil.Training.PlayerTagMap do
  @moduledoc """
  `--player-tag-map <json>`: per-file identity overrides from the style
  identity pipeline (STYLE_IDENTITY.md S3; `scripts/style_identify.exs` +
  `scripts/style_cluster.exs`). Entries are `path => %{tag, port, p}` where
  `tag` is either a real player's tag the closed-set matcher assigned, or a
  pseudo-tag `~cNN` from clustering. Consulted by `Streaming.subject_tag/3`
  BEFORE the in-file / filename fallback, so matched or clustered games
  train under that identity instead of the anonymous bucket.

  Paths are matched on the expanded absolute path, then on basename (the
  corpora are flat directories with unique basenames; the preflight subset
  copies files under another root). An entry only applies to the subject
  port it was computed for.
  """

  defstruct entries: %{}, by_base: %{}, source: nil, count: 0

  @type t :: %__MODULE__{}

  @spec load!(Path.t()) :: t()
  def load!(path) do
    %{"entries" => entries} = path |> File.read!() |> Jason.decode!()

    parsed =
      Map.new(entries, fn {p, e} ->
        {Path.expand(p), %{tag: e["tag"], port: e["port"], p: e["p"]}}
      end)

    %__MODULE__{
      entries: parsed,
      by_base: Map.new(parsed, fn {p, e} -> {Path.basename(p), e} end),
      source: path,
      count: map_size(parsed)
    }
  end

  @doc "Tag for `path`/`port` if the map has one (nil otherwise)."
  @spec lookup(t() | nil, Path.t(), integer() | nil) :: String.t() | nil
  def lookup(nil, _path, _port), do: nil

  def lookup(%__MODULE__{} = map, path, port) do
    case Map.get(map.entries, Path.expand(path)) || Map.get(map.by_base, Path.basename(path)) do
      %{tag: tag, port: p} when is_nil(port) or is_nil(p) or p == port -> tag
      _ -> nil
    end
  end

  @doc "Is `tag` a clustering pseudo-tag?"
  @spec pseudo?(String.t() | nil) :: boolean()
  def pseudo?(tag) when is_binary(tag), do: String.starts_with?(tag, "~c")
  def pseudo?(_), do: false
end
