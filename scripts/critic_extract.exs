# D2 critic — step 1: extract features, targets and candidate actions.
#
# For each replay: run the policy's trunk over every 60-frame window,
# concatenate the raw scalars the trunk discards, compute the discounted
# standard-reward return-to-go, flag decision frames, and draw K policy
# samples per row (the negatives the selector learns to rank below the
# master's action). Writes one Nx-serialized file.
#
# Usage:
#   mix run scripts/critic_extract.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 \
#     --limit-files 40 --k 8 --stride 2 --out cache/critic/fox_gen_v1_ep10_erickfm40.nx
#
# Options:
#   --policy PATH       exported policy .bin (temporal)        (required)
#   --replays GLOB      replays                                 (required)
#   --port N            pin the master's port; omit to auto-detect by --char-id
#   --char-id N         character for auto-detect (default 2 = Fox)
#   --limit-files N     default 40
#   --k N               policy samples per row (default 8)
#   --temperature T     sample decode temperature (default 0.5 = deploy)
#   --stride N          keep every Nth non-decision row (default 2); decision rows always kept
#   --gamma F           return-to-go discount (default 0.99)
#   --horizon N         return-to-go horizon in frames (default 600 = 10 s)
#   --out PATH          output file (required)
#   --seed N
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, replays: :string, port: :integer, char_id: :integer,
      limit_files: :integer, k: :integer, temperature: :float, stride: :integer,
      gamma: :float, horizon: :integer, out: :string, seed: :integer
    ]
  )

policy = opts[:policy] || raise "--policy required"
glob = opts[:replays] || raise "--replays required"
out = opts[:out] || raise "--out required"
char_id = opts[:char_id] || 2
limit = opts[:limit_files] || 40
k = opts[:k] || 8

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("D2 critic — feature extraction")

Output.config([
  {"Policy", Path.basename(policy)},
  {"Replays", "#{length(files)} files"},
  {"Port", if(opts[:port], do: "pinned #{opts[:port]}", else: "auto by char #{char_id}")},
  {"K samples", k},
  {"Temperature", opts[:temperature] || 0.5},
  {"Stride", opts[:stride] || 2},
  {"Out", out}
])

resolve_port = fn path ->
  if opts[:port] do
    {:ok, opts[:port]}
  else
    case Peppi.metadata(path) do
      {:ok, meta} ->
        case Enum.filter(meta.players, &(&1.character == char_id)) do
          [%{port: p}] -> {:ok, p}
          _ -> :skip
        end

      _ ->
        :skip
    end
  end
end

trunk = Activations.load_trunk(policy)
heads = Activations.load_heads_only(policy)
key = Nx.Random.key(opts[:seed] || 20_260_829)

Output.puts("Extracting (trunk hidden #{trunk.hidden_size}, raw #{CriticFeatures.raw_size()})...")

{extracts, skipped, _key} =
  files
  |> Enum.with_index(1)
  |> Enum.reduce({[], 0, key}, fn {path, i}, {acc, skipped, key} ->
    Output.progress_bar(i, length(files), label: "replays")
    sub = Nx.Random.fold_in(key, i)

    case resolve_port.(path) do
      {:ok, port} ->
        try do
          e =
            CriticFeatures.extract_replay(trunk, heads, path,
              port: port, k: k,
              temperature: opts[:temperature] || 0.5,
              stride: opts[:stride] || 2,
              gamma: opts[:gamma] || 0.99,
              horizon: opts[:horizon] || 600,
              key: sub
            )

          {[e | acc], skipped, key}
        rescue
          err ->
            Output.warning("skip #{Path.basename(path)}: #{Exception.message(err)}")
            {acc, skipped + 1, key}
        end

      :skip ->
        {acc, skipped + 1, key}
    end
  end)

Output.progress_done()
if extracts == [], do: raise("nothing extracted")

data = CriticFeatures.concat(Enum.reverse(extracts))
n = Nx.axis_size(data.phi, 0)
nd = Nx.sum(data.decision) |> Nx.to_number()
sample_hit = Nx.mean(Nx.as_type(data.match, :f32)) |> Nx.to_number()

meta = %{
  policy: policy, files: length(extracts), skipped: skipped, rows: n,
  decision_rows: nd, k: k, temperature: opts[:temperature] || 0.5,
  phi_size: Nx.axis_size(data.phi, 1), hidden: trunk.hidden_size,
  raw: CriticFeatures.raw_size(), created: DateTime.utc_now() |> DateTime.to_iso8601()
}

CriticFeatures.save!(out, Map.put(data, :meta, Nx.tensor(0)))
File.write!(out <> ".meta.json", Jason.encode!(meta, pretty: true))

Output.success(
  "#{n} rows (#{nd} decision) from #{length(extracts)} replays (#{skipped} skipped); " <>
    "sample==master rate #{Float.round(sample_hit * 100, 1)}% (this is sampling pass@1 on ALL rows) -> #{out}"
)
