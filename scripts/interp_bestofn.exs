# D2 critic — step 3: offline Best-of-N on FRESH replays.
#
# The Leg S cash-in, measured end to end on replays the critic never saw:
# for each decision frame, draw N policy samples, score each with the
# selector S(s, a) (and, for comparison, with V-free baselines), pick the
# argmax, and ask whether the pick matches the master.
#
#   sampling pass@1   what the deploy decode does today (one random sample)
#   selector pass@1   Best-of-N with the trained selector
#   oracle pass@N     the Leg S ceiling
#   logit-argmax      pick the sample with the highest policy log-prob
#                     (the "cheap selector" that needs no critic — argmax
#                     collapsed LIVE, but as a re-ranker of N samples it is
#                     the baseline the critic must beat)
#
# Usage:
#   mix run scripts/interp_bestofn.exs --policy checkpoints/fox_gen_v1_..._ep10.bin \
#     --critic checkpoints/critic_fox_gen_v1_ep10.bin \
#     --replays 'replays/fox_il_v1/*.slp' --n 16 --limit-files 20 \
#     --out eval_runs/0829_critic/bestofn_fox_il_v1.md
#
# Options: --port / --char-id as in critic_extract; --n samples (default 16);
#   --temperature (default 0.5); --limit-files (default 20); --seed; --out
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, critic: :string, replays: :string, port: :integer, char_id: :integer,
      n: :integer, temperature: :float, limit_files: :integer, seed: :integer, out: :string
    ]
  )

policy = opts[:policy] || raise "--policy required"
critic_path = opts[:critic] || raise "--critic required"
glob = opts[:replays] || raise "--replays required"
n_samples = opts[:n] || 16
temperature = opts[:temperature] || 0.5
char_id = opts[:char_id] || 2

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(opts[:limit_files] || 20)
if files == [], do: raise("no replays matched #{glob}")

critic = critic_path |> File.read!() |> :erlang.binary_to_term()
if critic.kind != :d2_critic_v1, do: raise("unexpected critic kind #{inspect(critic.kind)}")

Output.banner("D2 critic — offline Best-of-N")

Output.config([
  {"Policy", Path.basename(policy)},
  {"Critic", Path.basename(critic_path)},
  {"Replays", "#{length(files)} files"},
  {"N", n_samples},
  {"Temperature", temperature}
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

      _ -> :skip
    end
  end
end

trunk = Activations.load_trunk(policy)
heads = Activations.load_heads_only(policy)
key = Nx.Random.key(opts[:seed] || 20_260_829)

# selector score for {n, c, 13} candidates against standardized phi {n, d}
score = fn phi, cands ->
  xs = Nx.divide(Nx.subtract(phi, critic.mean), critic.std)
  proj = Nx.dot(xs, critic.selector.w) |> Nx.new_axis(1)
  Nx.add(Nx.sum(Nx.multiply(cands, proj), axes: [2]), Nx.dot(cands, critic.selector.v))
end

# Extract on decision rows only (stride huge so non-decision rows drop out)
{rows, _key} =
  files
  |> Enum.with_index(1)
  |> Enum.reduce({[], key}, fn {path, i}, {acc, key} ->
    Output.progress_bar(i, length(files), label: "replays")
    sub = Nx.Random.fold_in(key, i)

    case resolve_port.(path) do
      {:ok, port} ->
        try do
          e =
            CriticFeatures.extract_replay(trunk, heads, path,
              port: port, k: n_samples, temperature: temperature,
              stride: 1_000_000_000, key: sub
            )

          {[e | acc], key}
        rescue
          err ->
            Output.warning("skip #{Path.basename(path)}: #{Exception.message(err)}")
            {acc, key}
        end

      :skip -> {acc, key}
    end
  end)

Output.progress_done()
data = CriticFeatures.concat(Enum.reverse(rows)) |> Nx.backend_transfer(Nx.default_backend())

dec = data.decision |> Nx.to_list() |> Enum.with_index() |> Enum.filter(&(elem(&1, 0) == 1)) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)
phi = Nx.take(data.phi, dec, axis: 0)
samples = Nx.take(data.a_samples, dec, axis: 0)
match = Nx.take(data.match, dec, axis: 0) |> Nx.as_type(:f32)

s = score.(phi, samples)
pick = Nx.argmax(s, axis: 1)
picked = Nx.take_along_axis(match, Nx.new_axis(pick, 1), axis: 1) |> Nx.squeeze(axes: [1])

sampling = Nx.to_number(Nx.mean(match))
selector = Nx.to_number(Nx.mean(picked))
oracle = Nx.to_number(Nx.mean(Nx.reduce_max(match, axes: [1])))

# cheap baseline: most-frequent sample (mode of the N draws) — the
# critic-free re-ranker. Ties broken by first occurrence.
mode_pick =
  samples
  |> Nx.to_list()
  |> Enum.map(fn cands ->
    freq = Enum.frequencies(cands)
    {best, _} = Enum.max_by(freq, fn {_, c} -> c end)
    Enum.find_index(cands, &(&1 == best))
  end)
  |> Nx.tensor(type: :s64)

mode_hit = Nx.take_along_axis(match, Nx.new_axis(mode_pick, 1), axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean() |> Nx.to_number()

fmt = fn v -> :erlang.float_to_binary(v * 100, decimals: 1) end
recovered = if oracle > sampling, do: (selector - sampling) / (oracle - sampling), else: 0.0

table = """
| decode on #{Nx.axis_size(dec, 0)} decision frames, N=#{n_samples} | match rate |
|---|---|
| sampling pass@1 (today's decode) | #{fmt.(sampling)}% |
| mode-of-N (critic-free re-ranker) | #{fmt.(mode_hit)}% |
| **selector Best-of-N** | **#{fmt.(selector)}%** |
| oracle pass@#{n_samples} (ceiling) | #{fmt.(oracle)}% |
"""

IO.puts("\n" <> table)
Output.puts("Selector recovers #{fmt.(recovered)}% of the sampling->oracle gap; must beat mode-of-N to justify a critic at all.")

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, "# Offline Best-of-N\n\nPolicy `#{Path.basename(policy)}`, critic `#{Path.basename(critic_path)}`, #{length(files)} replays (#{glob}).\n\n#{table}\nGap recovered: #{fmt.(recovered)}%.\n")
  Output.success("wrote #{out}")
end
