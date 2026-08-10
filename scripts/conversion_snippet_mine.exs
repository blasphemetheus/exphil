# Conversion-arm harvest (CONVERSION_ARM_SPEC.md, arm C1): cut windows
# around SUCCESSFUL human conversions. No relabeling — the human's
# conversion IS the label (rule 2: recorded controllers are valid labels
# when the recorder is the teacher). This oversamples
# REPRESENTED-but-diluted behavior, unlike the edge arm's corrections.
#
# Anchor: a :conversion_open ONSET (ExPhil.Situations — an opener landed
# from neutral) whose outcome window shows real payoff: >= --min-damage
# positive-delta damage to the opponent within --post frames, or a stock
# taken. Windows [onset - pre, onset + post], overlaps merged, ranked by
# payoff (stock takes first), capped per file and globally by
# --target-frames.
#
#   mix run scripts/conversion_snippet_mine.exs \
#     --replays replays/fox_il_v1 --character fox \
#     --out eval_runs/0811_conversion_snippets \
#     [--pre 45] [--post 150] [--min-damage 25] [--target-frames 200000]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :string,
      character: :string,
      out: :string,
      pre: :integer,
      post: :integer,
      min_damage: :float,
      target_frames: :integer,
      max_windows_per_file: :integer,
      max_files: :integer
    ]
  )

replay_dir = Path.expand(opts[:replays] || raise("--replays is required"))
out_dir = opts[:out] || raise("--out is required")
pre = opts[:pre] || 45
post = opts[:post] || 150
min_damage = opts[:min_damage] || 25.0
target_frames = opts[:target_frames] || 200_000
per_file_cap = opts[:max_windows_per_file] || 4
character = opts[:character] || "fox"

external_ids = %{"fox" => 2, "falco" => 20, "marth" => 9, "mewtwo" => 10, "sheik" => 19}
want_id = external_ids[character] || raise("unknown character #{character}")

pick_ports = fn meta ->
  ports = Enum.map(meta.players, & &1.port)

  with true <- length(ports) == 2,
       %{port: p} <- Enum.find(meta.players, &(&1.character == want_id)) do
    {p, Enum.find(ports, &(&1 != p))}
  else
    _ -> nil
  end
end

files = Path.wildcard(Path.join(replay_dir, "**/*.slp")) |> Enum.sort()
files = if opts[:max_files], do: Enum.take(files, opts[:max_files]), else: files

File.mkdir_p!(out_dir)
Output.banner("Conversion snippet miner")
Output.config([
  {"Replays", "#{replay_dir} (#{length(files)})"},
  {"Window", "-#{pre}/+#{post}"},
  {"Payoff", ">= #{min_damage} dmg or a stock"},
  {"Target", "#{target_frames} frames"}
])

conv_bit = Situations.bit(:conversion_open)

{candidates, stats} =
  files
  |> Enum.with_index(1)
  |> Enum.reduce({[], %{files: 0, onsets: 0, qualified: 0}}, fn {path, idx}, {acc, st} ->
    if rem(idx, 200) == 0,
      do: IO.write(:stderr, "\r  [#{idx}/#{length(files)}] #{st.qualified} qualified\e[K")

    try do
      with {:ok, meta} <- Peppi.metadata(path),
           {port, opp_port} when is_integer(port) <- pick_ports.(meta) || :skip,
           {:ok, replay} <- Peppi.parse(path, player_port: port) do
        frames =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
          |> Enum.reject(&(&1.game_state.frame < 0))

        n = length(frames)

        if n < 500 do
          {acc, st}
        else
          states = Enum.map(frames, & &1.game_state)
          masks = Situations.label_states(states, port)
          marr = List.to_tuple(masks)
          farr = List.to_tuple(frames)

          opp = fn i -> elem(farr, i).game_state.players[opp_port] end

          onsets =
            for i <- 1..(n - 1),
                Bitwise.band(elem(marr, i), Bitwise.bsl(1, conv_bit)) != 0,
                Bitwise.band(elem(marr, i - 1), Bitwise.bsl(1, conv_bit)) == 0,
                do: i

          qualified =
            for i <- onsets, last = min(i + post, n - 1), last > i do
              {dmg, took} =
                Enum.reduce((i + 1)..last, {0.0, false}, fn j, {d, tk} ->
                  {p, q} = {opp.(j), opp.(j - 1)}

                  {d + max((p.percent || 0.0) - (q.percent || 0.0), 0.0),
                   tk or (p.stock || 0) < (q.stock || 0)}
                end)

              payoff = dmg + if(took, do: 200.0, else: 0.0)
              if dmg >= min_damage or took, do: {path, port, i, payoff}
            end
            |> Enum.reject(&is_nil/1)
            |> Enum.sort_by(&(-elem(&1, 3)))
            |> Enum.take(per_file_cap)

          {qualified ++ acc,
           %{st | files: st.files + 1, onsets: st.onsets + length(onsets), qualified: st.qualified + length(qualified)}}
        end
      else
        _ -> {acc, st}
      end
    rescue
      _ -> {acc, st}
    catch
      _, _ -> {acc, st}
    end
  end)

IO.write(:stderr, "\r\e[K")

# Global budget: best payoff first until target_frames
window_len = pre + post + 1
budget_windows = div(target_frames, window_len)
selected = candidates |> Enum.sort_by(&(-elem(&1, 3))) |> Enum.take(budget_windows)

Output.puts(
  "  #{stats.onsets} onsets -> #{stats.qualified} qualified -> #{length(selected)} selected " <>
    "(~#{length(selected) * window_len} frames) from #{stats.files} files"
)

# Cut the selected windows (re-parse per file, grouped)
snippets =
  selected
  |> Enum.group_by(&elem(&1, 0))
  |> Enum.flat_map(fn {path, wins} ->
    {:ok, meta} = Peppi.metadata(path)
    {port, opp_port} = pick_ports.(meta)
    {:ok, replay} = Peppi.parse(path, player_port: port)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
      |> Enum.reject(&(&1.game_state.frame < 0))

    n = length(frames)

    wins
    |> Enum.map(fn {_p, _port, i, _payoff} -> {max(i - pre, 0), min(i + post, n - 1)} end
    )
    |> Enum.sort()
    |> Enum.reduce([], fn {a, b}, racc ->
      case racc do
        [{pa, pb} | rest] when a <= pb + 1 -> [{pa, max(b, pb)} | rest]
        _ -> [{a, b} | racc]
      end
    end)
    |> Enum.map(fn {a, b} -> Enum.slice(frames, a, b - a + 1) end)
  end)

total_frames = snippets |> Enum.map(&length/1) |> Enum.sum()

File.write!(
  Path.join(out_dir, "snippets.frames"),
  :erlang.term_to_binary(
    %{
      expert: "human_conversion",
      exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      action_delay: 0,
      frame_lists: snippets
    },
    [:compressed]
  )
)

Output.success("#{length(snippets)} snippets / #{total_frames} frames -> #{out_dir}/snippets.frames")
