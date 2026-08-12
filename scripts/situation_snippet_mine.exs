# Generalized situation-window miner (round-2 flaw campaign,
# 2026-08-12): cut windows around ONSETS of any ExPhil.Situations
# label in the human corpus, keeping recorded controllers (the human is
# the teacher — the conversion-arm recipe generalized).
#
# Filters: --min-payoff D requires >= D positive-delta damage dealt to
# the opponent within the window; --require-survive drops windows where
# WE lose a stock within --post frames of onset. Windows ranked by
# payoff (stock takes bonused), per-file capped, globally budgeted.
#
#   mix run scripts/situation_snippet_mine.exs \
#     --replays replays/fox_il_v1 --character fox \
#     --label shield_pressure_ours --min-payoff 20 \
#     --out eval_runs/0812_mine_shieldpressure --target-frames 40000
#
#   --label L[,L2]      Situations label(s) — a window opens when ANY
#                       of them turns on (union onset)
#   --require-survive   drop windows where we lose a stock in the window
#   --min-payoff D      require >= D damage dealt in the window [0]
#   --pre/--post        window around onset [30 / 90]

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
      label: :string,
      out: :string,
      pre: :integer,
      post: :integer,
      min_payoff: :float,
      require_survive: :boolean,
      target_frames: :integer,
      max_windows_per_file: :integer,
      max_files: :integer
    ]
  )

replay_dir = Path.expand(opts[:replays] || raise("--replays is required"))
out_dir = opts[:out] || raise("--out is required")
labels = (opts[:label] || raise("--label is required")) |> String.split(",") |> Enum.map(&String.to_existing_atom/1)
pre = opts[:pre] || 30
post = opts[:post] || 90
min_payoff = opts[:min_payoff] || 0.0
require_survive = opts[:require_survive] || false
target_frames = opts[:target_frames] || 40_000
per_file_cap = opts[:max_windows_per_file] || 3
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
Output.banner("Situation snippet miner")
Output.config([
  {"Labels", inspect(labels)},
  {"Window", "-#{pre}/+#{post}"},
  {"Filters", "payoff>=#{min_payoff}#{if require_survive, do: " survive", else: ""}"},
  {"Target", "#{target_frames} frames"}
])

mask = labels |> Enum.map(&Bitwise.bsl(1, Situations.bit(&1))) |> Enum.reduce(&Bitwise.bor/2)

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
          masks = states |> Situations.label_states(port) |> List.to_tuple()
          farr = List.to_tuple(frames)
          own = fn i -> elem(farr, i).game_state.players[port] end
          opp = fn i -> elem(farr, i).game_state.players[opp_port] end

          onsets =
            for i <- 1..(n - 1),
                Bitwise.band(elem(masks, i), mask) != 0,
                Bitwise.band(elem(masks, i - 1), mask) == 0,
                do: i

          qualified =
            for i <- onsets, last = min(i + post, n - 1), last > i do
              {dmg, lost, took} =
                Enum.reduce((i + 1)..last, {0.0, false, false}, fn j, {d, l, t} ->
                  {d + max((opp.(j).percent || 0.0) - (opp.(j - 1).percent || 0.0), 0.0),
                   l or (own.(j).stock || 0) < (own.(j - 1).stock || 0),
                   t or (opp.(j).stock || 0) < (opp.(j - 1).stock || 0)}
                end)

              ok = dmg >= min_payoff and not (require_survive and lost)
              if ok, do: {path, i, dmg + if(took, do: 200.0, else: 0.0)}
            end
            |> Enum.reject(&is_nil/1)
            |> Enum.sort_by(&(-elem(&1, 2)))
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

window_len = pre + post + 1
selected = candidates |> Enum.sort_by(&(-elem(&1, 2))) |> Enum.take(div(target_frames, window_len))

Output.puts("  #{stats.onsets} onsets -> #{stats.qualified} qualified -> #{length(selected)} selected from #{stats.files} files")

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
    |> Enum.map(fn {_p, i, _payoff} -> {max(i - pre, 0), min(i + post, n - 1)} end)
    |> Enum.sort()
    |> Enum.reduce([], fn {a, b}, racc ->
      case racc do
        [{pa, pb} | rest] when a <= pb + 1 -> [{pa, max(b, pb)} | rest]
        _ -> [{a, b} | racc]
      end
    end)
    |> Enum.map(fn {a, b} -> Enum.slice(frames, a, b - a + 1) end)
  end)

total = snippets |> Enum.map(&length/1) |> Enum.sum()

File.write!(
  Path.join(out_dir, "snippets.frames"),
  :erlang.term_to_binary(
    %{
      expert: "human_#{Enum.join(labels, "_")}",
      exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      action_delay: 0,
      frame_lists: snippets
    },
    [:compressed]
  )
)

Output.success("#{length(snippets)} snippets / #{total} frames -> #{out_dir}/snippets.frames")
