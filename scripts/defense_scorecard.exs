# F1 — defense scorecard (EVAL_DIRECTIONS F1, Bradley's 08-31 live-look ask):
# "you can hold a certain direction and get to ledge — it doesn't hold that
# direction, airdodges off stage and dies."
#
# HITSTUN episodes (entry: :in_hitstun/:tumble label turns on). An episode is
# OFFSTAGE-CLASS if the subject is offstage during the hitstun run or within
# 30 f after it. Per episode:
#   - DI/hold-toward-stage % : among hitstun frames where the main stick is
#     held (|x-0.5| > 0.2), the share held TOWARD the stage (sign vs -sign(pos_x))
#   - drift-toward-stage %   : same measure over the 45 f after hitstun ends,
#     while airborne (the survival drift Bradley described)
#   - airdodge-in-danger     : an airdodge (action 236) initiated offstage
#     within the episode horizon; and died within 90 f of it
#   - outcome                : died (stock drop within --horizon) vs survived
# Global (non-episode): among ALL offstage airborne non-hitstun frames, the
# share of held-stick frames pointing toward the stage — the plain
# "does it hold the right direction out there" number.
#
#   mix run scripts/defense_scorecard.exs \
#     --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' --expert expert \
#     --expert-char 2 --expert-limit 600 \
#     --set AR_human='eval_runs/0831_livelook_v11ar/2026-08-Mainline/*.slp' \
#     --out eval_runs/0831_session_score/defense_scorecard.md
#
# Options mirror edge_scorecard: --set NAME=GLOB (repeatable) · --expert NAME ·
# --port N (bot sets, default 1) · --expert-port N | --expert-char N ·
# --expert-limit N (600) · --limit-files N · --horizon N (240) · --out PATH
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, expert: :string, port: :integer, expert_port: :integer,
             expert_char: :integer, expert_limit: :integer,
             limit_files: :integer, horizon: :integer, concurrency: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, g] = String.split(s, "=", parts: 2); {n, g} end)
if sets == [], do: raise("--set NAME=GLOB required")
expert = opts[:expert] || "expert"
port = opts[:port] || 1
expert_port = opts[:expert_port] || 1
horizon = opts[:horizon] || 240
conc = opts[:concurrency] || 12

files_for = fn {name, glob} ->
  fs = glob |> Path.wildcard() |> Enum.sort() |> Enum.reject(&(File.stat!(&1).size < 150_000))
  limit = if name == expert, do: opts[:expert_limit] || 600, else: opts[:limit_files]
  if limit, do: Enum.take(fs, limit), else: fs
end

Output.banner("Defense scorecard (F1)")
Output.config(Enum.map(sets, fn {n, g} = s -> {n, "#{length(files_for.(s))} files (#{g})"} end) ++ [{"Horizon", "#{horizon} f"}])

hold_dead = 0.2

scan = fn path, p ->
  o = if p == 1, do: 2, else: 1

  try do
    with {:ok, replay} <- Peppi.parse(path, player_port: p) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: p, opponent_port: o)
        |> Enum.reject(&(&1.game_state.frame < 0))

      n = length(frames)

      if n < 300 do
        nil
      else
        states = Enum.map(frames, & &1.game_state)
        arr = List.to_tuple(states)
        ctrl = frames |> Enum.map(& &1.controller) |> List.to_tuple()
        my_sits = Situations.label_states(states, p, as: :set) |> List.to_tuple()

        pl = fn i -> elem(arr, i).players[p] end
        off = fn i -> MapSet.member?(elem(my_sits, i), :offstage) end
        hit = fn i ->
          s = elem(my_sits, i)
          MapSet.member?(s, :in_hitstun) or MapSet.member?(s, :tumble)
        end
        act = fn i -> trunc(pl.(i).action || 0) end

        # Held-stick direction relative to the stage (center x = 0 on every
        # legal stage): +1 toward, -1 away, nil not held / at center.
        dir = fn i ->
          dx = (elem(ctrl, i).main_stick.x || 0.5) - 0.5
          px = pl.(i).x || 0.0

          cond do
            abs(dx) <= hold_dead or abs(px) < 1.0 -> nil
            dx * px < 0 -> :toward
            true -> :away
          end
        end

        count_dirs = fn range ->
          Enum.reduce(range, {0, 0}, fn i, {t, h} ->
            case dir.(i) do
              :toward -> {t + 1, h + 1}
              :away -> {t, h + 1}
              nil -> {t, h}
            end
          end)
        end

        # Contiguous hitstun runs
        runs =
          1..(n - 1)
          |> Enum.filter(fn i -> hit.(i) and not hit.(i - 1) end)
          |> Enum.map(fn i ->
            fin = Enum.find(i..(n - 1), fn k -> not hit.(k) end) || n - 1
            {i, fin}
          end)

        eps =
          Enum.map(runs, fn {i, fin} ->
            j = min(i + horizon, n - 1)
            offclass = Enum.any?(i..min(fin + 30, n - 1), off)
            my_stock = pl.(i).stock

            died_at =
              Enum.find(i..j, fn k ->
                is_integer(pl.(k).stock) and pl.(k).stock < my_stock
              end)

            {di_t, di_h} = count_dirs.(i..max(fin - 1, i))

            post_end =
              Enum.reduce_while(fin..min(fin + 45, n - 1), fin, fn k, _ ->
                if pl.(k).on_ground, do: {:halt, k}, else: {:cont, k}
              end)

            {dr_t, dr_h} = if post_end > fin, do: count_dirs.(fin..post_end), else: {0, 0}

            ad_at =
              Enum.find(i..j, fn k -> act.(k) == 236 and off.(k) end)

            died_after_ad =
              ad_at != nil and died_at != nil and died_at >= ad_at and died_at <= ad_at + 90

            %{offclass: offclass, died: died_at != nil,
              di: {di_t, di_h}, drift: {dr_t, dr_h},
              airdodge: ad_at != nil, died_after_ad: died_after_ad}
          end)

        # Global offstage drift (airborne, not in hitstun)
        {g_t, g_h} =
          count_dirs.(Enum.filter(0..(n - 1), fn i ->
            off.(i) and not hit.(i) and not pl.(i).on_ground
          end))

        %{eps: eps, g: {g_t, g_h}, files: 1, frames: n}
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

expert_char = opts[:expert_char]

port_for = fn name, path ->
  cond do
    name != expert -> port
    expert_char == nil -> expert_port
    true ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == expert_char)) do
            [%{port: pp}] -> pp
            _ -> nil
          end

        _ -> nil
      end
  end
end

agg =
  Map.new(sets, fn {name, _} = s ->
    files = files_for.(s)

    pairs =
      files |> Enum.map(&{&1, port_for.(name, &1)}) |> Enum.reject(fn {_, p} -> is_nil(p) end)

    Output.puts("Scanning #{name}: #{length(pairs)}/#{length(files)} files")

    r =
      pairs
      |> Task.async_stream(fn {f, p} -> scan.(f, p) end,
        max_concurrency: conc, timeout: :infinity, ordered: false)
      |> Enum.reduce(%{eps: [], g: {0, 0}, files: 0, frames: 0}, fn {:ok, x}, acc ->
        if x do
          {gt, gh} = acc.g
          {xt, xh} = x.g
          %{eps: x.eps ++ acc.eps, g: {gt + xt, gh + xh},
            files: acc.files + 1, frames: acc.frames + x.frames}
        else
          acc
        end
      end)

    {name, r}
  end)

names = Enum.map(sets, &elem(&1, 0))
f1 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
pct = fn t, h -> if h < 20, do: "– (#{h}f)", else: f1.(t / h) end

sum_pairs = fn eps, key ->
  Enum.reduce(eps, {0, 0}, fn e, {t, h} -> {et, eh} = Map.fetch!(e, key); {t + et, h + eh} end)
end

row = fn label, cell_fn ->
  "| #{label} | " <> Enum.map_join(names, " | ", cell_fn) <> " |"
end

offeps = fn n -> Enum.filter(agg[n].eps, & &1.offclass) end

table =
  [
    "| defense | " <> Enum.map_join(names, " | ", fn n -> "#{n} (off-eps n=#{length(offeps.(n))})" end) <> " |",
    "|---|" <> String.duplicate("---:|", length(names)),
    row.("offstage hitstun episodes / min", fn n ->
      :erlang.float_to_binary(length(offeps.(n)) / max(agg[n].frames / 3600, 1), decimals: 2)
    end),
    row.("DI toward stage % (in hitstun)", fn n -> {t, h} = sum_pairs.(offeps.(n), :di); pct.(t, h) end),
    row.("drift toward stage % (post-hitstun)", fn n -> {t, h} = sum_pairs.(offeps.(n), :drift); pct.(t, h) end),
    row.("airdodge offstage in episode %", fn n ->
      eps = offeps.(n)
      if eps == [], do: "–", else: f1.(Enum.count(eps, & &1.airdodge) / length(eps))
    end),
    row.("died within 90f of that airdodge %", fn n ->
      ads = offeps.(n) |> Enum.filter(& &1.airdodge)
      if length(ads) < 5, do: "– (#{length(ads)})", else: f1.(Enum.count(ads, & &1.died_after_ad) / length(ads)) <> " (#{length(ads)})"
    end),
    row.("episode died %", fn n ->
      eps = offeps.(n)
      if eps == [], do: "–", else: f1.(Enum.count(eps, & &1.died) / length(eps))
    end),
    row.("GLOBAL offstage drift toward stage %", fn n -> {t, h} = agg[n].g; pct.(t, h) end),
    row.("onstage-hitstun DI toward center %", fn n ->
      eps = Enum.reject(agg[n].eps, & &1.offclass)
      {t, h} = sum_pairs.(eps, :di)
      pct.(t, h)
    end)
  ]
  |> Enum.join("\n")

report = """
# Defense scorecard (F1)

Sets: #{Enum.map_join(names, " · ", fn n -> "#{n}: #{agg[n].files} files" end)}. Horizon #{horizon} f.
Episode = contiguous hitstun/tumble run; offstage-class if offstage during the run
or within 30 f after. "Toward stage" = main stick held (|x-0.5| > #{hold_dead}) with
sign opposing the subject's x position (stage center 0). Cells with under 20 held
frames / 5 episodes report "–" — do not read them.

#{table}

Read: the expert column is the DI/drift denominator Bradley described ("hold a
direction and get to ledge"). A bot matching expert drift but dying more is
route-selection (A2's lane); a bot NOT holding toward stage is missing the
survival input itself — a corpus/curation question (G1 target #1), never a
decode mask (standing rule).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
