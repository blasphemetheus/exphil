# Drill-table mining (DRILL_HITCONFIRM.md build step 1, 09-01).
#
# Groups expert conversions by (opener family, victim percent decile,
# stage zone) and reports per cell: frequency, continuation hits (hitstun
# rising edges within the window, the F4 counter), damage, duration —
# the reference distributions the hit-confirm drill scores against.
#
#   mix run scripts/drill_table_mine.exs \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --limit-files 200 --out eval_runs/0901_drill_table/RESULTS.md
#
# Options: --char-id (2) · --post (240) · --min-cell (25)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, limit_files: :integer, char_id: :integer,
             post: :integer, min_cell: :integer, out: :string]
  )

glob = opts[:replays] || raise "--replays required"
limit_files = opts[:limit_files] || 200
char_id = opts[:char_id] || 2
post = opts[:post] || 240
min_cell = opts[:min_cell] || 25

Output.banner("Hit-confirm drill-table mining")

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files * 3)

resolve = fn path ->
  case Peppi.metadata(path) do
    {:ok, meta} ->
      case Enum.filter(meta.players, &(&1.character == char_id)) do
        [%{port: p}] -> {:ok, p}
        _ -> :skip
      end

    _ -> :skip
  end
end

picked =
  files
  |> Enum.flat_map(fn f ->
    case resolve.(f) do
      {:ok, p} -> [{f, p}]
      _ -> []
    end
  end)
  |> Enum.take(limit_files)

Output.puts("  #{length(picked)} fox-resolved files")

# Opener families by subject action id (internal action-state space)
family = fn a ->
  cond do
    a in 44..49 -> :jab
    a == 50 -> :dash_attack
    a in 51..57 -> :tilt
    a in 58..64 -> :smash
    a == 65 -> :nair
    a == 66 -> :fair
    a == 67 -> :bair
    a == 68 -> :uair
    a == 69 -> :dair
    # libmelee enums: GRAB 0xd4=212 .. GRAB_PUMMEL 0xd9=217, GRAB_WAIT 0xd8;
    # throws THROW_FORWARD 0xdb=219, BACK 220, UP 221, DOWN 222 (the first
    # pass had these off by two — up-throws landed in :other).
    a == 219 -> :fthrow
    a == 220 -> :bthrow
    a == 221 -> :uthrow
    a == 222 -> :dthrow
    a in 212..218 -> :grab_hold
    a >= 341 -> :special
    true -> :other
  end
end

zone = fn x ->
  ax = abs(x || 0.0)

  cond do
    ax < 30 -> :mid
    ax < 70 -> :side
    true -> :edge
  end
end

# A "hit" on the victim = hitstun OR a THROWN_* state (239..243) — thrown
# victims carry hitstun_frames_left == 0, so a hitstun-only detector drops
# every grab->throw conversion (found 09-01: the throw table was EMPTY).
hitstun? = fn pl ->
  pl != nil and
    ((pl.hitstun_frames_left || 0) > 0 or pl.action in 239..243 or pl.action in 223..232)
end

conversions =
  picked
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {{path, port}, i} ->
    if rem(i, 20) == 0, do: Output.progress_bar(i, length(picked), label: "mining")
    opp = if port == 1, do: 2, else: 1

    case Peppi.parse(path) do
      {:ok, replay} ->
        frames =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp, remap_ports: true)
          |> Enum.reject(&(&1.game_state.frame < 0))

        states = Enum.map(frames, & &1.game_state)
        sits = Situations.label_states(states, 1, as: :set)
        states_t = List.to_tuple(states)
        n = tuple_size(states_t)

        # conversion onsets = :conversion_open rising edges
        onsets =
          Enum.zip(sits, [nil | sits])
          |> Enum.with_index()
          |> Enum.filter(fn {{s, prev}, _i} ->
            s != nil and MapSet.member?(s, :conversion_open) and
              (prev == nil or not MapSet.member?(prev, :conversion_open))
          end)
          |> Enum.map(fn {_, i} -> i end)

        Enum.flat_map(onsets, fn t0 ->
          if t0 + post >= n do
            []
          else
            gs0 = elem(states_t, t0)
            own0 = gs0.players[1]
            vic0 = gs0.players[2]

            # opener = subject's action at the first opp-hitstun rising edge
            # in [t0, t0+30]
            hit1 =
              Enum.find(t0..min(t0 + 30, n - 1), fn t ->
                hitstun?.(elem(states_t, t).players[2]) and
                  (t == 0 or not hitstun?.(elem(states_t, t - 1).players[2]))
              end)

            if hit1 == nil or own0 == nil or vic0 == nil do
              []
            else
              # Opener = the subject's most recent ATTACK-family action in
              # the 10 frames up to the hitstun edge — at the edge itself
              # the move is often already IASA'd/ended (the first-pass
              # "other" bucket at 640 rows was mostly this + lasers).
              opener =
                Enum.find_value(hit1..max(hit1 - 10, 0)//-1, :other, fn t ->
                  case family.(elem(states_t, t).players[1].action) do
                    :other -> nil
                    fam -> fam
                  end
                end)

              # A grab is not the payoff — resolve forward to the THROW
              # (subject enters 219..222 within 120 f) and anchor the
              # continuation window at the throw instead.
              {opener, hit1} =
                if opener == :grab_hold do
                  case Enum.find(hit1..min(hit1 + 120, n - 1), fn t ->
                         elem(states_t, t).players[1].action in 219..222
                       end) do
                    nil -> {:grab_hold, hit1}
                    tt -> {family.(elem(states_t, tt).players[1].action), tt}
                  end
                else
                  {opener, hit1}
                end

              # continuation stats over [hit1, hit1+post]
              {hits, _} =
                Enum.reduce(hit1..min(hit1 + post, n - 1), {0, true}, fn t, {h, prev_hs} ->
                  hs = hitstun?.(elem(states_t, t).players[2])
                  {(if hs and not prev_hs, do: h + 1, else: h), hs}
                end)

              vic_end = elem(states_t, min(hit1 + post, n - 1)).players[2]

              dmg =
                if vic_end, do: max((vic_end.percent || 0.0) - (vic0.percent || 0.0), 0.0), else: 0.0

              stock =
                if vic_end && vic0 && (vic_end.stock || 0) < (vic0.stock || 0), do: 1, else: 0

              [%{opener: opener, decile: min(div(trunc(vic0.percent || 0.0), 10), 12),
                 zone: zone.(own0.x), hits: hits + 1, dmg: dmg, stock: stock}]
            end
          end
        end)

      _ -> []
    end
  end)

Output.progress_done()
Output.puts("  #{length(conversions)} conversions mined")

cells =
  conversions
  |> Enum.group_by(fn c -> {c.opener, div(c.decile, 2), c.zone} end)
  |> Enum.map(fn {{op, band, z}, cs} ->
    nn = length(cs)
    mh = Enum.sum(Enum.map(cs, & &1.hits)) / nn
    md = Enum.sum(Enum.map(cs, & &1.dmg)) / nn
    deep = Enum.count(cs, &(&1.hits >= 3)) / nn
    st = Enum.sum(Enum.map(cs, & &1.stock)) / nn

    %{opener: op, pct_band: "#{band * 20}-#{band * 20 + 19}", zone: z, n: nn,
      mean_hits: mh, deep3: deep, mean_dmg: md, stock_rate: st,
      score: nn * (mh - 1.0)}
  end)
  |> Enum.filter(&(&1.n >= min_cell))
  |> Enum.sort_by(&(-&1.score))

f1 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 1) end
pct = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 0) end

row = fn c ->
  "| #{c.opener} | #{c.pct_band}% | #{c.zone} | #{c.n} | #{f1.(c.mean_hits)} | " <>
    "#{pct.(c.deep3)} | #{f1.(c.mean_dmg)} | #{pct.(c.stock_rate)} |"
end

rows = cells |> Enum.take(20) |> Enum.map_join("\n", row)

throw_rows =
  cells
  |> Enum.filter(&(&1.opener in [:uthrow, :dthrow, :fthrow, :bthrow]))
  |> Enum.map_join("\n", row)

report = """
# Hit-confirm drill table — expert conversion cells

#{length(conversions)} conversions from #{length(picked)} files; window
#{post} f after hit 1; cells with n >= #{min_cell}, ranked by
n x (mean_hits - 1) ("how much continuation this cell teaches").

| opener | victim % | zone | n | mean hits | >=3 hits % | mean dmg | stock % |
|---|---|---|---:|---:|---:|---:|---:|
#{rows}

## Throw cells (all, regardless of rank — Drill 1 candidates)

| opener | victim % | zone | n | mean hits | >=3 hits % | mean dmg | stock % |
|---|---|---|---:|---:|---:|---:|---:|
#{throw_rows}

Drill 1 pre-registration check (DRILL_HITCONFIRM.md): the uthrow rows in
the 0-39% bands are the candidate cells — their mean-hits / >=3-hit /
damage columns are the reference distributions the drill scores against.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
