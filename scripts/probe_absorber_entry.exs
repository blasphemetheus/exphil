# Absorber-entry forensics (INTERP_ROADMAP_V2 W1, model-free first pass).
#
# The YS stage sweep gave a stochastic contrastive pair: same checkpoint
# (ms_g4_d2mix), same stage, stand dummy — r1 plays (285/min c236), r2/r3
# absorb (40-44/min c1-2, 52% squat). This script needs NO model: it
# locates each replay's longest Squat/SquatWait spell and dumps the
# game-state trajectory entering it, so good and absorbed runs can be
# diffed on pure state (position, action history, dummy state) BEFORE any
# embedding-space work. The model-side probes (B logit through the spell)
# need the 336-dim queue/delay-id embed path, which the crouch-era
# instruments don't support yet — see task #2.
#
# Usage:
#   mix run scripts/probe_absorber_entry.exs \
#     --replays "eval_runs/0804_stage_yoshis_story/r*.slp" \
#     [--pre 90] [--min-spell 120] [--out eval_runs/interp/ys_entry.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, pre: :integer, min_spell: :integer, out: :string]
  )

paths = Path.wildcard(opts[:replays] || "eval_runs/0804_stage_yoshis_story/r*.slp")
pre = opts[:pre] || 90
min_spell = opts[:min_spell] || 120
out_path = opts[:out]

if paths == [], do: raise("no replays matched")

squat = ExPhil.Constants.squat()
squat_wait = ExPhil.Constants.squat_wait()

Output.banner("Absorber-entry forensics (model-free)")

results =
  for path <- paths do
    {:ok, replay} = Peppi.parse(path)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    rows =
      Enum.map(frames, fn f ->
        p = f.game_state.players[1]
        o = f.game_state.players[2]

        %{
          action: p.action,
          af: p.action_frame,
          x: p.x,
          y: p.y,
          in_basin: p.action in [squat, squat_wait],
          opp_action: o.action,
          opp_x: o.x,
          b: f.controller.button_b,
          x_btn: f.controller.button_x,
          stick_y: f.controller.main_stick.y
        }
      end)

    # Longest basin spell: {start_idx, len}
    {spells, _} =
      rows
      |> Enum.with_index()
      |> Enum.reduce({[], nil}, fn {r, i}, {acc, cur} ->
        cond do
          r.in_basin and cur == nil -> {acc, {i, 1}}
          r.in_basin -> {acc, {elem(cur, 0), elem(cur, 1) + 1}}
          cur != nil -> {[cur | acc], nil}
          true -> {acc, nil}
        end
      end)

    spells = Enum.sort_by(spells, &(-elem(&1, 1)))
    {entry, len} = List.first(spells) || {nil, 0}

    basin_frac = Enum.count(rows, & &1.in_basin) / max(length(rows), 1)
    n_long = Enum.count(spells, &(elem(&1, 1) >= min_spell))

    Output.puts("")
    Output.puts("#{Path.basename(path)}: frames=#{length(rows)} basin=#{Float.round(basin_frac * 100, 1)}% " <>
      "longest_spell=#{len}f@#{entry} spells>=#{min_spell}f: #{n_long}")

    entry_window =
      if entry && len >= min_spell do
        lo = max(entry - pre, 0)

        window = Enum.slice(rows, lo, entry - lo + 10)

        # Compact print: action transitions only
        window
        |> Enum.with_index(lo)
        |> Enum.chunk_by(fn {r, _} -> {r.action, r.b, r.x_btn} end)
        |> Enum.each(fn chunk ->
          {{r, i}, n} = {List.first(chunk), length(chunk)}
          mark = if i >= entry, do: " <<< SPELL", else: ""

          Output.puts(
            "  t#{i} x#{n}: action=#{r.action} af=#{r.af} pos=(#{Float.round(r.x * 1.0, 1)},#{Float.round(r.y * 1.0, 1)}) " <>
              "B=#{r.b} X=#{r.x_btn} sy=#{Float.round(r.stick_y * 1.0, 2)} opp=(#{r.opp_action}@#{Float.round(r.opp_x * 1.0, 1)})#{mark}"
          )
        end)

        window
      else
        Output.puts("  (no spell >= #{min_spell}f — GOOD run)")
        nil
      end

    %{
      replay: Path.basename(path),
      frames: length(rows),
      basin_frac: basin_frac,
      longest_spell: len,
      spell_start: entry,
      n_long_spells: n_long,
      entry_window: entry_window
    }
  end

if out_path do
  File.mkdir_p!(Path.dirname(out_path))
  File.write!(out_path, Jason.encode!(results))
  Output.success("Wrote #{out_path}")
end
