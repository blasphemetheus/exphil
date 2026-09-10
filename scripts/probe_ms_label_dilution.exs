# What does the multishine expert LABEL B as on aerial-reflector (365) and
# ground-reflector (361) frames, by action_frame, across the drill's rollout
# pool? (2026-09-10) Hypothesis: the policy's p(hold B) ~0.33 on aerial
# shine af1-3 comes from label dilution — rollouts contain long aerial
# reflectors (the bot floats), the recovery rule labels af>=4 "ride down
# neutral" (B released), and the 1/60-scaled action_frame scalar cannot
# separate af1-3 (hold) from af4+ (release), so the policy learns a blend.
#
#   mix run scripts/probe_ms_label_dilution.exs [--rollouts "glob,glob"]

alias ExPhil.Agents.MultishineExpert
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [rollouts: :string, port: :integer])

roll =
  opts[:rollouts] ||
    "eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"

port = opts[:port] || 1
paths = roll |> String.split(",", trim: true) |> Enum.flat_map(&Path.wildcard/1)
expert = MultishineExpert.from_fixture()

Output.banner("Expert label B by state across the rollout pool")
Output.puts("#{length(paths)} rollout replays; port #{port}")

rows =
  Enum.flat_map(paths, fn path ->
    case ExPhil.Data.Peppi.parse(path, player_port: port) do
      {:ok, replay} ->
        frames =
          ExPhil.Data.Peppi.to_training_frames(replay, player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true)

        recorded = Map.new(frames, fn f -> {f.game_state.frame, f.controller} end)

        Enum.flat_map(frames, fn f ->
          p = f.game_state.players[1]
          prev = recorded[f.game_state.frame - 1]

          case p && MultishineExpert.label(expert, p, prev) do
            {:ok, c} -> [{trunc(p.action || 0), min(trunc(p.action_frame || 0), 12), p.on_ground == true, c.button_b == true, c.button_x == true}]
            _ -> []
          end
        end)

      _ ->
        []
    end
  end)

Output.puts("#{length(rows)} labeled frames")
Output.puts("")
Output.puts("| state {action, af(capped 12), ground} | n | label B rate | label X rate |")
Output.puts("|---|---:|---:|---:|")

rows
|> Enum.group_by(fn {a, af, g, _, _} -> {a, af, g} end)
|> Enum.filter(fn {{a, _, _}, _} -> a in [361, 365, 366, 24] end)
|> Enum.sort()
|> Enum.each(fn {key, g} ->
  n = length(g)
  b = Enum.count(g, fn {_, _, _, bb, _} -> bb end) / n
  x = Enum.count(g, fn {_, _, _, _, xx} -> xx end) / n
  Output.puts("| #{inspect(key)} | #{n} | #{Float.round(b, 2)} | #{Float.round(x, 2)} |")
end)
