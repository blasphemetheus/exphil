# Stock-outcome rules vs human labels over a GA run's elite traces.
#
# The GA's kill bonus fires on "P2 was hit within 150 f of losing the stock",
# which credits a defender that drifts off after one shine. This script reads
# every traced elite of a run, classifies the defender's stock loss with a few
# candidate rules, and (with --labels labels.json exported from the GA page)
# reports how often each rule agrees with the human call.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/stock_outcome_eval.exs \
#     --run eval_runs/0922_ga/v2_prior_p0 [--labels path/labels.json] [--gap 150]
#
# Verdicts: :survived (no stock lost) · :sd (no hit within --gap f of the loss)
# · :kill (never actionable between the last hit and the loss) · :edgeguard
# (actionable again inside a recovery envelope, then died anyway).
{opts, _, _} = OptionParser.parse(System.argv(), strict: [run: :string, labels: :string, gap: :integer, edge: :float])
run = opts[:run] || raise("--run DIR required")
gap = opts[:gap] || 150
summary = Jason.decode!(File.read!(Path.join(run, "run.json")))
edge = opts[:edge] || %{32 => 85.5, 31 => 68.4, 8 => 56.0, 28 => 77.3, 3 => 87.75, 2 => 63.35}[summary["stage"]] || 85.5
labels =
  case opts[:labels] do
    nil -> %{}
    path -> Jason.decode!(File.read!(path))["labels"] |> Map.new(fn {k, v} -> {Path.basename(k), v["label"]} end)
  end

# MSLTRACE1 player row (ExPhil.Sim.Trace.player_row/2), keyframes only.
defmodule Row do
  def p2(row), do: Enum.at(Enum.at(row, 3), 1)
  def action(p), do: Enum.at(p, 1)
  def x(p), do: Enum.at(p, 3)
  def y(p), do: Enum.at(p, 4)
  def percent(p), do: Enum.at(p, 7)
  def stock(p), do: Enum.at(p, 9)
  def jumps(p), do: Enum.at(p, 10)
  def hitstun(p), do: Enum.at(p, 12)
end

classify = fn path ->
  trace = Jason.decode!(File.read!(path))
  rows = trace["frames"]["rows"]
  if Enum.any?(rows, &(hd(&1) != 0)), do: raise("#{path}: delta rows are not supported")
  ps = Enum.map(rows, &Row.p2/1) |> List.to_tuple()
  n = tuple_size(ps)
  at = &elem(ps, &1)
  death = Enum.find(1..(n - 1)//1, fn i -> Row.stock(at.(i)) < Row.stock(at.(i - 1)) end)
  hits = Enum.filter(1..(n - 1)//1, fn i -> Row.percent(at.(i)) > Row.percent(at.(i - 1)) end)
  last_hit = if death, do: hits |> Enum.filter(&(&1 < death)) |> List.last(), else: List.last(hits)
  hitstun_starts = Enum.filter(1..(n - 1)//1, fn i -> Row.hitstun(at.(i)) > 0 and Row.hitstun(at.(i - 1)) == 0 end)
  window = if death, do: max(1, death - gap)..(death - 1)//1, else: 1..0//1
  current_rule = death != nil and Enum.any?(window, &(&1 in hits or &1 in hitstun_starts))
  percent_rule = death != nil and Enum.any?(window, &(&1 in hits))
  after_hit = if death && last_hit, do: (last_hit + 1)..(death - 1)//1, else: 1..0//1
  actionable = Enum.filter(after_hit, fn i -> p = at.(i); Row.hitstun(p) == 0 and Row.action(p) > 10 end)
  first_act = List.first(actionable)
  recoverable = first_act != nil and abs(Row.x(at.(first_act))) <= edge + 45.0 and Row.y(at.(first_act)) >= -70.0
  # A hit only earns credit if the victim was still in reach of the stage when it landed:
  # tagging a defender who is already 100+ units below the edge changes nothing.
  in_reach = fn p -> (Row.y(p) >= 0.0 and abs(Row.x(p)) < edge) or (abs(Row.x(p)) <= edge + 45.0 and Row.y(p) >= -70.0) end
  hit_in_reach = last_hit != nil and in_reach.(at.(last_hit - 1))
  p1 = fn i -> Enum.at(Enum.at(Enum.at(rows, i), 3), 0) end
  p1_off_frames = if death, do: Enum.count(max(1, death - 60)..(death - 1)//1, fn i -> q = p1.(i); abs(Row.x(q)) > edge - 5.0 or Row.y(q) < -5.0 end), else: 0
  p1_at_death = death && {Float.round(Row.x(p1.(death)) * 1.0, 1), Float.round(Row.y(p1.(death)) * 1.0, 1)}
  # The lip decision point (Bradley 2026-09-22): knockback or a slide carries the
  # defender off the stage edge while actionable. Doing nothing there grabs the
  # ledge; waiting then jump + up-B also makes it. A defender that spends an
  # action in the 15 f after leaving the lip (jump, aerial, special, air dodge)
  # chose its own death: SD. No lip moment (launched off in hitstun) leaves the
  # earlier kill / edgeguard verdict standing.
  spent_actions = MapSet.new(Enum.to_list(24..28) ++ Enum.to_list(44..74) ++ [236] ++ Enum.to_list(341..400))
  lip =
    if death && last_hit do
      Enum.find((last_hit + 1)..(death - 1)//1, fn i ->
        a = at.(i - 1); b = at.(i)
        Enum.at(a, 6) == 1 and abs(Row.x(a)) < edge and Enum.at(b, 6) == 0 and abs(Row.x(b)) >= edge - 6.0 and Row.hitstun(b) == 0
      end)
    end
  lip_action = lip && Enum.find_value((lip)..min(death - 1, lip + 15)//1, fn i -> a = Row.action(at.(i)); if MapSet.member?(spent_actions, a), do: {i - lip, a}, else: nil end)
  lip_spent = lip_action != nil
  # What ended the slide-off: the first of ledge grab (CliffCatch 252 / CliffWait
  # 253), landing back on stage, a committed action (jump / aerial / special /
  # air dodge), taking a hit, or death.
  slide_end =
    lip && Enum.find_value((lip)..death//1, fn i ->
      p = at.(i)
      cond do
        i == death -> {:died, i - lip}
        Row.action(p) in [252, 253] -> {:ledge, i - lip}
        Enum.at(p, 6) == 1 and abs(Row.x(p)) < edge -> {:landed, i - lip}
        MapSet.member?(spent_actions, Row.action(p)) -> {:committed, i - lip, Row.action(p)}
        i > lip and Row.percent(p) > Row.percent(at.(i - 1)) -> {:hit, i - lip}
        true -> nil
      end
    end)
  # Melee's own credit rule (ftCo x18c4_source_ply): the attacker keeps credit
  # until the victim is grounded and actionable for 60 f (ftCommonData.x814),
  # then the death is a suicide. A defender back on stage that long went off
  # on its own; one still airborne offstage when actionable is an edgeguard
  # setup; one never actionable again was killed by the hit.
  grounded_act = Enum.filter(actionable, fn i -> p = at.(i); Row.y(p) >= 0.0 and abs(Row.x(p)) < edge end)
  credit_expired = death != nil and grounded_act != [] and death - hd(grounded_act) > 60
  # Situation at the defender's decision point, separate from the outcome
  # (Bradley 2026-09-23): :slide_off = left the edge/platform actionable, limited
  # options that can be read (worse than neutral, better than a punish);
  # :launched = first actionable already airborne offstage; :on_stage = back to
  # neutral on the ground; :none = never actionable again.
  situation =
    cond do
      death == nil -> :none
      lip != nil -> :slide_off
      actionable == [] -> :none
      grounded_act != [] and hd(grounded_act) == first_act -> :on_stage
      first_act != nil -> :launched
      true -> :none
    end
  verdict =
    cond do
      death == nil -> :survived
      last_hit == nil or death - last_hit > gap or not hit_in_reach -> :sd
      actionable == [] -> :kill
      lip_spent -> :sd
      grounded_act != [] -> :sd
      recoverable -> :sd
      credit_expired -> :sd
      first_act != nil -> :edgeguard
      true -> :kill
    end
  %{
    file: Path.basename(path), death: death, last_hit: last_hit, gap_f: death && last_hit && death - last_hit,
    hits: length(hits), actionable_f: length(actionable), p1_off: p1_off_frames, p1_at_death: p1_at_death, lip: lip && death - lip, lip_action: lip_action, situation: situation, slide_end: slide_end,
    at_first_act: first_act && {Float.round(Row.x(at.(first_act)) * 1.0, 1), Float.round(Row.y(at.(first_act)) * 1.0, 1), Row.jumps(at.(first_act))},
    current: current_rule, percent: percent_rule, verdict: verdict, label: labels[Path.basename(path)]
  }
end

files =
  summary["generations"] |> Enum.map(& &1["trace"]) |> Enum.reject(&is_nil/1) |> Kernel.++([summary["best"]["trace"]]) |> Enum.uniq()
rows = Enum.map(files, &classify.(Path.join(run, &1)))
pad = fn s, w -> String.pad_trailing(to_string(s), w) end
IO.puts(pad.("trace", 22) <> pad.("death", 7) <> pad.("lastHit", 9) <> pad.("gap", 6) <> pad.("hits", 6) <> pad.("act.f", 7) <> pad.("first actionable (x,y,jumps)", 30) <> pad.("lip-f", 7) <> pad.("slide-off ends by", 22) <> pad.("situation", 11) <> pad.("verdict", 11) <> "label")
for r <- rows do
  IO.puts(pad.(r.file, 22) <> pad.(r.death || "-", 7) <> pad.(r.last_hit || "-", 9) <> pad.(r.gap_f || "-", 6) <> pad.(r.hits, 6) <> pad.(r.actionable_f, 7) <> pad.(inspect(r.at_first_act), 30) <> pad.(r.lip || "-", 7) <> pad.(inspect(r.slide_end), 22) <> pad.(r.situation, 11) <> pad.(r.verdict, 11) <> to_string(r.label || ""))
end
IO.puts("\nverdict counts: " <> inspect(Enum.frequencies_by(rows, & &1.verdict)))
labelled = Enum.reject(rows, &is_nil(&1.label))
if labelled != [] do
  IO.puts("labels: " <> inspect(Enum.frequencies_by(labelled, & &1.label)))
  agree = fn name, pred -> n = Enum.count(labelled, fn r -> pred.(r) == (r.label == "kill") end); IO.puts("  #{name}: #{n}/#{length(labelled)} agree with 'kill' vs not") end
  agree.("current (hit or hitstun within gap)", & &1.current)
  agree.("percent-only within gap", & &1.percent)
  agree.("Melee credit rule (verdict :kill)", &(&1.verdict == :kill))
  IO.puts("  confusion (verdict x label): " <> inspect(Enum.frequencies_by(labelled, &{&1.verdict, &1.label})))
  IO.puts("  situation x label: " <> inspect(Enum.frequencies_by(labelled, &{&1.situation, &1.label})))
end
