# Closed-loop button simulation over the boundary map (INIT_FORENSICS
# option 1 analysis). The (prev -> B output) map measured by
# probe_crouch_boundary.exs defines closed-loop dynamics on the button
# channel alone: at SquatWait af t with prev-state s in {no_b, held},
# deterministic output b(t) = logit(t, s) > 0; a shine needs an EDGE
# (b true while s = no_b); next prev-state = b(t). Walking af 1..40 from
# both entry prevs counts edges — an offline prediction of "does this seed
# escape the basin under deterministic decode," per seed, in microseconds.
#
# Fixed-point taxonomy this exposes:
#   hold-B absorber   - output true under held  -> B held forever, no edge
#   silent absorber   - output false under no_b -> never presses
#   edge cycle        - presses on no_b, releases on held -> multishine
#
# Usage: mix run scripts/analyze_boundary_map.exs [--map eval_runs/interp/boundary_map.json]

{opts, _, _} = OptionParser.parse(System.argv(), strict: [map: :string])
map_path = opts[:map] || "eval_runs/interp/boundary_map.json"

data = File.read!(map_path) |> Jason.decode!()

live = %{
  "ms_crouch_a" => {:escape, "99-129/min c19-22"},
  "ms_crouch_b" => {:escape, "68/min c2"},
  "ms_crouch_c" => {:escape, "95-99/min c8-11"},
  "ms_crouch_d" => {:escape, "86-90/min c2"},
  "ms_crouch_e" => {:fail, "1-7/min"},
  "ms_crouch_f" => {:fail, "3-19/min"},
  "ms_crouch_g" => {:fail, "0/min dead"},
  "ms_crouch_h" => {:fail, "5-7/min"},
  "ms_crouch_i" => {:escape, "66-74/min c3"},
  "ms_crouch_j" => {:fail, "2-4/min"},
  "ms_crouch_k" => {:escape, "69-81/min c3-4"},
  "ms_crouch_l" => {:fail, "13-24/min oscillating"}
}

logit_by_af = fn seed_data, variant ->
  seed_data[variant]["rows"]
  |> Enum.filter(&(&1["state"] == "squat_wait"))
  |> Map.new(&{&1["af"], &1["b_logit"]})
end

simulate = fn no_b_map, held_map, start_state ->
  afs = no_b_map |> Map.keys() |> Enum.sort()

  {edges, _} =
    Enum.reduce(afs, {0, start_state}, fn af, {edges, s} ->
      logit = if s == :no_b, do: no_b_map[af], else: held_map[af]
      b = (logit || 0.0) > 0.0
      edge = if b and s == :no_b, do: 1, else: 0
      {edges + edge, if(b, do: :held, else: :no_b)}
    end)

  edges
end

IO.puts(
    String.pad_trailing("seed", 13) <>
      String.pad_trailing("edges(noB/held start)", 22) <>
      String.pad_trailing("predicted", 11) <> "live outcome"
)

rows =
  data
  |> Enum.sort_by(&elem(&1, 0))
  |> Enum.map(fn {seed, sd} ->
    no_b = logit_by_af.(sd, "live_absorbed")
    held = logit_by_af.(sd, "live_held")
    e1 = simulate.(no_b, held, :no_b)
    e2 = simulate.(no_b, held, :held)
    predicted = if min(e1, e2) > 0, do: :escape, else: :fail
    {truth, desc} = live[seed]
    match = if predicted == truth, do: "", else: "   <-- MISMATCH"

    IO.puts(
      String.pad_trailing(seed, 13) <>
        String.pad_trailing("#{e1} / #{e2}", 22) <>
        String.pad_trailing("#{predicted}", 11) <> "#{truth} (#{desc})#{match}"
    )

    predicted == truth
  end)

acc = Enum.count(rows, & &1) / length(rows)
IO.puts("\nClassifier accuracy vs live outcome: #{Float.round(acc * 100, 1)}% (#{Enum.count(rows, & &1)}/#{length(rows)})")
