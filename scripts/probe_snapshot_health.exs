# Snapshot health probe (2026-09-12, Astra's request on the g26 low-loss guard):
# is a snapshot the guard rejected as "collapse-suspect" numerically dead
# (GOTCHA #99: reported loss ~0 while the weights are garbage) or a genuine
# improvement that the absolute 1e-5 threshold can no longer tell apart from
# collapse now that item-14 labels make the pool fully fittable?
#
# Per policy, independent of the trainer's own loss:
#   1. parameter finiteness (any NaN/Inf anywhere in the params)
#   2. logit finiteness on a teacher-forced pass over the fixture
#   3. an independently recomputed loss: mean BCE of the button head's p(B),
#      p(X) against the fixture's issued input at the policy's reaction delay
#      (label offset k), over every loop-state frame
#   4. p(correct) on the same frames (the coverage map's baseline number)
#
#   mix run scripts/probe_snapshot_health.exs --policies "a.bin,b.bin" \
#     [--fixture test/fixtures/replays/fox_multishine_closed_d1.slp] [--delay-id 4] \
#     [--offset 4] [--limit 600] [--out path.json]

alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, fixture: :string, delay_id: :integer, offset: :integer, limit: :integer, out: :string]
  )

policies = (opts[:policies] || raise("--policies required")) |> String.split(",", trim: true)
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
delay_id = opts[:delay_id] || 4
offset = opts[:offset] || delay_id
limit = opts[:limit] || 600

{:ok, replay} = ExPhil.Data.Peppi.parse(fixture, player_port: 1)

frames =
  replay
  |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2, remap_ports: true)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(limit)

arr = List.to_tuple(frames)
n = tuple_size(arr)

loop_keys =
  MapSet.new([{361, 1, true}, {361, 2, true}, {361, 3, true}, {24, 0, true}, {24, 1, true}, {24, 2, true}, {365, 1, false}, {365, 2, false}, {365, 3, false}, {366, 0, false}, {366, 1, false}])

key_of = fn gs ->
  p = gs.players[1]
  {trunc(p.action || 0), trunc(p.action_frame || 0), p.on_ground == true}
end

# Walk any nested params structure and count non-finite tensor entries.
count_nonfinite = fn params ->
  walk = fn walk, v, acc ->
    cond do
      is_struct(v, Nx.Tensor) ->
        bad = v |> Nx.is_nan() |> Nx.logical_or(Nx.is_infinity(v)) |> Nx.sum() |> Nx.to_number()
        {elems, badc} = acc
        {elems + Nx.size(v), badc + bad}

      is_map(v) and not is_struct(v) ->
        Enum.reduce(Map.values(v), acc, fn x, a -> walk.(walk, x, a) end)

      is_struct(v, Axon.ModelState) ->
        walk.(walk, v.data, acc)

      is_list(v) ->
        Enum.reduce(v, acc, fn x, a -> walk.(walk, x, a) end)

      true ->
        acc
    end
  end

  walk.(walk, params, {0, 0})
end

eps = 1.0e-7
clip = fn p -> min(max(p, eps), 1.0 - eps) end
bce = fn p, y -> if y, do: -:math.log(clip.(p)), else: -:math.log(1.0 - clip.(p)) end

Output.banner("Snapshot health: params/logits finiteness + independent BCE (fixture, offset #{offset}, id #{delay_id})")

rows =
  Enum.map(policies, fn path ->
    {:ok, %{params: params}} = ExPhil.Training.Checkpoint.load_policy(path)
    {elems, nonfinite_params} = count_nonfinite.(params)

    {:ok, agent} = Agent.start_link(policy_path: path, deterministic: false, temperature: 1.0, reaction_delay: delay_id, allow_untrained_delay_id: true)
    Agent.warmup(agent)

    {stats, _} =
      frames
      |> Enum.with_index()
      |> Enum.reduce({%{n: 0, bce: 0.0, correct: 0.0, nonfinite_logits: 0}, nil}, fn {f, i}, {acc, _} ->
        probe? = MapSet.member?(loop_keys, key_of.(f.game_state)) and i + offset < n

        case Agent.observe(agent, f.game_state, f.controller, player_port: 1, probe: probe?) do
          {:ok, %{buttons: b}} ->
            pb = Enum.at(b, 1)
            px = Enum.at(b, 2)
            target = elem(arr, i + offset).controller
            tb = target.button_b == true
            tx = target.button_x == true

            finite? = Enum.all?(b, fn v -> is_float(v) and v == v and abs(v) != :infinity end)

            if finite? do
              pc = (if tb, do: pb, else: 1.0 - pb) * if tx, do: px, else: 1.0 - px
              {%{acc | n: acc.n + 1, bce: acc.bce + bce.(pb, tb) + bce.(px, tx), correct: acc.correct + pc}, nil}
            else
              {%{acc | nonfinite_logits: acc.nonfinite_logits + 1}, nil}
            end

          _ ->
            {acc, nil}
        end
      end)

    GenServer.stop(agent)

    row = %{
      policy: Path.basename(path),
      param_elements: elems,
      nonfinite_params: nonfinite_params,
      nonfinite_logit_frames: stats.nonfinite_logits,
      loop_frames: stats.n,
      bce_bx: if(stats.n > 0, do: stats.bce / stats.n, else: nil),
      p_correct: if(stats.n > 0, do: stats.correct / stats.n, else: nil)
    }

    Output.puts(
      "#{row.policy}: params #{elems} (#{nonfinite_params} non-finite) | logits non-finite on #{stats.nonfinite_logits} frames | " <>
        "loop frames #{stats.n} | BCE(B,X) #{if row.bce_bx, do: Float.round(row.bce_bx, 5)} | p(correct) #{if row.p_correct, do: Float.round(row.p_correct, 4)}"
    )

    row
  end)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{fixture: fixture, offset: offset, delay_id: delay_id, rows: rows}, pretty: true))
  Output.success("wrote #{out}")
end

Output.puts("")
Output.puts("Reading: a GOTCHA #99 collapse has finite-looking params but BCE >> the healthy snapshot's and")
Output.puts("p(correct) near chance; a genuine low-loss epoch has BCE at or below it. Neither replaces the live gates.")
