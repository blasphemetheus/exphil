# Per-STATE button confidence of a multishine policy on its own fixture
# (2026-09-10). The g19 record was an argmax artifact (437/min c438 argmax
# vs 45/min c3 at T=1.0), and both g20 arms wall at chain 2-4 under sampling
# at every rung/temperature: the trace says the cycle breaks at the B press
# that must be ISSUED on the last jumpsquat frame. This asks the policy
# directly: on each loop state, how sure is it about B and X?
#
#   mix run scripts/probe_ms_state_confidence.exs --policy checkpoints/X.bin \
#     [--fixture test/fixtures/replays/fox_multishine_closed_d1.slp] [--port 1]
#     [--k 32] [--limit 600] [--delay-id 0] [--temperature 1.0]
#
# Output: one row per {action, action_frame, grounded} key (loop states
# first): n frames, policy p(B)/p(X) from K live-path draws, and the
# fixture's ISSUED-input B/X rate on the same state at offsets 0..3 (the
# policy emits for t+offset; g20 lands at offset ~2). A decisive state has
# p near 0 or 1; the chain breaks where p sits in the middle.

alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, fixture: :string, port: :integer, k: :integer, limit: :integer, delay_id: :integer, temperature: :float]
  )

policy_path = opts[:policy] || raise "--policy required"
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
port = opts[:port] || 1
k = opts[:k] || 32
limit = opts[:limit] || 600
temperature = opts[:temperature] || 1.0

{:ok, replay} = ExPhil.Data.Peppi.parse(fixture, player_port: port)

frames =
  replay
  |> ExPhil.Data.Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(limit)

{:ok, %{config: cfg}} = ExPhil.Training.Checkpoint.load_policy(policy_path)
axis_buckets = cfg[:axis_buckets] || 16

Output.banner("Multishine per-state confidence (live sampling path)")
Output.puts("policy: #{Path.basename(policy_path)}  fixture: #{Path.basename(fixture)} (#{length(frames)} frames)  K=#{k} T=#{temperature}")

{:ok, agent} =
  Agent.start_link(
    policy_path: policy_path,
    deterministic: false,
    temperature: temperature,
    delay_id: opts[:delay_id] || 0
  )

Agent.warmup(agent)

key_of = fn f ->
  p = f.game_state.players[1]
  {trunc(p.action || 0), trunc(p.action_frame || 0), p.on_ground == true}
end

arr = List.to_tuple(frames)
n = tuple_size(arr)

# policy draws per frame
rng = Nx.Random.key(2026)

{rows, _} =
  frames
  |> Enum.with_index()
  |> Enum.map_reduce(rng, fn {f, i}, key ->
    split = Nx.Random.split(key)
    {sub, key} = {split[0], split[1]}

    {pb, px} =
      case Agent.get_action_samples(agent, f.game_state, player_port: 1, n: k, key: sub) do
        {:ok, actions} ->
          cs = Enum.map(actions, &ExPhil.Networks.Policy.to_controller_state(&1, axis_buckets: axis_buckets))
          {Enum.count(cs, & &1.button_b) / k, Enum.count(cs, & &1.button_x) / k}

        _ ->
          # Independent head: the button head is a sigmoid per button, so the
          # probability is read directly from the logits (order a,b,x,y,z,l,r,d_up).
          case Agent.get_action_with_confidence(agent, f.game_state, player_port: 1) do
            {:ok, action, _conf} ->
              logits = action[:buttons_logits] || get_in(action, [:logits, :buttons])

              if logits do
                p = logits |> Nx.squeeze() |> Nx.divide(temperature) |> Nx.sigmoid() |> Nx.to_flat_list()
                {Enum.at(p, 1), Enum.at(p, 2)}
              else
                {nil, nil}
              end

            _ ->
              {nil, nil}
          end
      end

    if rem(i, 100) == 0, do: IO.write(:stderr, "\r  #{i}/#{n}\e[K")
    {{key_of.(f), i, pb, px}, key}
  end)

IO.write(:stderr, "\r\e[K")

issued = fn i, off ->
  j = i + off
  if j < n, do: elem(arr, j).controller, else: nil
end

groups = Enum.group_by(rows, fn {key, _, _, _} -> key end)

loop_keys = [{361, 1, true}, {361, 2, true}, {361, 3, true}, {24, 0, true}, {24, 1, true}, {24, 2, true}, {365, 1, false}, {365, 2, false}, {365, 3, false}, {366, 0, false}, {366, 1, false}]

ordered =
  (loop_keys |> Enum.filter(&Map.has_key?(groups, &1))) ++
    (groups |> Map.keys() |> Enum.reject(&(&1 in loop_keys)) |> Enum.sort_by(fn k -> -length(groups[k]) end) |> Enum.take(8))

fmt = fn v -> if v == nil, do: "   - ", else: String.pad_leading(:erlang.float_to_binary(v * 1.0, decimals: 2), 5) end

Output.puts("")
Output.puts("| state {action,af,ground} | n | policy p(B) | p(X) | fixture B@off0 | B@1 | B@2 | B@3 | X@0 | X@1 | X@2 | X@3 |")
Output.puts("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for key <- ordered do
  g = groups[key]
  valid = Enum.reject(g, fn {_, _, pb, _} -> pb == nil end)
  nn = length(valid)
  pb = if nn > 0, do: Enum.sum(Enum.map(valid, fn {_, _, b, _} -> b end)) / nn
  px = if nn > 0, do: Enum.sum(Enum.map(valid, fn {_, _, _, x} -> x end)) / nn

  rate = fn off, field ->
    vals = for {_, i, _, _} <- g, c = issued.(i, off), c != nil, do: if(Map.get(c, field), do: 1, else: 0)
    if vals == [], do: nil, else: Enum.sum(vals) / length(vals)
  end

  Output.puts(
    "| #{inspect(key)} | #{nn} | #{fmt.(pb)} | #{fmt.(px)} | #{fmt.(rate.(0, :button_b))} | #{fmt.(rate.(1, :button_b))} | #{fmt.(rate.(2, :button_b))} | #{fmt.(rate.(3, :button_b))} | #{fmt.(rate.(0, :button_x))} | #{fmt.(rate.(1, :button_x))} | #{fmt.(rate.(2, :button_x))} | #{fmt.(rate.(3, :button_x))} |"
  )
end

Output.puts("")
Output.puts("Reading: find the offset column whose fixture rates the policy tracks (g20 ~2). On the")
Output.puts("last-jumpsquat state the fixture's B@offset is ~1.0; a policy p(B) near 0.5 there is the chain wall.")
