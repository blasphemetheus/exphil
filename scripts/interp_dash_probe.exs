# Dash probe — is the missing dash a DECODE artifact or a LEARNED gap?
#
# A1 (eval_runs/0829_situation_hist) found the bot in DASHING on 0.1% of
# frames vs the expert's 9–12%, with zero dash-dances. Two candidate causes:
#   (a) decode: the main-stick head never puts enough mass on FULL TILT for a
#       single-frame flick (a dash needs |x| >= ~0.8 on the frame the stick
#       leaves neutral), so sampling ramps instead of flicking;
#   (b) learned/state: the head CAN produce full tilt, but the bot is never
#       in the standing states from which the expert dashes.
#
# Test: take expert frames where the expert INITIATES a dash on the next
# frame (the state is exactly the one the expert dashed from), give the
# policy the same 60-frame history, run ONE forward, and read the main-x
# head directly: probability mass on the full-tilt buckets at T=0.5 and
# T=1.0, argmax bucket, and the sampled flick rate at the deploy decode.
# Control: expert frames where the expert stays STANDING next frame.
#
#   (a) is true if mass on full tilt at dash-initiation states is low
#       (<< the expert's ~100%), even at T=1.0.
#   (b) is true if the mass is high — then the fix is upstream (training /
#       the states the bot gets itself into), not the decode.
#
# Sticks in ControllerState are 0..1 (bucket/16); full tilt = <= 0.1 or
# >= 0.9 (buckets 0-1, 15-16 of 17). The expert's own stick at the same
# frames is reported as the empirical zone.
#
#   mix run scripts/interp_dash_probe.exs --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --limit-files 20 \
#     --limit-frames 400 --n 16 --out eval_runs/0829_dash_probe/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, port: :integer, n: :integer, temperature: :float,
             limit_frames: :integer, limit_files: :integer, out: :string, seed: :integer]
  )

policy_path = opts[:policy] || raise("--policy required")
glob = opts[:replays] || raise("--replays required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
n_samples = opts[:n] || 16
temperature = opts[:temperature] || 0.5
limit_frames = opts[:limit_frames] || 400
limit_files = opts[:limit_files] || 20

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("Dash probe — decode artifact or learned gap?")
Output.config([{"Policy", Path.basename(policy_path)}, {"Replays", "#{length(files)} files"},
               {"Port", port}, {"n samples", n_samples}, {"Deploy main T", temperature},
               {"Frames per set", limit_frames}])


dashing = 20
standing = 14
walking = [15, 16, 17]

# ---- select frames ---------------------------------------------------------
{init, stand, frames_by_path} =
  Enum.reduce(files, {[], [], %{}}, fn path, {ia, sa, bp} ->
    case Peppi.parse(Path.expand(path)) do
      {:ok, replay} ->
        frames =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
          |> Enum.reject(&(&1.game_state.frame < 0))

        arr = List.to_tuple(frames)
        n = tuple_size(arr)
        act = fn i -> trunc(elem(arr, i).game_state.players[port].action || 0) end
        ground = fn i -> elem(arr, i).game_state.players[port].on_ground end

        inits =
          for i <- 60..(n - 2), act.(i + 1) == dashing and act.(i) != dashing and
              (act.(i) == standing or act.(i) in walking or act.(i) == 18) and ground.(i),
              do: {path, i}

        stands =
          for i <- 60..(n - 2), act.(i) == standing and act.(i + 1) == standing and ground.(i),
              do: {path, i}

        {[inits | ia], [stands | sa], Map.put(bp, path, arr)}

      _ ->
        {ia, sa, bp}
    end
  end)

spread = fn list, k ->
  list = List.flatten(list)
  if length(list) > k, do: list |> Enum.take_every(max(div(length(list), k), 1)) |> Enum.take(k), else: list
end

init_sel = spread.(init, limit_frames)
stand_sel = spread.(stand, limit_frames)
Output.puts("dash-initiation frames: #{length(List.flatten(init))} -> #{length(init_sel)}; standing-hold frames: #{length(List.flatten(stand))} -> #{length(stand_sel)}")

# ---- agent -----------------------------------------------------------------
{:ok, agent} =
  Agent.start_link(policy_path: policy_path, deterministic: false,
    temperature: %{buttons: 0.5, main: temperature, c: temperature, shoulder: temperature}, delay_id: 0)

Agent.warmup(agent)
cfg = Agent.get_config(agent)
window = if cfg.temporal, do: cfg.window_size || 60, else: 0

full = [0, 1, 15, 16]
mid = [2, 3, 4, 5, 11, 12, 13, 14]

zone_of = fn x01 ->
  cond do
    x01 <= 0.1 or x01 >= 0.9 -> :full
    x01 <= 0.35 or x01 >= 0.65 -> :mid
    true -> :center
  end
end

mass = fn logits, t ->
  p = logits |> Nx.squeeze() |> Nx.divide(t) |> then(&Nx.exp(Nx.subtract(&1, Nx.reduce_max(&1)))) |> then(&Nx.divide(&1, Nx.sum(&1))) |> Nx.to_flat_list()
  %{full: Enum.sum(Enum.map(full, &Enum.at(p, &1))), mid: Enum.sum(Enum.map(mid, &Enum.at(p, &1))),
    argmax_full: Enum.max_by(Enum.with_index(p), fn {v, _} -> v end) |> elem(1) |> then(&(&1 in full))}
end

rng = Nx.Random.key(opts[:seed] || 829)

probe = fn selected, label ->
  {rows, _} =
    selected
    |> Enum.with_index(1)
    |> Enum.map_reduce(rng, fn {{path, i}, idx}, key ->
      if rem(idx, 50) == 0, do: Output.progress_bar(idx, length(selected), label: label)
      arr = frames_by_path[path]
      Agent.reset_buffer(agent)
      for h <- max(i - window, 0)..(i - 1)//1, do: Agent.get_controller(agent, elem(arr, h).game_state, player_port: port)
      f = elem(arr, i)

      case Agent.get_action_with_confidence(agent, f.game_state, player_port: port) do
        {:ok, action, _} ->
          lx = action.logits.main_x
          m05 = mass.(lx, 0.5)
          m10 = mass.(lx, 1.0)
          # sampled flick rate at the deploy main temperature
          scaled = lx |> Nx.squeeze() |> Nx.divide(temperature)
          {r, key} = Nx.Random.uniform(key, shape: {n_samples, Nx.size(scaled)})
          g = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(r, 1.0e-10)))))
          picks = Nx.argmax(Nx.add(Nx.new_axis(scaled, 0), g), axis: 1) |> Nx.to_flat_list()
          sampled_full = Enum.count(picks, &(&1 in full)) / n_samples
          expert_x = f.controller.main_stick.x
          {%{p_full_05: m05.full, p_full_10: m10.full, p_mid_05: m05.mid, argmax_full: m05.argmax_full,
             sampled_full: sampled_full, expert_zone: zone_of.(expert_x)}, key}

        _ ->
          {nil, key}
      end
    end)

  Output.progress_done()
  rows = Enum.reject(rows, &is_nil/1)
  n = length(rows)
  avg = fn k -> Enum.sum(Enum.map(rows, &Map.fetch!(&1, k))) / max(n, 1) end
  ez = Enum.frequencies_by(rows, & &1.expert_zone)
  %{n: n, p_full_05: avg.(:p_full_05), p_full_10: avg.(:p_full_10), p_mid_05: avg.(:p_mid_05),
    argmax_full: Enum.count(rows, & &1.argmax_full) / max(n, 1), sampled_full: avg.(:sampled_full),
    expert_full: Map.get(ez, :full, 0) / max(n, 1), expert_mid: Map.get(ez, :mid, 0) / max(n, 1)}
end

r_init = probe.(init_sel, "dash-init")
r_stand = probe.(stand_sel, "standing")

pct = fn v -> :erlang.float_to_binary(v * 100, decimals: 1) <> "%" end

table = """
| at frames where the expert… | n | expert stick full-tilt | policy mass on full tilt, T=0.5 | T=1.0 | argmax is full tilt | sampled flick rate (deploy T=#{temperature}, n=#{n_samples}) | policy mass mid-zone T=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| …initiates a DASH next frame | #{r_init.n} | #{pct.(r_init.expert_full)} | **#{pct.(r_init.p_full_05)}** | #{pct.(r_init.p_full_10)} | #{pct.(r_init.argmax_full)} | **#{pct.(r_init.sampled_full)}** | #{pct.(r_init.p_mid_05)} |
| …stays STANDING (control) | #{r_stand.n} | #{pct.(r_stand.expert_full)} | #{pct.(r_stand.p_full_05)} | #{pct.(r_stand.p_full_10)} | #{pct.(r_stand.argmax_full)} | #{pct.(r_stand.sampled_full)} | #{pct.(r_stand.p_mid_05)} |
"""

verdict =
  cond do
    r_init.p_full_10 < 0.25 ->
      "DECODE/HEAD: at the exact states the expert dashes from, the main-x head puts only #{pct.(r_init.p_full_10)} on full tilt even at T=1.0 (expert #{pct.(r_init.expert_full)}). The dash is not in the head's output; sampling cannot flick. Look at the training target (17-bucket labels, stride-5 windows) before anything else."

    r_init.sampled_full < 0.5 * r_init.p_full_10 ->
      "DECODE (temperature): the head has the mass (#{pct.(r_init.p_full_10)} at T=1.0) but the deploy decode samples a flick only #{pct.(r_init.sampled_full)} of the time — cooling the main stick is what removed the dash."

    true ->
      "LEARNED/STATE: the head produces full tilt at dash-initiation states (#{pct.(r_init.p_full_05)} mass, #{pct.(r_init.sampled_full)} sampled flicks vs expert #{pct.(r_init.expert_full)}). The missing dash is upstream of the decode — the bot does not get itself into these states (WAIT 0.4% of its frames). Training / behaviour question, not a decode one."
  end

report = """
# Dash probe — decode artifact or learned gap?

Policy `#{Path.basename(policy_path)}`, #{length(files)} expert files, port #{port}, #{n_samples} samples/frame,
deploy main-stick T=#{temperature}. Full tilt = stick ≤0.1 or ≥0.9 (buckets 0–1, 15–16 of 17).

#{table}

**Verdict:** #{verdict}
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
