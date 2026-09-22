# Seed the sim at frame N of a real replay and report fidelity (the coach's
# "load the game" step). Steps the NIF batch from -123 with the replay's own
# pre-frame inputs, compares every post-frame with the replay, saves the
# state at N, and (optionally) writes a pool entry + a viewer trace of the
# 90 frames that actually followed in the replay.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/seed_replay.exs PATH.slp --frame 1200 \
#     [--out eval_runs/0921_seed/game1_f1200]

alias ExPhil.Sim.{Env, Seed, Trace}
alias ExPhil.Training.Output

{opts, [path | _], _} = OptionParser.parse(System.argv(), strict: [frame: :integer, out: :string, warm: :integer])
frame = opts[:frame] || 0

{:ok, meta} = ExPhil.Data.Peppi.metadata(path)
Output.banner("Seed the sim from a replay")
Output.config([{"Replay", Path.basename(path)}, {"Stage", meta.stage}, {"Players", Enum.map(meta.players, &"#{&1.port}: #{&1.character_name} c#{&1.costume} #{&1.player_type}")}, {"Seed", meta.random_seed}, {"Frames", meta.duration_frames}, {"Target frame", frame}])

t0 = System.monotonic_time(:millisecond)
{:ok, s} = Seed.from_replay(path, frame: frame, warm: opts[:warm] || 30)
ms = System.monotonic_time(:millisecond) - t0

Output.puts("seeded at frame #{s.frame} in #{ms} ms (#{Float.round((frame + 124) / max(ms, 1) * 1000, 0)} frames/s); savestate #{byte_size(s.blob)} bytes, history #{length(s.history)} frames")
Output.puts("state: p1 #{inspect(s.summary.p1)}  p2 #{inspect(s.summary.p2)}")

case s.divergence do
  nil -> Output.success("EXACT: sim matched the replay on every frame through #{s.frame} (x, y, action, percent, both ports)")
  {f, port, field, replay_v, sim_v} -> Output.warning("first divergence at frame #{f}: port #{port} #{field} replay #{inspect(replay_v)} vs sim #{inspect(sim_v)} — frames after that are the sim's continuation of the recorded inputs, not the replay")
  other -> Output.warning("divergence: #{inspect(other)}")
end

if out = opts[:out] do
  File.mkdir_p!(out)
  File.write!(Path.join(out, "seed.term"), :erlang.term_to_binary(%{blob: s.blob, frame: s.frame, history: s.history, summary: s.summary, players: s.players, stage: s.stage, replay: path, divergence: s.divergence}, [:compressed]))
  # what actually happened next in the replay, as a viewer trace (Fox = the subject port if any)
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)
  after_frames = replay.frames |> Enum.filter(&(&1.frame_number >= s.frame and &1.frame_number < s.frame + 90)) |> Enum.sort_by(& &1.frame_number)
  states = Enum.map(after_frames, fn f -> %{frame: f.frame_number, players: Map.new(f.players, fn {port, p} -> {port, Map.merge(p, %{stock: p.stock, shield_strength: p.shield_strength, jumps_left: p.jumps_left, hitstun_frames_left: round(p.hitstun_frames_left || 0), action_frame: round(p.action_frame || 0)})} end)} end)
  chars = Enum.map(s.players, fn p -> Map.get(%{"fox" => 1, "falco" => 22, "marth" => 18, "mewtwo" => 16, "sheik" => 7, "captainfalcon" => 2, "peach" => 9, "jigglypuff" => 15, "samus" => 13, "ganondorf" => 25, "link" => 6, "zelda" => 19, "iceclimbers" => 10, "mario" => 0, "luigi" => 17, "drmario" => 21, "pikachu" => 12, "yoshi" => 14, "donkeykong" => 3, "kirby" => 4, "bowser" => 5, "ness" => 8, "younglink" => 20, "roy" => 26, "pichu" => 23, "gameandwatch" => 24, "mrgamewatch" => 24}, p.character, 1) end)
  Trace.from_game_states(states, chars: chars, stage: s.stage, label: "replay #{Path.basename(path)} from frame #{s.frame}") |> Trace.write!(Path.join(out, "replay_from_#{s.frame}.msltrace.json"))
  Output.success("wrote #{out}/seed.term and replay_from_#{s.frame}.msltrace.json")
end

Env.stop(s.sim)
