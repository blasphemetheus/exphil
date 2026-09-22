# Chess-review v0 of a real replay: eval bar (the prior's expected value from
# each moment, by self-play rollouts in the sim), what actually happened, and
# blunder marks where the actual outcome fell far below the expectation.
#
#   devenv shell -- mix run scripts/coach_review.exs PATH.slp --subject 2 \
#     --policy checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin \
#     --every 120 --horizon 240 --samples 16 --out eval_runs/0921_coach/g1
#
# Writes review.json (points + blunders), and for each blunder the actual
# replay window and the best sampled line as viewer traces.

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Coach, Trace}
alias ExPhil.Training.Output

{opts, [path | _], _} = OptionParser.parse(System.argv(), strict: [policy: :string, subject: :integer, every: :integer, horizon: :integer, samples: :integer, blunder: :float, out: :string, start: :integer, stop: :integer, temperature: :float])
policy = opts[:policy] || raise("--policy required")
subject = opts[:subject] || 1
out = opts[:out]

Output.banner("Coach review v0")
{:ok, meta} = ExPhil.Data.Peppi.metadata(path)
Output.config([{"Replay", Path.basename(path)}, {"Stage", meta.stage}, {"Players", Enum.map(meta.players, &"#{&1.port}: #{&1.character_name} (#{&1.player_type})")}, {"Subject port", subject}, {"Policy", policy}, {"Every", opts[:every] || 120}, {"Horizon", opts[:horizon] || 240}, {"Samples", opts[:samples] || 16}])

agent_opts = [policy_path: policy, deterministic: false, temperature: opts[:temperature] || 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]
Output.puts("⏳ loading two agents (JIT on first use)…")
{:ok, a1} = Agent.start_link(agent_opts)
{:ok, a2} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(a1)
{:ok, _} = Agent.warmup(a2)

t0 = System.monotonic_time(:millisecond)

review =
  Coach.review(path,
    subject: subject, every: opts[:every] || 120, horizon: opts[:horizon] || 240, samples: opts[:samples] || 16,
    blunder: opts[:blunder] || 0.5, agents: {a1, a2}, start: opts[:start] || 0, stop: opts[:stop],
    on_point: fn p ->
      bar = String.duplicate("█", round(max(0.0, min(2.0, p.expected + 1.0)) * 10))
      flag = if p.delta && p.delta < -(opts[:blunder] || 0.5), do: "  ◀ BLUNDER", else: ""
      Output.puts("f#{String.pad_leading(Integer.to_string(p.frame), 5)}  expected #{:io_lib.format("~6.2f", [p.expected])} ±#{:io_lib.format("~4.2f", [p.sd])}  actual #{if p.actual, do: :io_lib.format("~6.2f", [p.actual]), else: "   n/a"}  #{String.pad_trailing(bar, 20, "░")}#{flag}")
    end)

ms = System.monotonic_time(:millisecond) - t0
Output.puts("")
Output.puts("#{length(review.points)} points in #{Float.round(ms / 1000, 1)} s; divergence #{inspect(review.divergence)}")
Output.puts("blunders (actual − expected < −#{opts[:blunder] || 0.5}): #{Enum.map_join(review.blunders, ", ", &"f#{&1.frame} (#{Float.round(&1.delta, 2)})")}")

if out do
  File.mkdir_p!(out)
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)
  by_frame = Map.new(replay.frames, &{&1.frame_number, &1})
  chars = Enum.map(meta.players |> Enum.sort_by(& &1.port), fn p -> Trace.char_id(p.character_name) end)
  to_state = fn fr -> %{frame: fr.frame_number, players: Map.new(fr.players, fn {port, pl} -> {port, Map.merge(pl, %{shield_strength: pl.shield_strength, jumps_left: pl.jumps_left, hitstun_frames_left: round(pl.hitstun_frames_left || 0), action_frame: round(pl.action_frame || 0)})} end)} end

  # the whole game, playable in the viewer (frame index i = Slippi frame first_frame + i)
  all_states = replay.frames |> Enum.sort_by(& &1.frame_number) |> Enum.map(to_state)
  first_frame = hd(all_states).frame
  Trace.from_game_states(all_states, chars: chars, stage: meta.stage, label: "replay #{Path.basename(path)}") |> Trace.write!(Path.join(out, "game.msltrace.json"))

  rows =
    Enum.map(review.points, fn p ->
      files =
        for {kind, line} <- [{"best", p.lines.best}, {"typical", p.lines.typical}, {"worst", p.lines.worst}], into: %{} do
          name = "p#{p.frame}_#{kind}.msltrace.json"
          Trace.from_game_states(line.states, chars: chars, stage: meta.stage, label: "#{kind} continuation from f#{p.frame} (value #{Float.round(line.value, 2)})") |> Trace.write!(Path.join(out, name))
          {kind, %{file: name, value: line.value}}
        end

      actual_name = "p#{p.frame}_actual.msltrace.json"
      actual_states = for f <- p.frame..min(p.frame + review.horizon, review.frames), by_frame[f], do: to_state.(by_frame[f])
      if length(actual_states) > 1, do: Trace.from_game_states(actual_states, chars: chars, stage: meta.stage, label: "ACTUAL from f#{p.frame} (value #{p.actual && Float.round(p.actual, 2)})") |> Trace.write!(Path.join(out, actual_name))

      %{frame: p.frame, expected: p.expected, sd: p.sd, actual: p.actual, delta: p.delta, samples: p.samples, diverged: p.diverged?, blunder: not p.diverged? and p.delta != nil and p.delta < -(opts[:blunder] || 0.5),
        lines: Map.put(files, "actual", %{file: actual_name, value: p.actual}),
        subject: Map.take(p.state.players[subject], [:x, :y, :action, :percent, :stock]), opponent: Map.take(p.state.players[Coach.other_port(subject)], [:x, :y, :action, :percent, :stock])}
    end)

  File.write!(Path.join(out, "review.json"), Jason.encode!(%{replay: Path.basename(path), stage: meta.stage, players: Enum.map(meta.players |> Enum.sort_by(& &1.port), &%{port: &1.port, character: &1.character_name, type: &1.player_type}), subject: subject, horizon: review.horizon, every: review.every, samples: review.samples, first_frame: first_frame, last_frame: review.frames, game: "game.msltrace.json", points: rows, blunders: Enum.map(review.blunders, & &1.frame)}, pretty: true))
  Output.success("wrote #{out}/review.json, game.msltrace.json + #{4 * length(review.points)} continuation traces")
end
