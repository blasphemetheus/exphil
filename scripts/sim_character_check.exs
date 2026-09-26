# Sim admission probe for a character that has never been rolled in the sim.
#
#   devenv shell -- mix run scripts/sim_character_check.exs \
#     --policy checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin \
#     --character mewtwo --envs 2 --frames 180
#
# Answers, before any PPO time is spent: does the sim admit this character on
# both ports, does the prior step inside it, what feature width do the trunk
# features have, do rewards/terminals actually fire, and is the character id
# the sim reports the one we asked for. It deliberately reuses
# `ExPhil.Sim.Critic.collect/5` — the exact call PPO makes — so a pass here
# means the PPO path is exercised, not a parallel one.

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.SimBatch
alias ExPhil.Sim.{Critic, Env}
alias ExPhil.Training.Output

{opts, _, invalid} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, character: :string, stage: :string, envs: :integer,
             frames: :integer, seed: :integer]
  )

if invalid != [], do: raise("Invalid options: #{inspect(invalid)}")
policy = opts[:policy] || raise("--policy required")
character = opts[:character] || raise("--character required")
stage = opts[:stage] || "final_destination"
n = opts[:envs] || 2
frames = opts[:frames] || 180
seed = opts[:seed] || 924
want_id = SimBatch.character_id(character)

Output.banner("Sim admission — #{character} ditto on #{stage}")
Output.config([
  {"Policy", policy}, {"Character", "#{character} (sim id #{want_id})"},
  {"Stage", "#{stage} (id #{SimBatch.stage_id(stage)})"},
  {"Envs × frames", "#{n} × #{frames}"}, {"Seed", seed}
])

players = [%{character: character, costume: 1}, %{character: character, costume: 0}]
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: seed)
Output.success("sim started")

agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed,
              frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]

Output.puts("⏳ loading the prior on both ports (JIT on first use)…")
{:ok, a1} = Agent.start_link(agent_opts)
{:ok, a2} = Agent.start_link(agent_opts)
for a <- [a1, a2], do: {:ok, _} = Agent.warmup(a)
Output.success("prior loaded and warm")

# --- what the sim actually reports for the players we asked for
{:ok, _} = Env.reset(sim)
{:ok, _, _} = Env.observe(sim)
{:ok, [gs | _]} = Env.frames(sim)

seen =
  gs.players
  |> Enum.sort_by(fn {port, _} -> port end)
  |> Enum.map(fn {port, p} -> {port, p.character, p.stock, p.percent, Float.round(p.x * 1.0, 1), Float.round(p.y * 1.0, 1)} end)

Output.puts("frame #{gs.frame}, stage #{inspect(gs.stage)}")

for {port, char, stock, pct, x, y} <- seen do
  Output.puts("  port #{port}: character #{inspect(char)}  stock #{stock}  #{pct}%  at (#{x}, #{y})")
end

# `character` is the game's INTERNAL fighter kind as an integer, in the same
# space the frame parser and the embedding use (see `SimState` moduledoc): the
# sim mapping is the identity, so this must equal the id we asked for. The
# external/CSS id space (Mewtwo 10) never appears here.
chars = Enum.map(seen, fn {_, c, _, _, _, _} -> c end)
char_ok = Enum.all?(chars, &(&1 == want_id))

if char_ok,
  do: Output.success("both ports report internal character id #{want_id} (#{character})"),
  else: Output.error("port characters #{inspect(chars)} != #{want_id} — the name mapping is wrong")

# --- roll it, exactly the way the critic/PPO collector does
t0 = System.monotonic_time(:millisecond)
roll = Critic.collect(sim, {a1, a2}, n, frames)
ms = System.monotonic_time(:millisecond) - t0

# rewards/dones/features come back as Nx tensors ([n, frames] and [n, frames, d])
num = fn t -> Nx.to_number(t) end
nonzero = num.(Nx.sum(Nx.as_type(Nx.greater(Nx.abs(roll.rewards), 1.0e-9), :s64)))
stockish = num.(Nx.sum(Nx.as_type(Nx.greater(Nx.abs(roll.rewards), 0.5), :s64)))
finite_feats = num.(Nx.all(Nx.is_nan(roll.features) |> Nx.logical_not())) == 1 and
                 num.(Nx.all(Nx.is_infinity(roll.features) |> Nx.logical_not())) == 1
rolled = n * frames

Output.puts("")
Output.puts("feature width d = #{roll.d}  (features #{inspect(Nx.shape(roll.features))})")
Output.puts("frames rolled   = #{rolled} (#{n}×#{frames}) in #{ms} ms  (#{Float.round(rolled * 1000 / max(1, ms), 0)} env-frames/s)")
Output.puts("reward nonzero  = #{nonzero} frames, of which |r|>0.5 (stock events) = #{stockish}")
Output.puts("reward sum      = #{Float.round(num.(Nx.sum(roll.rewards)), 4)}   min #{Float.round(num.(Nx.reduce_min(roll.rewards)), 3)}  max #{Float.round(num.(Nx.reduce_max(roll.rewards)), 3)}")
Output.puts("per-env totals  = #{inspect(Enum.map(roll.total_reward, &Float.round(&1, 3)))}")
Output.puts("terminals       = #{num.(Nx.sum(Nx.as_type(roll.dones, :s64)))} done frames")

checks = [
  {"both ports are #{character} (internal id #{want_id})", char_ok},
  {"feature width > 0", roll.d > 0},
  {"features are finite numbers", finite_feats},
  {"the game moves (some nonzero reward)", nonzero > 0}
]

Output.puts("")
for {name, ok} <- checks, do: if(ok, do: Output.success(name), else: Output.error(name))

Env.stop(sim)

if Enum.all?(checks, &elem(&1, 1)) do
  Output.success("ADMITTED — #{character} is safe to roll for critic fitting and PPO")
else
  Output.error("NOT ADMITTED — fix the failures above before spending PPO time")
  System.halt(1)
end
