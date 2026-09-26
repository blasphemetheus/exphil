# Write the prior's OWN head out in the `head_iterN.bin` shape, so the same
# evaluator can be pointed at it to produce a prior-vs-prior control.
#
#   devenv shell -- mix run scripts/ppo_make_prior_head.exs \
#     --policy checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin \
#     --character mewtwo --out eval_runs/0924_mewtwo_ppo/prior_head.bin
#
# A control run should land near 50 %. Without one, a win rate is a number with
# no scale on it — the Fox gate only became believable once prior-vs-prior came
# back 49.0 %.

alias ExPhil.Sim.PPO
alias ExPhil.Training.Output

{opts, _, invalid} =
  OptionParser.parse(System.argv(), strict: [policy: :string, out: :string, character: :string])

if invalid != [], do: raise("Invalid options: #{inspect(invalid)}")
policy = opts[:policy] || raise("--policy required")
out = opts[:out] || raise("--out required")
character = opts[:character] || "fox"
_ = ExPhil.Bridge.SimBatch.character_id(character)

ar = PPO.head_params(policy)
d = Nx.axis_size(ar["ar_residual_proj"]["kernel"], 0)
File.mkdir_p!(Path.dirname(out))

File.write!(out, :erlang.term_to_binary(%{
  ar: ExPhil.Training.PPO.to_binary_backend(ar),
  iter: 0, policy: policy, d: d, character: character, control: true
}))

Output.success("prior head (#{map_size(ar)} ar_* layers, d=#{d}, #{character}) → #{out}")
