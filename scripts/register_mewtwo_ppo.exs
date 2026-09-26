# Register the two human-tested September 24 heads without promoting either.
# CPU-only: devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/register_mewtwo_ppo.exs
alias ExPhil.Training.{Registry, Output}
{:ok, parent} = Registry.get("n8T_RBQtCMg")
report = "eval_runs/0924_mewtwo_ppo/eval/degeneracy_report.json"
for iteration <- [150, 300] do
  name = "mewtwo-gru-ppo-v1-i#{iteration}"
  case Registry.get(name) do
    {:ok, entry} -> Output.puts("Already registered #{entry.name}: #{entry.id}")
    {:error, _} ->
      policy = "checkpoints/mewtwo_ppo_v1_iter#{iteration}_policy.bin"
      head_path = "eval_runs/0924_mewtwo_ppo/v1/head_iter#{iteration}.bin"
      head = head_path |> File.read!() |> :erlang.binary_to_term()
      true = head.character == "mewtwo"
      true = Path.expand(head.policy) == Path.expand(parent.policy_path)
      true = head.iter == iteration
      artifacts = Enum.map([policy, head_path], fn p ->
        %{path: p, bytes: File.stat!(p).size,
          sha256: Base.encode16(:crypto.hash(:sha256, File.read!(p)), case: :lower)}
      end)
      verdict = if iteration == 150, do: "user_preferred_over_iter300; drift_warning; not_promoted",
        else: "human_reported_degenerate_roll_grab_backthrow; not_promoted"
      config = parent.training_config
        |> Map.put("training_method", "head_only_ppo")
        |> Map.put("provenance", %{kind: "policy", artifacts: artifacts,
          iteration: iteration, prior_policy: head.policy, parent_evidence: head_path,
          character: head.character, stage: head.stage, seed: head.seed,
          training_log: "eval_runs/0924_mewtwo_ppo/v1/log.json",
          drift_report: report, human_verdict_source: "docs/planning/HANDOFF_2026-09-25.md section 4.7"})
      selection = "eval_runs/0924_mewtwo_ppo/eval/sel_head_iter#{iteration}/summary.json"
      metrics = %{selection: selection |> File.read!() |> Jason.decode!() |> Map.drop(["rows"]),
        human_verdict: verdict, live_sd_report: "eval_runs/0925_mewtwo_review/iter#{iteration}_sd.md",
        promotion_status: "not_promoted", frozen_opponent_result_is_not_general_strength: true}
      metrics = if iteration == 150, do: Map.put(metrics, :test,
        "eval_runs/0924_mewtwo_ppo/eval/test/summary.json" |> File.read!() |> Jason.decode!() |> Map.drop(["rows"])), else: metrics
      {:ok, entry} = Registry.register(%{name: name, checkpoint_path: head_path, policy_path: policy,
        training_config: config, metrics: metrics, parent_id: parent.id,
        tags: ["mewtwo", "gru", "ppo", "head-only", "candidate", if(iteration == 150, do: "human-preferred", else: "degenerate-zoo")]})
      Output.success("Registered #{name}: #{entry.id}")
  end
end
