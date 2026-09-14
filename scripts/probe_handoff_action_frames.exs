alias ExPhil.Agents.Agent, as: PolicyAgent
alias ExPhil.Training.{Checkpoint, Data, Labels, RecordedFrames, Utils}
alias ExPhil.Data.{ActionFrameConvention, Peppi}
alias ExPhil.Eval.TeacherFit
alias ExPhil.Networks.Policy

{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [policy: :string, scores: :string, out: :string])

path = Keyword.fetch!(opts, :policy)
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
scores = opts[:scores] |> File.read!() |> Jason.decode!()
{:ok, export} = Checkpoint.load_policy(path)

{_, predict} =
  Policy.build_temporal(Map.to_list(export.config)) |> Utils.build_compiled(mode: :inference)

params = Utils.ensure_model_state(export.params)
heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

for module <- [
      ExPhil.Bridge.GameState,
      ExPhil.Bridge.Player,
      ExPhil.Bridge.ControllerState,
      ExPhil.Bridge.Projectile
    ],
    do: Code.ensure_loaded!(module)

teachers =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(&RecordedFrames.load!/1)
  |> Map.new(&{hd(&1).game_state.frame, &1})

results =
  scores["runs"]
  |> Enum.uniq_by(& &1["frame"])
  |> Enum.map(fn run ->
    [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    frame = Peppi.to_training_frames(replay) |> Enum.find(&(&1.game_state.frame == run["frame"]))
    target = teachers[run["frame"]] |> Labels.at_delay(2) |> hd() |> Map.put(:delay_id, 2)
    dataset = Data.from_frame_lists([[target]])

    dataset =
      %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
      |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

    [batch] =
      Data.batched_sequences(dataset, lazy: true, window_size: 16, shuffle: false)
      |> Enum.to_list()

    target_row =
      Map.new(heads, fn head ->
        {head, batch.actions[head] |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list() |> hd()}
      end)

    raw_players =
      Map.new(frame.game_state.players, fn {port, player} ->
        action_frame =
          if port == 1,
            do: hd(run["policy_inputs"])["action_frame"],
            else: ActionFrameConvention.parsed_to_live(player.action, player.action_frame)

        {port, %{player | action_frame: action_frame}}
      end)

    raw_state = %{frame.game_state | players: raw_players}

    variants =
      Map.new([:parsed, :live], fn convention ->
        {:ok, agent} =
          PolicyAgent.start_link(
            policy_path: path,
            harness: :scenario_suite,
            reaction_delay: 2,
            harness_knob: 2,
            af_convention: convention
          )

        PolicyAgent.observe(agent, raw_state, target.controller, player_port: 1)
        [embedded] = :sys.get_state(agent).frame_buffer |> :queue.to_list()

        states =
          embedded
          |> Nx.reshape({1, 1, export.config.embed_size})
          |> Nx.broadcast({1, 16, export.config.embed_size})

        logits =
          predict.(
            params,
            ExPhil.Training.Imitation.Loss.policy_forward_inputs(
              :autoregressive,
              true,
              states,
              batch.actions
            )
          )

        output =
          logits
          |> Tuple.to_list()
          |> Enum.map(&(Nx.backend_copy(&1, Nx.BinaryBackend) |> Nx.to_list() |> hd()))

        difference =
          Nx.subtract(states, batch.states) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        GenServer.stop(agent)

        {convention,
         %{
           max_embedding_difference_from_teacher: difference,
           fit: TeacherFit.row(Map.new(Enum.zip(heads, output)), target_row)
         }}
      end)

    %{handoff: run["frame"], variants: variants}
  end)

File.write!(
  out,
  Jason.encode!(
    %{
      policy: path,
      scope:
        "first cold frame only, replay-reconstructed state; P1 raw AF from live trace, P2 reconstructed by convention inverse; no new live run",
      cases: results
    },
    pretty: true
  ),
  [:exclusive]
)
