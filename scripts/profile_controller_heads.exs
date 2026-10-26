# Isolate trained head/sampling cost on real recurrent features. Run through
# run_recurrent_validation.exs; this does not change production sampling code.
alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Networks.Policy
alias ExPhil.Training.Output

policies = [
  {"min_gru", "checkpoints/coh_prev_q_ck4/model_best_policy.bin",
   "/home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp", "fox"},
  {"mamba2", "checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin",
   "replays/greg/mewtwo/Game_20240222T222930.slp", "mewtwo"}
]

results =
  for {name, path, replay_path, character} <- policies, rows <- [1, 8] do
    {:ok, export} = ExPhil.Training.load_policy(path)
    {:ok, metadata} = Peppi.metadata(replay_path)
    own = Enum.find(metadata.players, &(String.downcase(&1.character_name || "") == character))
    opp = Enum.find(metadata.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(replay_path, player_port: own.port)

    frames =
      replay
      |> Peppi.to_training_frames(
        player_port: own.port,
        opponent_port: opp.port,
        remap_ports: true
      )
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.dedup_by(& &1.game_state.frame)
      |> Enum.take(export.config.window_size + 1)

    {:ok, agent} =
      Agent.start_link(
        policy: export,
        deterministic: true,
        stateful_step: true,
        harness: :sync_runner,
        frame_delay: 0,
        reaction_delay: 0,
        af_convention: :parsed
      )

    :ok = Agent.batch_init(agent, rows)

    for frame <- Enum.drop(frames, -1) do
      {:ok, _} = Agent.batch_get_controllers(agent, List.duplicate(frame.game_state, rows))
    end

    {:ok, _, features} =
      Agent.batch_get_controllers(
        agent,
        List.duplicate(List.last(frames).game_state, rows),
        return_features: true
      )

    features = Nx.backend_transfer(features, {EXLA.Backend, client: EXLA.Client.default_name()})
    state = :sys.get_state(agent)
    key = Nx.Random.key(20_261_026)

    samples =
      for {mode, opts} <- [
            deterministic: [deterministic: true],
            stochastic_fresh: [deterministic: false],
            stochastic_fixed_key: [deterministic: false, key: key]
          ] do
        run = fn ->
          action =
            if state.head == :autoregressive do
              Policy.sample_autoregressive_from_features(state.policy_params, features, opts)
            else
              Policy.sample(state.policy_params, state.heads_predict_fn, features, opts)
            end

          for k <- [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder],
              do: Nx.backend_transfer(action[k], Nx.BinaryBackend)
        end

        times =
          for i <- 1..110, reduce: [] do
            acc ->
              {us, _} = :timer.tc(run)
              if i > 10, do: [us | acc], else: acc
          end
          |> Enum.sort()

        result = %{
          mode: mode,
          median_us: Enum.at(times, 50),
          p95_us: Enum.at(times, 94),
          times_us: times
        }

        Output.puts("#{name} rows=#{rows}: #{inspect(Map.delete(result, :times_us))}")
        result
      end

    GenServer.stop(agent)
    %{policy: path, name: name, head: state.head, rows: rows, results: samples}
  end

[out] = System.argv()

File.write!(
  out,
  Jason.encode!(
    %{
      client: EXLA.Client.default_name(),
      results: results,
      note:
        "Trained heads and sampling plus six output transfers; fixed real features; no recurrence, embedding, controller conversion or Agent messaging. Ten warmups, 100 samples."
    },
    pretty: true
  )
)
