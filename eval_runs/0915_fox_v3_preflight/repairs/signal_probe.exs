alias ExPhil.Training.{Imitation, TrainingState}
alias ExPhil.Training.Imitation.TrainLoop
alias ExPhil.Training.Callbacks.GracefulShutdown
opts = [embed_size: 16, temporal: true, bptt: true, backbone: :gru,
  hidden_size: 8, num_layers: 2, unroll: 4, batch_size: 2, dropout: 0.3, seed: 915,
  precision: :f32, head: :autoregressive]
t = Imitation.new(opts)
batch = %{states: Nx.broadcast(0.3, {2, 4, 16}),
  actions: %{buttons: Nx.broadcast(0, {2, 4, 8}), main_x: Nx.broadcast(8, {2, 4}),
    main_y: Nx.broadcast(8, {2, 4}), c_x: Nx.broadcast(8, {2, 4}), c_y: Nx.broadcast(8, {2, 4}),
    shoulder: Nx.broadcast(0, {2, 4})}, frame_weights: Nx.broadcast(1.0, {2, 4}),
  is_resetting: Nx.tensor([1, 1], type: :u8)}
carry = Nx.broadcast(0.0, {2, 2, 8})
{t, _, _} = TrainLoop.train_step_bptt(t, batch, carry)
path = Path.expand("eval_runs/0915_fox_v3_preflight/repairs/signal_model.axon")
state = %TrainingState{trainer: t, opts: [checkpoint: path], step: 1, epoch: 1}
{:cont, state, cb} = GracefulShutdown.on_train_begin(state, GracefulShutdown.init([]))
{_, 0} = System.cmd("kill", ["-TERM", System.pid()])
Enum.reduce_while(1..100, nil, fn _, _ ->
  if Agent.get(cb.agent_pid, & &1.interrupted), do: {:halt, :ok},
    else: (Process.sleep(10); {:cont, nil})
end)
{:halt, state, cb} = GracefulShutdown.on_batch_end(state, cb)
{:cont, _, cb} = GracefulShutdown.on_train_end(state, cb)
true = cb.agent_pid == nil and cb.signal_id == nil
saved = String.replace(path, ".axon", "_interrupt.axon")
{:ok, resumed} = Imitation.load_checkpoint(Imitation.new(opts), saved)
{next, a, _} = TrainLoop.train_step_bptt(t, batch, carry)
{loaded_next, b, _} = TrainLoop.train_step_bptt(resumed, batch, carry)
true = Nx.to_number(a.loss) == Nx.to_number(b.loss)
true = Nx.serialize(next.policy_params.data) == Nx.serialize(loaded_next.policy_params.data)
report = %{passed: true, signal: "SIGTERM", saved_step: resumed.step,
  resumed_step: loaded_next.step, next_update_matches: true, checkpoint: saved}
File.write!(Path.join(Path.dirname(path), "signal_probe.json"), Jason.encode!(report, pretty: true))
IO.inspect(report)

GracefulShutdown.finish_shutdown()
