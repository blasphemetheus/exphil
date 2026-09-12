defmodule ExPhil.Evaluation.BPTTTest do
  use ExUnit.Case, async: false

  alias ExPhil.Evaluation.{BPTT, Forward}
  alias ExPhil.Training.{Data, Imitation}
  alias ExPhil.Training.Imitation.TrainLoop

  defp trainer(head \\ :autoregressive) do
    Imitation.new(
      embed_size: 32,
      temporal: true,
      backbone: :gru,
      bptt: true,
      head: head,
      hidden_size: 12,
      num_layers: 1,
      unroll: 4,
      dropout: 0.0,
      precision: :f32,
      batch_size: 1
    )
  end

  defp batch(start, count, reset) do
    states = Nx.iota({1, count, 32}, type: :f32) |> Nx.add(start * 32) |> Nx.divide(100)

    actions =
      Map.new([:main_x, :main_y, :c_x, :c_y, :shoulder], &{&1, Nx.broadcast(0, {1, count})})

    %{
      states: states,
      actions: Map.put(actions, :buttons, Nx.broadcast(0.0, {1, count, 8})),
      is_resetting: Nx.tensor([reset], type: :u8),
      frame_weights: Nx.broadcast(1.0, {1, count})
    }
  end

  for head <- [:autoregressive, :independent] do
    test "#{head}: contiguous chunks equal a complete unroll, including the short tail" do
      trainer = trainer(unquote(head))
      evaluator = Forward.new(trainer.policy_params, trainer.config)
      {whole, _, _} = Forward.batch(evaluator, batch(0, 7, 1))
      {first, _, evaluator} = Forward.batch(evaluator, batch(0, 4, 1))
      {last, _, _} = Forward.batch(evaluator, batch(4, 3, 0))

      Enum.zip([Tuple.to_list(whole), Tuple.to_list(first), Tuple.to_list(last)])
      |> Enum.each(fn {expected, prefix, suffix} ->
        error =
          expected
          |> Nx.subtract(Nx.concatenate([prefix, suffix]))
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert error < 1.0e-5
      end)
    end
  end

  test "reset clears prior carry and teacher-forced inputs affect the autoregressive heads" do
    trainer = trainer()

    trainer =
      update_in(trainer.policy_params.data["ar_buttons_embed"]["kernel"], &Nx.add(&1, 0.05))

    evaluator = Forward.new(trainer.policy_params, trainer.config)
    {expected, _, _} = Forward.batch(evaluator, batch(4, 3, 1))
    {_, _, carried} = Forward.batch(evaluator, batch(0, 4, 1))
    {actual, _, _} = Forward.batch(carried, batch(4, 3, 1))
    assert Nx.to_flat_list(elem(actual, 1)) == Nx.to_flat_list(elem(expected, 1))

    source = batch(4, 3, 1)
    changed = put_in(source.actions.buttons, Nx.broadcast(1.0, {1, 3, 8}))
    {conditioned, _, _} = Forward.batch(evaluator, changed)
    assert Nx.to_flat_list(elem(conditioned, 1)) != Nx.to_flat_list(elem(expected, 1))
  end

  test "batching resets at gaps and replay boundaries and retains every frame exactly once" do
    frames =
      [0, 1, 2, 8, 9, 0, 1]
      |> Enum.map(&%{game_state: %{frame: &1}})

    dataset = %Data{frames: frames, size: 7, embedded_frames: Nx.iota({7, 32})}
    batches = BPTT.batches(dataset, 2) |> Enum.to_list()
    assert Enum.map(batches, &Nx.axis_size(&1.states, 1)) == [2, 1, 2, 2]
    assert Enum.map(batches, &Nx.to_number(Nx.squeeze(&1.is_resetting))) == [1, 0, 1, 1]

    assert Enum.flat_map(batches, &Nx.to_flat_list(&1.states)) ==
             Nx.to_flat_list(dataset.embedded_frames)
  end

  @tag :tmp_dir
  test "train, export, load without a sidecar, and evaluate", %{tmp_dir: directory} do
    trainer = trainer()

    {trainer, _, _} =
      TrainLoop.train_step_bptt(trainer, batch(0, 4, 1), Nx.broadcast(0.0, {1, 1, 12}))

    path = Path.join(directory, "standalone.bin")
    :ok = Imitation.export_policy(trainer, path)
    artifact = Forward.load!(path)
    assert artifact.config.bptt
    assert artifact.config.unroll == 4
    evaluator = Forward.new(artifact.params, artifact.config)
    result = BPTT.evaluate(evaluator, [batch(0, 4, 1), batch(4, 3, 0)])
    whole = BPTT.evaluate(evaluator, [batch(0, 7, 1)])
    assert result.frames == 7
    assert result.loss > 0
    assert_in_delta result.loss, whole.loss, 1.0e-5
    assert result.protocol == "bptt_teacher_forced_plain_ce_v1"
  end

  test "empty evaluation and unsupported BPTT backbones fail explicitly" do
    trainer = trainer()
    evaluator = Forward.new(trainer.policy_params, trainer.config)
    assert_raise ArgumentError, ~r/No frames/, fn -> BPTT.evaluate(evaluator, []) end

    assert_raise ArgumentError, ~r/gru/, fn ->
      Forward.new(trainer.policy_params, Map.put(trainer.config, :backbone, :lstm))
    end
  end

  test "windowed independent and autoregressive policies retain their input contracts" do
    for head <- [:independent, :autoregressive] do
      trainer =
        Imitation.new(
          embed_size: 32,
          temporal: true,
          backbone: :gru,
          head: head,
          hidden_size: 12,
          num_layers: 1,
          window_size: 4,
          dropout: 0.0
        )

      source = batch(0, 4, 1)

      actions =
        Map.new(source.actions, fn {key, tensor} ->
          {key, tensor |> Nx.slice_along_axis(3, 1, axis: 1) |> Nx.squeeze(axes: [1])}
        end)

      {logits, targets, _} =
        Forward.batch(Forward.new(trainer.policy_params, trainer.config), %{
          source
          | actions: actions
        })

      assert Nx.shape(elem(logits, 0)) == {1, 8}
      assert Nx.shape(targets.main_x) == {1}
    end
  end

  @tag :tmp_dir
  @tag :nif
  test "a standalone policy evaluates real replay frames in a fresh process", %{
    tmp_dir: directory
  } do
    trainer =
      Imitation.new(
        temporal: true,
        backbone: :gru,
        bptt: true,
        head: :autoregressive,
        hidden_size: 12,
        num_layers: 1,
        unroll: 80,
        dropout: 0.0,
        stage_internals: true,
        precision: :f32
      )

    path = Path.join(directory, "fixture_policy.bin")
    :ok = Imitation.export_policy(trainer, path)
    File.cp!("test/fixtures/replays/fox_multishine.slp", Path.join(directory, "fixture.slp"))
    code_paths = Enum.flat_map(:code.get_path(), &["-pa", List.to_string(&1)])

    source = """
    Application.put_env(:nx, :default_backend, Nx.BinaryBackend)
    Application.put_env(:exla, :clients, host: [platform: :host])
    Application.put_env(:exla, :default_client, :host)
    Application.ensure_all_started(:exla)
    Application.ensure_all_started(:axon)
    Application.ensure_all_started(:jason)
    Code.require_file("lib/exphil/evaluation/forward.ex")
    Code.require_file("lib/exphil/evaluation/bptt.ex")
    Code.require_file("scripts/eval_model.exs")
    """

    {output, status} =
      System.cmd(
        "elixir",
        ["--erl", "+S 2:2"] ++
          code_paths ++
          ["-e", source, "--policy", path, "--replays", directory, "--max-files", "1"],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert output =~ "teacher-forced plain CE="
  end

  test "BPTT diagnostics accept multi-input batches without failing" do
    trainer = trainer()

    state = %{
      trainer: trainer,
      pipeline: %{val_batches: [batch(0, 4, 1), batch(4, 4, 0)]},
      opts: []
    }

    ExUnit.CaptureIO.capture_io(fn ->
      callback = ExPhil.Training.Callbacks.Diagnostics.init([])
      {:cont, _, callback} = ExPhil.Training.Callbacks.Diagnostics.on_epoch_end(state, callback)
      assert length(callback.head_loss_history.buttons) == 1
    end)
  end
end
