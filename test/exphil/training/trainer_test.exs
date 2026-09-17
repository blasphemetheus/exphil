defmodule ExPhil.Training.TrainerTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Trainer
  alias ExPhil.Training.Imitation

  defmodule StopAfterBatch do
    use ExPhil.Training.Callback
    def init(_), do: %{}
    def on_batch_end(state, cb), do: {:halt, state, cb}
    def on_epoch_end(_state, _cb), do: raise("interrupted epoch must not run validation")

    def on_train_end(state, cb),
      do: {:cont, %{state | meta: Map.put(state.meta, :cleaned_up, true)}, cb}
  end

  test "batch halt stops the whole fit and still runs cleanup" do
    action = %{
      buttons: Map.new([:a, :b, :x, :y, :z, :l, :r, :d_up], &{&1, false}),
      main_x: 8,
      main_y: 8,
      c_x: 8,
      c_y: 8,
      shoulder: 0
    }

    ds = %ExPhil.Training.Data{
      frames: for(i <- 0..7, do: %{game_state: %{frame: i}, action: action}),
      embedded_frames: Nx.broadcast(0.5, {8, 16}),
      size: 8
    }

    # A resumed trainer must keep absolute optimizer-step numbering in
    # callbacks/checkpoint names, even though this fit begins a fresh epoch.
    trainer = %{Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false) | step: 7}

    pipeline = %ExPhil.Training.Pipeline{
      streaming: false,
      train_dataset: ds,
      resolved_opts: [epochs: 3, batch_size: 2, temporal: false],
      estimated_batches: 4
    }

    {:ok, state} = Trainer.fit(trainer, pipeline, callbacks: [StopAfterBatch])
    assert state.step == 8
    assert state.trainer.step == 8
    assert state.epoch == 1
    assert state.history == []
    assert state.meta.cleaned_up
  end

  # A minimal callback that tracks what hooks were called
  defmodule TrackingCallback do
    use ExPhil.Training.Callback

    @impl true
    def init(_), do: %{hooks: []}

    @impl true
    def on_train_begin(state, cb), do: {:cont, state, %{cb | hooks: [:train_begin | cb.hooks]}}
    @impl true
    def on_epoch_begin(state, cb), do: {:cont, state, %{cb | hooks: [:epoch_begin | cb.hooks]}}
    @impl true
    def on_batch_end(state, cb), do: {:cont, state, %{cb | hooks: [:batch_end | cb.hooks]}}
    @impl true
    def on_epoch_end(state, cb), do: {:cont, state, %{cb | hooks: [:epoch_end | cb.hooks]}}
    @impl true
    def on_train_end(state, cb), do: {:cont, state, %{cb | hooks: [:train_end | cb.hooks]}}
  end

  defmodule StopAfterOneEpoch do
    use ExPhil.Training.Callback
    @impl true
    def init(_), do: %{}
    @impl true
    def on_epoch_end(state, cb), do: {:halt, %{state | halt: true}, cb}
  end

  describe "param_count/1" do
    test "counts parameters in a model state map" do
      # Simulate a trainer with nested param maps
      params = %Axon.ModelState{
        data: %{
          "layer1" => %{
            "kernel" => Nx.iota({10, 5}),
            "bias" => Nx.iota({5})
          },
          "layer2" => %{
            "kernel" => Nx.iota({5, 3}),
            "bias" => Nx.iota({3})
          }
        },
        parameters: MapSet.new(),
        state: %{}
      }

      trainer = %Imitation{
        policy_params: params,
        policy_model: nil,
        optimizer: nil,
        optimizer_state: nil,
        embed_config: nil,
        config: %{},
        step: 0,
        metrics: %{},
        predict_fn: nil,
        apply_updates_fn: nil,
        loss_and_grad_fn: nil,
        eval_loss_fn: nil,
        mixed_precision_state: nil
      }

      # 10*5 + 5 + 5*3 + 3 = 50 + 5 + 15 + 3 = 73
      assert Trainer.param_count(trainer) == 73
    end

    test "handles raw map params (not ModelState)" do
      params = %{
        "dense" => %{"kernel" => Nx.iota({4, 4}), "bias" => Nx.iota({4})}
      }

      trainer = %Imitation{
        policy_params: params,
        policy_model: nil,
        optimizer: nil,
        optimizer_state: nil,
        embed_config: nil,
        config: %{},
        step: 0,
        metrics: %{},
        predict_fn: nil,
        apply_updates_fn: nil,
        loss_and_grad_fn: nil,
        eval_loss_fn: nil,
        mixed_precision_state: nil
      }

      assert Trainer.param_count(trainer) == 20
    end
  end

  describe "check_nan!" do
    test "empty training data cannot be reported as a successful zero-loss epoch" do
      trainer = Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false)

      pipeline = %ExPhil.Training.Pipeline{
        streaming: false,
        train_dataset: %ExPhil.Training.Data{frames: [], size: 0},
        resolved_opts: [epochs: 2, batch_size: 2, temporal: false],
        estimated_batches: 0
      }

      assert_raise RuntimeError, ~r/produced no optimizer batches/, fn ->
        Trainer.fit(trainer, pipeline)
      end
    end

    test "a real non-finite training batch aborts the fit" do
      action = %{
        buttons: Map.new([:a, :b, :x, :y, :z, :l, :r, :d_up], &{&1, false}),
        main_x: 8,
        main_y: 8,
        c_x: 8,
        c_y: 8,
        shoulder: 0
      }

      ds = %ExPhil.Training.Data{
        frames: for(i <- 0..3, do: %{game_state: %{frame: i}, action: action}),
        embedded_frames: Nx.broadcast(:nan, {4, 16}),
        size: 4
      }

      trainer = Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false)

      pipeline = %ExPhil.Training.Pipeline{
        streaming: false,
        train_dataset: ds,
        resolved_opts: [epochs: 2, batch_size: 2, temporal: false],
        estimated_batches: 2
      }

      assert_raise RuntimeError, ~r/Training diverged: loss is nan/, fn ->
        Trainer.fit(trainer, pipeline)
      end
    end
  end
end
