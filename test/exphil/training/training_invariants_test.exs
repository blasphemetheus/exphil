defmodule ExPhil.Training.TrainingInvariantsTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.{TrainingState, Callback}

  @doc """
  Tests for training loop invariants — properties that should always hold
  regardless of model, data, or configuration.
  """

  describe "TrainingState invariants" do
    test "step increases monotonically" do
      states = [
        %TrainingState{step: 0},
        %TrainingState{step: 1},
        %TrainingState{step: 2},
        %TrainingState{step: 5}
      ]

      steps = Enum.map(states, & &1.step)
      assert steps == Enum.sort(steps), "Steps should be monotonically increasing"
    end

    test "epoch_losses accumulates correctly" do
      state = %TrainingState{epoch_losses: []}
      state = %{state | epoch_losses: [3.0 | state.epoch_losses]}
      state = %{state | epoch_losses: [2.5 | state.epoch_losses]}
      state = %{state | epoch_losses: [2.0 | state.epoch_losses]}

      avg = Enum.sum(state.epoch_losses) / length(state.epoch_losses)
      assert_in_delta avg, 2.5, 0.01
    end

    test "history grows by one entry per epoch" do
      state = %TrainingState{history: []}
      state = %{state | history: state.history ++ [%{epoch: 1, train_loss: 3.0}]}
      state = %{state | history: state.history ++ [%{epoch: 2, train_loss: 2.5}]}

      assert length(state.history) == 2
      assert hd(state.history).epoch == 1
      assert List.last(state.history).epoch == 2
    end

    test "best_val_loss only decreases" do
      losses = [3.0, 2.5, 2.8, 2.3, 2.4]
      best_history = Enum.scan(losses, nil, fn loss, best ->
        if best == nil or loss < best, do: loss, else: best
      end)

      assert best_history == [3.0, 2.5, 2.5, 2.3, 2.3]
    end

    test "meta map preserves across updates" do
      state = %TrainingState{}
      state = TrainingState.put_meta(state, :key1, "value1")
      state = TrainingState.put_meta(state, :key2, "value2")

      assert TrainingState.get_meta(state, :key1) == "value1"
      assert TrainingState.get_meta(state, :key2) == "value2"
    end

    test "running_average converges to true mean" do
      state = %TrainingState{}
      values = [10.0, 20.0, 30.0, 40.0, 50.0]

      state = Enum.reduce(values, state, fn v, s ->
        TrainingState.running_average(s, :test_metric, v)
      end)

      avg = TrainingState.get_metric(state, :test_metric)
      assert_in_delta avg, 30.0, 0.01
    end

    test "event_counts increment correctly" do
      state = %TrainingState{event_counts: %{}}
      state = Callback.increment_event_count(state, :on_batch_end)
      state = Callback.increment_event_count(state, :on_batch_end)
      state = Callback.increment_event_count(state, :on_epoch_end)

      assert state.event_counts[:on_batch_end] == 2
      assert state.event_counts[:on_epoch_end] == 1
    end
  end

  describe "callback ordering invariants" do
    defmodule OrderTracker do
      use ExPhil.Training.Callback
      @impl true
      def init(_), do: %{order: []}
      @impl true
      def on_train_begin(state, cb), do: {:cont, state, %{cb | order: [:train_begin | cb.order]}}
      @impl true
      def on_epoch_begin(state, cb), do: {:cont, state, %{cb | order: [:epoch_begin | cb.order]}}
      @impl true
      def on_batch_end(state, cb), do: {:cont, state, %{cb | order: [:batch_end | cb.order]}}
      @impl true
      def on_epoch_end(state, cb), do: {:cont, state, %{cb | order: [:epoch_end | cb.order]}}
      @impl true
      def on_train_end(state, cb), do: {:cont, state, %{cb | order: [:train_end | cb.order]}}
    end

    test "hooks fire in correct lifecycle order" do
      callbacks = Callback.init_all([{OrderTracker, []}])
      state = %TrainingState{}

      {_, state, callbacks} = Callback.run(callbacks, :on_train_begin, state)
      {_, state, callbacks} = Callback.run(callbacks, :on_epoch_begin, state)
      {_, state, callbacks} = Callback.run(callbacks, :on_batch_end, state)
      {_, state, callbacks} = Callback.run(callbacks, :on_batch_end, state)
      {_, state, callbacks} = Callback.run(callbacks, :on_epoch_end, state)
      {_, _state, callbacks} = Callback.run(callbacks, :on_train_end, state)

      [{_, cb}] = callbacks
      order = Enum.reverse(cb.order)
      assert order == [:train_begin, :epoch_begin, :batch_end, :batch_end, :epoch_end, :train_end]
    end

    test "all callbacks run even after halt" do
      defmodule HaltFirst do
        use ExPhil.Training.Callback
        @impl true
        def init(_), do: %{ran: false}
        @impl true
        def on_epoch_end(state, cb), do: {:halt, state, %{cb | ran: true}}
      end

      defmodule CheckSecond do
        use ExPhil.Training.Callback
        @impl true
        def init(_), do: %{ran: false}
        @impl true
        def on_epoch_end(state, cb), do: {:cont, state, %{cb | ran: true}}
      end

      callbacks = Callback.init_all([{HaltFirst, []}, {CheckSecond, []}])
      state = %TrainingState{}

      {:halt, _state, callbacks} = Callback.run(callbacks, :on_epoch_end, state)

      [{_, first}, {_, second}] = callbacks
      assert first.ran == true
      assert second.ran == true, "Second callback should run even after first halts"
    end
  end

  describe "filter predicates" do
    test "{:every, n} matches correctly" do
      state = %TrainingState{event_counts: %{on_batch_end: 10}}
      assert Callback.filter_matches?({:every, 5}, state, :on_batch_end)
      assert Callback.filter_matches?({:every, 10}, state, :on_batch_end)
      refute Callback.filter_matches?({:every, 3}, state, :on_batch_end)
    end

    test ":once matches only first" do
      state1 = %TrainingState{event_counts: %{on_batch_end: 1}}
      state2 = %TrainingState{event_counts: %{on_batch_end: 2}}
      assert Callback.filter_matches?(:once, state1, :on_batch_end)
      refute Callback.filter_matches?(:once, state2, :on_batch_end)
    end

    test "{:after, n} matches after n firings" do
      state5 = %TrainingState{event_counts: %{on_batch_end: 5}}
      state10 = %TrainingState{event_counts: %{on_batch_end: 10}}
      refute Callback.filter_matches?({:after, 10}, state5, :on_batch_end)
      refute Callback.filter_matches?({:after, 10}, state10, :on_batch_end)
      assert Callback.filter_matches?({:after, 4}, state5, :on_batch_end)
    end

    test "custom filter function" do
      state = %TrainingState{event_counts: %{on_batch_end: 7}}
      filter = fn _state, count -> rem(count, 7) == 0 end
      assert Callback.filter_matches?(filter, state, :on_batch_end)
    end
  end
end
