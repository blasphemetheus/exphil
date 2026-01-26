defmodule ExPhil.TelemetryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Telemetry

  describe "training_step/2" do
    test "emits training step event" do
      test_pid = self()

      :telemetry.attach(
        "test-training-step",
        [:exphil, :training, :step],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.training_step(%{loss: 0.5, step: 1}, %{epoch: 1})

      assert_receive {:telemetry, [:exphil, :training, :step], %{loss: 0.5, step: 1}, %{epoch: 1}}

      :telemetry.detach("test-training-step")
    end
  end

  describe "training_epoch/2" do
    test "emits training epoch event" do
      test_pid = self()

      :telemetry.attach(
        "test-training-epoch",
        [:exphil, :training, :epoch],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.training_epoch(%{avg_loss: 0.3, epoch: 1}, %{total_steps: 100})

      assert_receive {:telemetry, [:exphil, :training, :epoch], %{avg_loss: 0.3, epoch: 1},
                      %{total_steps: 100}}

      :telemetry.detach("test-training-epoch")
    end
  end

  describe "checkpoint_saved/2" do
    test "emits checkpoint event" do
      test_pid = self()

      :telemetry.attach(
        "test-checkpoint",
        [:exphil, :training, :checkpoint],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.checkpoint_saved(%{step: 500}, %{path: "/tmp/checkpoint.axon"})

      assert_receive {:telemetry, [:exphil, :training, :checkpoint], %{step: 500},
                      %{path: "/tmp/checkpoint.axon"}}

      :telemetry.detach("test-checkpoint")
    end
  end

  describe "agent_action/2" do
    test "emits agent action event" do
      test_pid = self()

      :telemetry.attach(
        "test-agent-action",
        [:exphil, :agent, :action],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.agent_action(%{frame: 100}, %{agent: :mewtwo})

      assert_receive {:telemetry, [:exphil, :agent, :action], %{frame: 100}, %{agent: :mewtwo}}

      :telemetry.detach("test-agent-action")
    end
  end

  describe "agent_inference/2" do
    test "emits inference timing event" do
      test_pid = self()

      :telemetry.attach(
        "test-agent-inference",
        [:exphil, :agent, :inference],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.agent_inference(%{duration_us: 1500}, %{agent: :mewtwo})

      assert_receive {:telemetry, [:exphil, :agent, :inference], %{duration_us: 1500},
                      %{agent: :mewtwo}}

      :telemetry.detach("test-agent-inference")
    end
  end

  describe "bridge_step/2" do
    test "emits bridge step event" do
      test_pid = self()

      :telemetry.attach(
        "test-bridge-step",
        [:exphil, :bridge, :step],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.bridge_step(%{frame: 50}, %{game_id: "abc123"})

      assert_receive {:telemetry, [:exphil, :bridge, :step], %{frame: 50}, %{game_id: "abc123"}}

      :telemetry.detach("test-bridge-step")
    end
  end

  describe "game_end/2" do
    test "emits game end event" do
      test_pid = self()

      :telemetry.attach(
        "test-game-end",
        [:exphil, :bridge, :game_end],
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Telemetry.game_end(%{total_frames: 3600}, %{winner: 1})

      assert_receive {:telemetry, [:exphil, :bridge, :game_end], %{total_frames: 3600},
                      %{winner: 1}}

      :telemetry.detach("test-game-end")
    end
  end

  describe "attach_console_logger/1" do
    test "attaches handlers for all events" do
      Telemetry.attach_console_logger(level: :debug)

      # Verify handlers are attached by checking they exist
      handlers = :telemetry.list_handlers([:exphil, :training, :step])
      assert Enum.any?(handlers, fn h -> String.contains?(h.id, "exphil-console") end)

      Telemetry.detach_console_logger()
    end
  end

  describe "detach_console_logger/0" do
    test "detaches all console handlers" do
      Telemetry.attach_console_logger()
      Telemetry.detach_console_logger()

      handlers = :telemetry.list_handlers([:exphil, :training, :step])
      refute Enum.any?(handlers, fn h -> String.contains?(h.id, "exphil-console") end)
    end
  end
end

defmodule ExPhil.Telemetry.CollectorTest do
  use ExUnit.Case, async: false

  alias ExPhil.Telemetry

  # Collector is already started by Application
  setup do
    # Flush any existing metrics before each test
    Telemetry.flush_metrics()
    :ok
  end

  describe "get_metrics/0" do
    test "returns empty metrics initially" do
      result = Telemetry.get_metrics()

      assert is_map(result)
      assert result.metrics == %{}
      assert result.counts == %{}
    end

    test "accumulates training step metrics" do
      # Emit some events
      Telemetry.training_step(%{loss: 0.5}, %{})
      Telemetry.training_step(%{loss: 0.3}, %{})

      # Give collector time to process
      Process.sleep(50)

      result = Telemetry.get_metrics()

      assert Map.has_key?(result.metrics, "exphil.training.step")
      assert result.counts["exphil.training.step"] == 2
    end
  end

  describe "flush_metrics/0" do
    test "resets accumulated metrics" do
      Telemetry.training_step(%{loss: 0.5}, %{})
      Process.sleep(50)

      assert :ok = Telemetry.flush_metrics()

      result = Telemetry.get_metrics()
      assert result.metrics == %{}
      assert result.counts == %{}
    end
  end
end
