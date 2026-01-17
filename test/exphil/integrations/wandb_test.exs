defmodule ExPhil.Integrations.WandbTest do
  use ExUnit.Case, async: false

  alias ExPhil.Integrations.Wandb

  # Most tests require API key, so we mock/skip accordingly
  # These tests verify the module structure without hitting the API

  describe "active?/0" do
    test "returns false when no run active" do
      refute Wandb.active?()
    end
  end

  describe "run_info/0" do
    test "returns nil when no run active" do
      assert Wandb.run_info() == nil
    end
  end

  describe "log/2" do
    test "returns error when no run active" do
      assert {:error, :no_active_run} = Wandb.log(%{loss: 0.5})
    end
  end

  describe "finish_run/0" do
    test "returns error when no run active" do
      assert {:error, :no_active_run} = Wandb.finish_run()
    end
  end

  describe "start_run/1" do
    test "fails without API key" do
      # Ensure no API key in env
      original = System.get_env("WANDB_API_KEY")
      System.delete_env("WANDB_API_KEY")

      # Trap exit since GenServer will exit
      Process.flag(:trap_exit, true)

      # GenServer.start_link returns {:error, reason} when init returns {:stop, reason}
      result = Wandb.start_run(project: "test")
      assert {:error, _reason} = result

      # Restore
      if original, do: System.put_env("WANDB_API_KEY", original)
      Process.flag(:trap_exit, false)
    end

    @tag :integration
    @tag :wandb
    test "starts run with valid API key" do
      # Skip if no API key available
      api_key = System.get_env("WANDB_API_KEY")

      if api_key do
        assert {:ok, run_id} = Wandb.start_run(
          project: "exphil-test",
          name: "test-run-#{System.system_time(:second)}",
          config: %{test: true}
        )

        assert is_binary(run_id)
        assert Wandb.active?()

        info = Wandb.run_info()
        assert info.project == "exphil-test"

        # Cleanup
        Wandb.finish_run()
      end
    end
  end

  describe "telemetry integration" do
    @tag :integration
    @tag :wandb
    test "logs telemetry events automatically" do
      api_key = System.get_env("WANDB_API_KEY")

      if api_key do
        {:ok, _run_id} = Wandb.start_run(
          project: "exphil-test",
          name: "telemetry-test-#{System.system_time(:second)}"
        )

        # Emit a training step event
        ExPhil.Telemetry.training_step(%{loss: 0.5, step: 1}, %{epoch: 1})

        # Give it time to process
        Process.sleep(100)

        info = Wandb.run_info()
        assert info.step >= 1

        Wandb.finish_run()
      end
    end
  end
end
