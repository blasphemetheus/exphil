defmodule ExPhil.SelfPlay.ExperienceCollectorTest do
  use ExUnit.Case, async: true

  alias ExPhil.SelfPlay.ExperienceCollector

  @moduletag :self_play

  describe "start_link/1" do
    test "starts with default options" do
      assert {:ok, pid} = ExperienceCollector.start_link(name: nil)
      assert Process.alive?(pid)

      stats = ExperienceCollector.get_stats(pid)
      assert stats.batch_size == 2048
      assert stats.buffer_size == 0

      GenServer.stop(pid)
    end

    test "starts with custom batch size" do
      assert {:ok, pid} = ExperienceCollector.start_link(name: nil, batch_size: 512)

      stats = ExperienceCollector.get_stats(pid)
      assert stats.batch_size == 512

      GenServer.stop(pid)
    end
  end

  describe "submit/2" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil, batch_size: 10)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "adds experience to buffer", %{collector: collector} do
      exp = mock_experience()
      :ok = ExperienceCollector.submit(collector, exp)

      # Give cast time to process
      :timer.sleep(10)

      assert ExperienceCollector.buffer_size(collector) == 1
    end

    test "increments total submitted", %{collector: collector} do
      for _ <- 1..5 do
        ExperienceCollector.submit(collector, mock_experience())
      end

      :timer.sleep(10)

      stats = ExperienceCollector.get_stats(collector)
      assert stats.total_submitted == 5
    end
  end

  describe "submit_batch/2" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "adds multiple experiences", %{collector: collector} do
      experiences = for _ <- 1..10, do: mock_experience()
      :ok = ExperienceCollector.submit_batch(collector, experiences)

      :timer.sleep(10)

      assert ExperienceCollector.buffer_size(collector) == 10
    end
  end

  describe "get_batch/3" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil, batch_size: 10)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "returns batch when enough experiences", %{collector: collector} do
      experiences = for _ <- 1..10, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)
      :timer.sleep(10)

      assert {:ok, batch} = ExperienceCollector.get_batch(collector, 5)

      assert Map.has_key?(batch, :states)
      assert Map.has_key?(batch, :rewards)
      assert Map.has_key?(batch, :dones)
    end

    test "buffer is reduced after get_batch", %{collector: collector} do
      experiences = for _ <- 1..10, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)
      :timer.sleep(10)

      {:ok, _batch} = ExperienceCollector.get_batch(collector, 5)

      assert ExperienceCollector.buffer_size(collector) == 5
    end
  end

  describe "get_all/1" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "returns all experiences and empties buffer", %{collector: collector} do
      experiences = for _ <- 1..5, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)
      :timer.sleep(10)

      {:ok, batch} = ExperienceCollector.get_all(collector)

      assert is_map(batch)
      assert ExperienceCollector.buffer_size(collector) == 0
    end
  end

  describe "flush/1" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "returns remaining experiences", %{collector: collector} do
      experiences = for _ <- 1..3, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)
      :timer.sleep(10)

      {:ok, batch} = ExperienceCollector.flush(collector)

      assert is_map(batch)
      assert ExperienceCollector.buffer_size(collector) == 0
    end
  end

  describe "reset/1" do
    setup do
      {:ok, pid} = ExperienceCollector.start_link(name: nil)
      on_exit(fn -> safe_stop(pid) end)
      %{collector: pid}
    end

    test "clears buffer and stats", %{collector: collector} do
      experiences = for _ <- 1..5, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)
      :timer.sleep(10)

      :ok = ExperienceCollector.reset(collector)

      stats = ExperienceCollector.get_stats(collector)
      assert stats.buffer_size == 0
      assert stats.total_submitted == 0
    end
  end

  describe "auto_batch mode" do
    test "calls callback when batch ready" do
      test_pid = self()

      callback = fn batch ->
        send(test_pid, {:batch_ready, batch})
      end

      {:ok, collector} =
        ExperienceCollector.start_link(
          name: nil,
          batch_size: 5,
          auto_batch: true,
          ready_callback: callback
        )

      experiences = for _ <- 1..6, do: mock_experience()
      ExperienceCollector.submit_batch(collector, experiences)

      # Should receive callback
      assert_receive {:batch_ready, _batch}, 1000

      GenServer.stop(collector)
    end
  end

  # Helper functions

  defp mock_experience do
    %{
      state: Nx.tensor(Enum.map(1..100, fn _ -> :rand.uniform() end)),
      action: %{
        buttons: Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        main_x: Nx.tensor(8),
        main_y: Nx.tensor(8),
        c_x: Nx.tensor(8),
        c_y: Nx.tensor(8),
        shoulder: Nx.tensor(0)
      },
      log_prob: Nx.tensor(-0.5),
      value: Nx.tensor(0.1),
      reward: 0.01,
      done: false
    }
  end

  defp safe_stop(pid) do
    if Process.alive?(pid) do
      GenServer.stop(pid, :normal, 1000)
    end
  rescue
    _ -> :ok
  end
end
