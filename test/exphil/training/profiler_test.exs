defmodule ExPhil.Training.ProfilerTest do
  use ExUnit.Case, async: false

  alias ExPhil.Training.Profiler

  # Not async because Profiler uses a named Agent that could conflict

  setup do
    # Ensure profiler is stopped before each test
    Profiler.stop()
    :ok
  end

  describe "start/stop" do
    test "starts the profiler agent" do
      assert {:ok, _pid} = Profiler.start()
      assert Process.whereis(Profiler) != nil
    end

    test "starting twice returns the existing pid" do
      assert {:ok, pid1} = Profiler.start()
      assert {:ok, pid2} = Profiler.start()
      assert pid1 == pid2
    end

    test "stop/0 stops the agent" do
      Profiler.start()
      assert Process.whereis(Profiler) != nil
      Profiler.stop()
      assert Process.whereis(Profiler) == nil
    end

    test "stop/0 is idempotent" do
      Profiler.stop()
      assert :ok = Profiler.stop()
    end
  end

  describe "enabled?/set_enabled" do
    test "profiler starts disabled" do
      Profiler.start()
      refute Profiler.enabled?()
    end

    test "set_enabled/1 enables profiling" do
      Profiler.start()
      Profiler.set_enabled(true)
      assert Profiler.enabled?()
    end

    test "set_enabled/1 can disable profiling" do
      Profiler.start()
      Profiler.set_enabled(true)
      assert Profiler.enabled?()
      Profiler.set_enabled(false)
      refute Profiler.enabled?()
    end

    test "enabled? returns false when profiler not started" do
      refute Profiler.enabled?()
    end
  end

  describe "time/2" do
    test "returns the function result" do
      Profiler.start()
      Profiler.set_enabled(true)

      result = Profiler.time(:test, fn -> 42 end)
      assert result == 42
    end

    test "records timing when enabled" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.time(:test_key, fn -> :timer.sleep(10) end)

      stats = Profiler.get_stats()
      assert Map.has_key?(stats, :test_key)
      assert stats[:test_key].count == 1
      assert stats[:test_key].total_ms >= 10
    end

    test "does not record timing when disabled" do
      Profiler.start()
      Profiler.set_enabled(false)

      Profiler.time(:test_key, fn -> :timer.sleep(5) end)

      stats = Profiler.get_stats()
      refute Map.has_key?(stats, :test_key)
    end

    test "accumulates multiple timings for same key" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.time(:multi, fn -> :timer.sleep(10) end)
      Profiler.time(:multi, fn -> :timer.sleep(15) end)
      Profiler.time(:multi, fn -> :timer.sleep(5) end)

      stats = Profiler.get_stats()
      assert stats[:multi].count == 3
      assert stats[:multi].total_ms >= 30
      assert stats[:multi].min_ms >= 5
      assert stats[:multi].max_ms >= 15
    end

    test "still executes function when profiler not started" do
      result = Profiler.time(:no_profiler, fn -> :hello end)
      assert result == :hello
    end
  end

  describe "record/2" do
    test "records timing directly" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:manual, 100.0)

      stats = Profiler.get_stats()
      assert stats[:manual].count == 1
      assert stats[:manual].total_ms == 100.0
    end

    test "accumulates multiple recordings" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:manual, 10.0)
      Profiler.record(:manual, 20.0)
      Profiler.record(:manual, 30.0)

      stats = Profiler.get_stats()
      assert stats[:manual].count == 3
      assert stats[:manual].total_ms == 60.0
      assert stats[:manual].min_ms == 10.0
      assert stats[:manual].max_ms == 30.0
    end

    test "does not record when disabled" do
      Profiler.start()
      Profiler.set_enabled(false)

      Profiler.record(:disabled, 100.0)

      stats = Profiler.get_stats()
      refute Map.has_key?(stats, :disabled)
    end
  end

  describe "reset/0" do
    test "clears all timing statistics" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:a, 10.0)
      Profiler.record(:b, 20.0)

      stats_before = Profiler.get_stats()
      assert map_size(stats_before) == 2

      Profiler.reset()

      stats_after = Profiler.get_stats()
      assert map_size(stats_after) == 0
    end
  end

  describe "report/0" do
    test "returns 'no data' message when empty" do
      Profiler.start()
      Profiler.set_enabled(true)

      report = Profiler.report()
      assert report =~ "No profiling data collected"
    end

    test "generates formatted report with data" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:forward, 100.0)
      Profiler.record(:backward, 80.0)
      Profiler.record(:validation, 50.0)

      report = Profiler.report()

      # Check header
      assert report =~ "TRAINING PROFILE REPORT"
      assert report =~ "Phase"
      assert report =~ "Count"
      assert report =~ "Total (s)"
      assert report =~ "Avg (ms)"

      # Check phases appear
      assert report =~ "forward"
      assert report =~ "backward"
      assert report =~ "validation"

      # Check footer
      assert report =~ "Total tracked time"
    end

    test "sorts phases by total time descending" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:small, 10.0)
      Profiler.record(:large, 100.0)
      Profiler.record(:medium, 50.0)

      report = Profiler.report()

      # Large should appear before medium, medium before small
      large_pos = :binary.match(report, "large") |> elem(0)
      medium_pos = :binary.match(report, "medium") |> elem(0)
      small_pos = :binary.match(report, "small") |> elem(0)

      assert large_pos < medium_pos
      assert medium_pos < small_pos
    end
  end

  describe "inline_summary/0" do
    test "returns empty string when no data" do
      Profiler.start()
      Profiler.set_enabled(true)

      assert Profiler.inline_summary() == ""
    end

    test "returns top 3 phases by total time" do
      Profiler.start()
      Profiler.set_enabled(true)

      Profiler.record(:a, 100.0)
      Profiler.record(:b, 80.0)
      Profiler.record(:c, 60.0)
      Profiler.record(:d, 40.0)

      summary = Profiler.inline_summary()

      assert summary =~ "[Profile:"
      assert summary =~ "a="
      assert summary =~ "b="
      assert summary =~ "c="
      refute summary =~ "d="
    end
  end

  describe "samples tracking" do
    test "keeps last 1000 samples" do
      Profiler.start()
      Profiler.set_enabled(true)

      # Record 1100 samples
      for i <- 1..1100 do
        Profiler.record(:many, i * 1.0)
      end

      stats = Profiler.get_stats()
      assert length(stats[:many].samples) == 1000

      # Samples should be the most recent ones (1100 down to 101)
      # The most recent sample should be 1100.0
      assert hd(stats[:many].samples) == 1100.0
    end
  end
end
