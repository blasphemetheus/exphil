defmodule ExPhil.Training.MemoryLedgerTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.MemoryLedger

  @tmp System.tmp_dir!()

  defp tmp_ledger(name),
    do: Path.join(@tmp, "memory_ledger_test_#{name}_#{System.unique_integer([:positive])}.jsonl")

  describe "parse_proc_status/1" do
    test "extracts VmRSS and VmHWM in bytes" do
      text = """
      Name:\tbeam.smp
      VmPeak:\t 9000000 kB
      VmHWM:\t 5000000 kB
      VmRSS:\t 4000000 kB
      """

      assert %{rss_bytes: 4_096_000_000, hwm_bytes: 5_120_000_000} =
               MemoryLedger.parse_proc_status(text)
    end

    test "missing keys parse as zero" do
      assert %{rss_bytes: 0, hwm_bytes: 0} = MemoryLedger.parse_proc_status("Name:\tx\n")
    end
  end

  describe "parse_meminfo/1" do
    test "extracts MemAvailable, MemTotal, SwapFree" do
      text = """
      MemTotal:       65000000 kB
      MemFree:        20000000 kB
      MemAvailable:   50000000 kB
      SwapFree:       30000000 kB
      """

      m = MemoryLedger.parse_meminfo(text)
      assert m.total_bytes == 65_000_000 * 1024
      assert m.available_bytes == 50_000_000 * 1024
      assert m.swap_free_bytes == 30_000_000 * 1024
    end
  end

  describe "process_memory/0" do
    test "reads live values from /proc/self/status" do
      m = MemoryLedger.process_memory()
      assert m.rss_bytes > 0
      # High-water mark can never be below current RSS
      assert m.hwm_bytes >= m.rss_bytes
    end
  end

  describe "ledger append/entries" do
    test "round-trips entries and skips malformed lines" do
      path = tmp_ledger("roundtrip")

      MemoryLedger.append(%{mode: "preflight", pool_frames: 1_000, peak_rss_bytes: 5_000}, path)
      File.write!(path, "not json\n", [:append])
      File.write!(path, ~s({"pool_frames": "bad-type", "peak_rss_bytes": 1}\n), [:append])
      MemoryLedger.append(%{mode: "train", pool_frames: 2_000, peak_rss_bytes: 9_000}, path)

      entries = MemoryLedger.entries(path)
      assert length(entries) == 2
      assert Enum.all?(entries, & &1.recorded_at)

      File.rm!(path)
    end

    test "missing file yields no entries" do
      assert MemoryLedger.entries(tmp_ledger("missing")) == []
    end
  end

  describe "fit/1" do
    test "no data" do
      assert {:error, :no_data} = MemoryLedger.fit([])
    end

    test "single point falls back to proportional through the origin" do
      assert {:ok, slope, +0.0, :proportional} = MemoryLedger.fit([{1_000, 4_000_000}])
      assert_in_delta slope, 4_000.0, 1.0e-9
    end

    test "several points at ONE pool size stay proportional, using the worst peak" do
      points = [{1_000, 4_000_000}, {1_000, 6_000_000}]
      assert {:ok, slope, +0.0, :proportional} = MemoryLedger.fit(points)
      assert_in_delta slope, 6_000.0, 1.0e-9
    end

    test "two distinct pool sizes give an exact linear fit" do
      # peak = 3000 * frames + 1_000_000
      points = [{1_000, 4_000_000}, {2_000, 7_000_000}]
      assert {:ok, slope, intercept, :linear_fit} = MemoryLedger.fit(points)
      assert_in_delta slope, 3_000.0, 1.0e-6
      assert_in_delta intercept, 1_000_000.0, 1.0e-3
    end
  end

  describe "predict/2" do
    test "filters by window, tolerating entries that predate the field" do
      path = tmp_ledger("predict")

      MemoryLedger.append(%{pool_frames: 1_000, peak_rss_bytes: 4_000_000, window: 60}, path)
      MemoryLedger.append(%{pool_frames: 2_000, peak_rss_bytes: 7_000_000, window: 60}, path)
      # Different window — must be excluded by the filter
      MemoryLedger.append(%{pool_frames: 2_000, peak_rss_bytes: 99_000_000, window: 16}, path)
      # Pre-field entry (no window) — included
      MemoryLedger.append(%{pool_frames: 3_000, peak_rss_bytes: 10_000_000}, path)

      assert {:ok, %{model: :linear_fit, points: 3, predicted_bytes: p}} =
               MemoryLedger.predict(4_000, path: path, window: 60)

      # exact fit through (1k,4M) (2k,7M) (3k,10M): 3000 b/frame + 1M
      assert_in_delta p, 13_000_000, 1.0

      File.rm!(path)
    end

    test "no matching entries" do
      path = tmp_ledger("nomatch")
      MemoryLedger.append(%{pool_frames: 1_000, peak_rss_bytes: 1, window: 16}, path)

      assert {:error, :no_data} = MemoryLedger.predict(5_000, path: path, window: 60)
      File.rm!(path)
    end
  end

  describe "headroom_check/2" do
    test "small predicted peak passes against the live budget" do
      path = tmp_ledger("headroom_ok")
      # 1 byte/frame — any real machine has headroom for this
      MemoryLedger.append(%{pool_frames: 1_000, peak_rss_bytes: 1_000}, path)

      assert {:ok, info} = MemoryLedger.headroom_check(2_000, path: path)
      assert info.predicted_bytes == 2_000
      assert info.budget_bytes > 0

      File.rm!(path)
    end

    test "absurd predicted peak warns" do
      path = tmp_ledger("headroom_warn")
      # 1 TB/frame — no machine passes
      MemoryLedger.append(%{pool_frames: 1, peak_rss_bytes: 1_000_000_000_000}, path)

      assert {:warn, info} = MemoryLedger.headroom_check(1_000, path: path)
      assert info.predicted_bytes > info.budget_bytes

      File.rm!(path)
    end

    test "empty ledger reports no_data" do
      assert {:no_data, %{}} = MemoryLedger.headroom_check(1_000, path: tmp_ledger("nodata"))
    end
  end
end
