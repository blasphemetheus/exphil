defmodule ExPhil.BenchmarkSweep do
  @moduledoc "Shared file integrity, GPU isolation, and subprocess helpers for benchmark scripts."

  def read(path), do: path |> File.read!() |> Jason.decode!()
  def write(path, data), do: File.write!(path, Jason.encode!(data, pretty: true) <> "\n")

  def hashes(paths) do
    Application.ensure_all_started(:crypto)

    Map.new(Enum.uniq(paths), fn path ->
      digest =
        path
        |> File.stream!(2048 * 1024)
        |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
        |> :crypto.hash_final()
        |> Base.encode16(case: :lower)

      {path, digest}
    end)
  end

  def gpu_pids do
    {output, 0} = System.cmd("nvidia-smi", ["--query-compute-apps=pid", "--format=csv,noheader"])
    String.split(output, "\n", trim: true)
  end

  def wait_for_gpu(quiet_seconds \\ 10, timeout_seconds \\ 600) do
    IO.puts("Waiting for #{quiet_seconds}s with no GPU compute process")
    wait_gpu(quiet_seconds, System.monotonic_time(:second) + timeout_seconds, nil)
  end

  defp wait_gpu(quiet, deadline, since) do
    now = System.monotonic_time(:second)
    if now >= deadline, do: raise("GPU did not become idle; resume later")
    since = if gpu_pids() == [], do: since || now, else: nil

    if since && now - since >= quiet do
      :ok
    else
      Process.sleep(1000)
      wait_gpu(quiet, deadline, since)
    end
  end

  defp monitor(samples) do
    receive do
      :stop -> Enum.reverse(samples)
    after
      1000 ->
        pids = gpu_pids()

        samples =
          if length(pids) > 1,
            do: [%{time: System.system_time(:second), pids: pids} | samples],
            else: samples

        monitor(samples)
    end
  end

  def run(args, env, log) do
    watcher = Task.async(fn -> monitor([]) end)

    try do
      {_, status} =
        System.cmd("elixir", args, env: env, into: File.stream!(log), stderr_to_stdout: true)

      send(watcher.pid, :stop)
      conflicts = Task.await(watcher, 5000)
      if status != 0, do: raise("Benchmark exited #{status}; see #{log}")
      conflicts
    after
      if Process.alive?(watcher.pid), do: Task.shutdown(watcher, :brutal_kill)
    end
  end

  def policies do
    fox = "/home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp"

    [
      {"min_gru", "checkpoints/coh_prev_q_ck4/model_best_policy.bin", fox, "fox"},
      {"gru", "checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin", fox, "fox"},
      {"mamba", "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin", fox, "fox"},
      {"mamba2", "checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin",
       "replays/greg/mewtwo/Game_20240222T222930.slp", "mewtwo"}
    ]
  end

  def inference_sources do
    [
      "scripts/run_recurrent_validation.exs",
      "lib/exphil/agents/agent.ex",
      "lib/exphil/networks/policy/sampling.ex",
      "../edifice/lib/edifice/utils/sampling_random.ex",
      "../edifice/lib/edifice/stateful.ex",
      "../edifice/lib/edifice/stateful/replay.ex"
    ]
  end

  def inputs, do: Enum.flat_map(policies(), fn {_, policy, replay, _} -> [policy, replay] end)
end
