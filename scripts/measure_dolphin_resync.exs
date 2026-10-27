# AI-generated. Method and attribution: docs/guides/MAMBA_BATCH_PARITY.md.
# Two real Dolphin-controlled bots; no training or policy promotion.
Code.compile_file("lib/exphil/embeddings/player.ex")
Code.compile_file("lib/exphil/embeddings/game/spatial.ex")

defmodule DolphinResyncMeasurement do
  alias ExPhil.Agents.Agent, as: PolicyAgent
  alias ExPhil.Bridge.MeleePort
  alias ExPhil.Training.Output

  def input(c, port) do
    %{
      port: port,
      main_stick: %{x: c.main_stick.x, y: c.main_stick.y},
      c_stick: %{x: c.c_stick.x, y: c.c_stick.y},
      shoulder: c.l_shoulder + c.r_shoulder,
      buttons: %{
        a: c.button_a,
        b: c.button_b,
        x: c.button_x,
        y: c.button_y,
        z: c.button_z,
        l: c.button_l,
        r: c.button_r,
        d_up: c.button_d_up
      }
    }
  end

  def run(opts) do
    frames = opts[:frames] || 600
    phase = opts[:phase] || 0
    policy = "checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin"

    agents =
      for port <- [1, 2] do
        {:ok, a} =
          PolicyAgent.start_link(
            policy: policy,
            deterministic: true,
            stateful_step: true,
            stateful_resync: 30,
            stateful_resync_phase: if(port == 2, do: phase, else: 0),
            harness: :sync_runner,
            frame_delay: 0,
            reaction_delay: 0,
            af_convention: :live
          )

        {:ok, _} = PolicyAgent.warmup(a)
        a
      end

    {:ok, bridge} = MeleePort.start_link()

    try do
      config = %{
        dolphin_path:
          opts[:dolphin] || Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
        iso_path: opts[:iso] || Path.expand("~/isos/melee.iso"),
        controller_port: 1,
        opponent_port: 2,
        character: "fox",
        dummy_character: "fox",
        dummy_mode: "external",
        dummy_cpu_level: 0,
        stage: "final_destination",
        headless: true,
        no_audio: true,
        blocking_input: true,
        emulation_speed: 1.0,
        online_delay: 0,
        slippi_port: 51462,
        save_replays: false
      }

      Output.puts("Starting two-bot Dolphin measurement; phase #{phase}")
      {:ok, _} = MeleePort.init_console(bridge, config, 60_000)
      deadline = System.monotonic_time(:millisecond) + 120_000
      samples = loop(bridge, agents, frames, deadline, [], nil)
      # Exclude startup/history fill and first complete resync from timing claims.
      warm = Enum.drop(samples, 160)

      report = %{
        client: EXLA.Client.default_name(),
        phase: phase,
        frames: length(samples),
        warm_frames: length(warm),
        pair_over_budget: Enum.count(warm, &(&1.inference_us > 16667)),
        pair_p95_us: percentile(warm, :inference_us, 0.95),
        pair_p99_us: percentile(warm, :inference_us, 0.99),
        frame_p95_us: percentile(warm, :frame_us, 0.95),
        frame_median_us: percentile(warm, :frame_us, 0.5),
        samples: samples,
        note:
          "Two independent deterministic Fox Mamba agents, real Dolphin, blocking input, absolute 60 Hz request schedule. Inference budget exceeds are not independent asynchronous deadline misses. No quality or win-rate claim."
      }

      out = Keyword.fetch!(opts, :out)
      File.mkdir_p!(Path.dirname(out))
      File.write!(out, Jason.encode!(report, pretty: true))

      Output.puts(
        "Saved #{out}: #{report.pair_over_budget}/#{length(warm)} warm frames exceed inference budget"
      )
    after
      MeleePort.stop(bridge)
      for a <- agents, do: GenServer.stop(a)
    end
  end

  def percentile([], _, _), do: nil

  def percentile(samples, key, q),
    do: samples |> Enum.map(& &1[key]) |> Enum.sort() |> Enum.at(ceil(length(samples) * q) - 1)

  def loop(_bridge, _agents, 0, _deadline, samples, _last), do: Enum.reverse(samples)

  def loop(bridge, agents, left, deadline, samples, last) do
    if last do
      target = Process.get(:resync_measurement_epoch) + div(length(samples) * 1_000_000, 60)
      wait_us = target - System.monotonic_time(:microsecond)
      if wait_us > 0, do: Process.sleep(ceil(wait_us / 1000))
    end

    if System.monotonic_time(:millisecond) > deadline, do: raise("Dolphin frame deadline expired")

    case MeleePort.step(bridge, [poll: true], 10_000) do
      {:ok, game} when game.frame >= 0 ->
        now = System.monotonic_time(:microsecond)

        if is_nil(last),
          do: Process.put(:resync_measurement_epoch, now - div(length(samples) * 1_000_000, 60))

        tasks =
          for {a, port} <- Enum.with_index(agents, 1) do
            Task.async(fn ->
              {us, {:ok, c}} =
                :timer.tc(fn -> PolicyAgent.get_controller(a, game, player_port: port) end)

              {port, us, c}
            end)
          end

        results = Enum.map(tasks, &Task.await(&1, 30_000))
        inference_us = System.monotonic_time(:microsecond) - now
        for {port, _, c} <- results, do: :ok = MeleePort.send_controller(bridge, input(c, port))

        sample = %{
          game_frame: game.frame,
          inference_us: inference_us,
          frame_us: if(last, do: now - last, else: 0),
          actor_us: Enum.map(results, &elem(&1, 1))
        }

        if rem(left, 120) == 0, do: Output.puts("#{length(samples)} game frames measured")
        loop(bridge, agents, left - 1, deadline, [sample | samples], now)

      {:game_ended, reason} ->
        raise("Dolphin ended: #{inspect(reason)}")

      {:error, reason} ->
        raise("Dolphin error: #{inspect(reason)}")

      {:postgame, _} ->
        for a <- agents, do: PolicyAgent.reset_buffer(a)
        loop(bridge, agents, left, deadline, samples, nil)

      _ ->
        loop(bridge, agents, left, deadline, samples, last)
    end
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [out: :string, frames: :integer, phase: :integer, dolphin: :string, iso: :string]
  )

DolphinResyncMeasurement.run(opts)
