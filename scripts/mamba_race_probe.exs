# Widen the chunk-transition race until it reproduces on demand (2026-09-26).
#
# The illegal-write crash (Xid 31, FAULT_PDE VIRT_WRITE) hits the two-chunk
# transition test intermittently — steps 21 / 81 / 639 and three passes on the
# same inputs. The transition overlaps ~35 s of background chunk preparation
# with training; this probe keeps that preparation running in a LOOP for the
# whole run so the overlap is continuous, then trains from a saved capture on
# a saved batch (no data-dependence, ~30 ms/step).
#
#   mix run scripts/mamba_race_probe.exs CAPTURE_DIR OUT MODE STEPS [CHUNK]
#
#   CAPTURE_DIR  eval_runs/0925_fox_mamba/crash/step600 (before_step.axon + batch.bin)
#   MODE         real        — parse + Streaming.create_dataset exactly as the
#                              pipeline does (eager GPU embedding + transfer)
#                cpu         — same, embedding on Nx.BinaryBackend
#                parse_only  — Streaming.parse_chunk only (CPU, no Nx)
#                none        — no background work (control)
#   STEPS        training steps before declaring a pass
#   CHUNK        original 64-game corpus chunk to prepare (default 13)
#
# Writes OUT/last_step.json every step, OUT/background.json per prep loop,
# OUT/completed.json on a pass. A crash kills the BEAM; the unit's exit code
# and the kernel log (journalctl -k | grep Xid) are the evidence.
alias ExPhil.Training.{Imitation, Streaming, Output}
alias ExPhil.Data.{Peppi, SubjectResolver}

[capture, out, mode | rest] = System.argv()
unless mode in ~w(real cpu parse_only none), do: raise("mode must be real|cpu|parse_only|none")
{steps, chunk_no} =
  case rest do
    [s] -> {String.to_integer(s), 13}
    [s, c] -> {String.to_integer(s), String.to_integer(c)}
    _ -> raise("usage: CAPTURE_DIR OUT MODE STEPS [CHUNK]")
  end

File.mkdir_p!(out)
t_start = System.monotonic_time(:millisecond)

# --- trainer + batch from the capture -------------------------------------
saved = capture |> Path.join("before_step.axon") |> File.read!() |> :erlang.binary_to_term()
cfg = Map.put(saved.config, :button_pos_weight, Nx.tensor(List.duplicate(1.0, 8), backend: Nx.BinaryBackend))
trainer = Imitation.new(Map.to_list(cfg))
{:ok, restored} = Imitation.load_checkpoint(trainer, Path.join(capture, "before_step.axon"))
trainer = %{restored | config: cfg}
batch = capture |> Path.join("batch.bin") |> File.read!() |> :erlang.binary_to_term()

# --- the real chunk, prepared exactly as mamba_crash_stream.exs does --------
files = "checkpoints/fox_mamba_v1_20260925/split.json" |> File.read!() |> Jason.decode!() |> Map.fetch!("train")
paths = files |> Enum.chunk_every(64) |> Enum.at(chunk_no - 1)
port_map =
  for path <- paths, into: %{} do
    {:ok, meta} = Peppi.metadata(path)
    {:ok, %{subject_port: port}} = SubjectResolver.resolve(meta.players, subject_character: 2, ditto_tie_break: :port1)
    {path, port}
  end

embed_cfg = ExPhil.Embeddings.config_for_source([stage_internals: true], Peppi.provides())
chunk_opts = [port_map: port_map, player_port: 1, label_delay: 0, subject_character: "fox"]
dataset_opts = [temporal: true, window_size: 80, stride: 5, precompute: true, embed_config: embed_cfg]

Output.banner("Mamba race probe")
Output.config([{"Mode", mode}, {"Steps", steps}, {"Chunk", "#{chunk_no} (#{length(paths)} files)"}, {"Out", out}])

Output.puts("Warmup training step (JIT)")
{trainer, m} = Imitation.train_step(trainer, batch, nil)
Nx.to_number(m.loss)

defmodule MambaRacePrep do
  def loop(paths, chunk_opts, dataset_opts, mode, out, n) do
    receive do
      :stop -> n
    after
      0 ->
        t0 = System.monotonic_time(:millisecond)
        {:ok, frames, errors} = Streaming.parse_chunk(paths, chunk_opts)
        if errors != [], do: raise("parse errors: #{inspect(errors)}")

        size =
          case mode do
            "parse_only" -> length(frames)
            "cpu" -> Nx.with_default_backend(Nx.BinaryBackend, fn -> Streaming.create_dataset(frames, dataset_opts).size end)
            "real" -> Streaming.create_dataset(frames, dataset_opts).size
          end

        File.write!(Path.join(out, "background.json"),
          Jason.encode!(%{iterations: n + 1, mode: mode, last_size: size, last_ms: System.monotonic_time(:millisecond) - t0}))
        :erlang.garbage_collect()
        loop(paths, chunk_opts, dataset_opts, mode, out, n + 1)
    end
  end
end

task =
  if mode == "none",
    do: nil,
    else: Task.async(fn -> MambaRacePrep.loop(paths, chunk_opts, dataset_opts, mode, out, 0) end)

Output.puts("Training #{steps} steps with background prep mode=#{mode}")
t_train = System.monotonic_time(:millisecond)

trainer =
  Enum.reduce(1..steps, trainer, fn step, tr ->
    File.write!(Path.join(out, "last_step.json"),
      Jason.encode!(%{step: step, mode: mode, elapsed_ms: System.monotonic_time(:millisecond) - t_train}))
    {next, m} = Imitation.train_step(tr, batch, nil)
    loss = Nx.to_number(m.loss)
    unless is_number(loss), do: raise("nonfinite loss at step #{step}")
    if rem(step, 500) == 0 do
      bg = case File.read(Path.join(out, "background.json")) do
        {:ok, b} -> Jason.decode!(b)["iterations"]
        _ -> 0
      end
      Output.puts("step #{step}: loss #{Float.round(loss, 5)}  prep loops so far: #{bg}")
    end
    next
  end)

iterations =
  if task do
    send(task.pid, :stop)
    Task.await(task, :infinity)
  else
    0
  end

_ = trainer
File.write!(Path.join(out, "completed.json"),
  Jason.encode!(%{steps: steps, mode: mode, prep_iterations: iterations, chunk: chunk_no,
    train_ms: System.monotonic_time(:millisecond) - t_train, total_ms: System.monotonic_time(:millisecond) - t_start}))
Output.success("PASS: #{steps} steps with #{iterations} background prep loops (mode=#{mode})")
