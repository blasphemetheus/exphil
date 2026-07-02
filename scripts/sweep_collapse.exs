#!/usr/bin/env elixir
# Fast sweep to find settings that prevent mode collapse on 200-file datasets
# Each config runs 2 epochs (~5 min), checks action diversity at end
#
# Run: mix run scripts/sweep_collapse.exs 2>&1 | tee logs/sweep_collapse.log

alias ExPhil.Training.{Config, Pipeline, Trainer, Output, Imitation}

Output.banner("Mode Collapse Sweep")

# Shared pipeline — parse and embed once, reuse across configs
base_opts = Config.parse_args([
  "--backbone", "mamba", "--replays", "./replays/huggingface",
  "--max-files", "200", "--batch-size", "16", "--seed", "42"
]) |> Config.validate!() |> Config.ensure_checkpoint_name()

Output.puts("Setting up shared pipeline (200 files)...")
pipeline = Pipeline.setup!(base_opts)
Output.puts("  Train: #{pipeline.train_dataset.size} frames")
Output.puts("  Val batches: #{Pipeline.val_batch_count(pipeline)}")

# Configs to test — each overrides specific settings
# Only 3 configs — EXLA JIT cache fills GPU memory, can't run many sequentially
# These test the most likely root causes
configs = [
  %{name: "baseline (gamma=3, lr=1e-4, cosine)",
    focal_gamma: 3.0, learning_rate: 1.0e-4, lr_schedule: :cosine_restarts,
    button_weight: 2.0, warmup_steps: 1},

  %{name: "gamma=1 + lr=3e-4 + constant",
    focal_gamma: 1.0, learning_rate: 3.0e-4, lr_schedule: :constant,
    button_weight: 2.0, warmup_steps: 1},

  %{name: "gamma=1 + lr=3e-4 + constant + bw=5",
    focal_gamma: 1.0, learning_rate: 3.0e-4, lr_schedule: :constant,
    button_weight: 5.0, warmup_steps: 1},
]

results = Enum.map(configs, fn config ->
  Output.puts("\n" <> String.duplicate("=", 60))
  Output.puts("  #{config.name}")
  Output.puts(String.duplicate("=", 60))

  # Override pipeline opts for this config
  run_opts = Keyword.merge(base_opts, [
    focal_gamma: config.focal_gamma,
    focal_loss: config.focal_gamma > 0,
    learning_rate: config.learning_rate,
    lr_schedule: config.lr_schedule,
    button_weight: config.button_weight,
    warmup_steps: config.warmup_steps,
    epochs: 2
  ])

  run_pipeline = %{pipeline | resolved_opts: run_opts}

  try do
    trainer = Trainer.new(run_pipeline, run_opts)

    # Train 2 epochs
    {stream, _} = Pipeline.batch_stream(run_pipeline, [])
    batches = stream |> Enum.to_list()

    # Epoch 1
    {trainer, losses1} = Enum.reduce(batches, {trainer, []}, fn batch, {t, losses} ->
      {new_t, metrics} = Imitation.train_step(t, batch, nil)
      {new_t, [Nx.to_number(metrics.loss) | losses]}
    end)
    avg1 = Enum.sum(losses1) / length(losses1)

    # Epoch 2
    {trainer, losses2} = Enum.reduce(batches, {trainer, []}, fn batch, {t, losses} ->
      {new_t, metrics} = Imitation.train_step(t, batch, nil)
      {new_t, [Nx.to_number(metrics.loss) | losses]}
    end)
    avg2 = Enum.sum(losses2) / length(losses2)

    # Check action diversity
    val_batches = pipeline.val_batches || []
    diversity = if length(val_batches) >= 3 do
      sample = Enum.take(val_batches, 5)
      combos = Enum.flat_map(sample, fn batch ->
        {btn_logits, mx_logits, _, _, _, _} =
          trainer.predict_fn.(trainer.policy_params, batch.states)

        pred_buttons = Nx.greater(Nx.sigmoid(btn_logits), 0.5) |> Nx.as_type(:u8)
        pred_mx = Nx.argmax(mx_logits, axis: -1)

        batch_size = elem(Nx.shape(pred_buttons), 0)
        for i <- 0..min(batch_size - 1, 7) do
          {Nx.to_flat_list(pred_buttons[i]), Nx.to_number(pred_mx[i])}
        end
      end)
      MapSet.new(combos) |> MapSet.size()
    else
      0
    end

    # Check button press rates
    any_buttons = if length(val_batches) >= 1 do
      batch = hd(val_batches)
      {btn_logits, _, _, _, _, _} = trainer.predict_fn.(trainer.policy_params, batch.states)
      pred_rate = Nx.greater(Nx.sigmoid(btn_logits), 0.5) |> Nx.mean() |> Nx.to_number()
      pred_rate > 0.001
    else
      false
    end

    Output.puts("  Epoch 1 loss: #{Float.round(avg1, 2)}")
    Output.puts("  Epoch 2 loss: #{Float.round(avg2, 2)}")
    Output.puts("  Action diversity: #{diversity}")
    Output.puts("  Any buttons pressed: #{any_buttons}")

    # Force cleanup — release trainer's GPU tensors and JIT caches
    :erlang.garbage_collect()
    Process.sleep(1000)
    :erlang.garbage_collect()

    %{name: config.name, loss1: avg1, loss2: avg2, diversity: diversity, buttons: any_buttons}
  rescue
    e ->
      Output.error("  FAILED: #{Exception.message(e) |> String.slice(0, 80)}")
      :erlang.garbage_collect()
      Process.sleep(1000)
      :erlang.garbage_collect()
      %{name: config.name, loss1: nil, loss2: nil, diversity: 0, buttons: false}
  end
end)

# Summary table
Output.puts("\n" <> String.duplicate("=", 70))
Output.puts("  SWEEP RESULTS")
Output.puts(String.duplicate("=", 70))

headers = ["Config", "Loss E1", "Loss E2", "Diversity", "Buttons?"]
rows = Enum.map(results, fn r ->
  [
    r.name,
    if(r.loss1, do: to_string(Float.round(r.loss1, 2)), else: "FAIL"),
    if(r.loss2, do: to_string(Float.round(r.loss2, 2)), else: "FAIL"),
    to_string(r.diversity),
    if(r.buttons, do: "YES", else: "no")
  ]
end)

Output.puts_raw(Output.table(headers, rows))

best = Enum.filter(results, & &1.diversity > 1) |> Enum.max_by(& &1.diversity, fn -> nil end)
if best do
  Output.puts("\n  Best: #{best.name} (diversity=#{best.diversity})")
else
  Output.puts("\n  No config achieved diversity > 1. Try entropy regularization.")
end
