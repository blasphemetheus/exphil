#!/usr/bin/env elixir
# Hyperparameter sweep — runs each config as a separate OS process
# to avoid JIT cache OOM between configs.
#
# Usage:
#   mix run scripts/sweep.exs
#   mix run scripts/sweep.exs --max-files 100 --epochs 3
#   mix run scripts/sweep.exs --config sweep_configs.json
#
# Each config spawns a fresh `mix run scripts/train.exs` process,
# captures the output, extracts metrics, and produces a comparison table.

alias ExPhil.Training.Output

# Parse sweep-level args
args = System.argv()
max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> 200
  i -> String.to_integer(Enum.at(args, i + 1, "200"))
end

epochs = case Enum.find_index(args, &(&1 == "--epochs")) do
  nil -> 2
  i -> String.to_integer(Enum.at(args, i + 1, "2"))
end

replays = case Enum.find_index(args, &(&1 == "--replays")) do
  nil -> "./replays/huggingface"
  i -> Enum.at(args, i + 1, "./replays/huggingface")
end

batch_size = case Enum.find_index(args, &(&1 == "--batch-size")) do
  nil -> 16
  i -> String.to_integer(Enum.at(args, i + 1, "16"))
end

Output.banner("Hyperparameter Sweep")
Output.puts("  Max files: #{max_files}")
Output.puts("  Epochs: #{epochs}")
Output.puts("  Batch size: #{batch_size}")
Output.puts("")

# Define configs to sweep
configs = [
  %{name: "baseline",
    args: []},

  %{name: "entropy=0.01",
    args: ["--entropy-weight", "0.01"]},

  %{name: "entropy=0.05",
    args: ["--entropy-weight", "0.05"]},

  %{name: "entropy=0.01+lr=3e-4+const",
    args: ["--entropy-weight", "0.01", "--learning-rate", "3e-4", "--lr-schedule", "constant"]},

  %{name: "entropy=0.01+gamma=1",
    args: ["--entropy-weight", "0.01", "--focal-gamma", "1.0"]},

  %{name: "entropy=0.01+bw=5",
    args: ["--entropy-weight", "0.01", "--button-weight", "5.0"]},
]

# Common args for all configs
common_args = [
  "scripts/train.exs",
  "--backbone", "mamba",
  "--replays", replays,
  "--max-files", to_string(max_files),
  "--epochs", to_string(epochs),
  "--batch-size", to_string(batch_size),
  "--seed", "42"
]

# Run each config as a separate process
results = Enum.map(configs, fn config ->
  Output.puts("=" |> String.duplicate(60))
  Output.puts("  #{config.name}")
  Output.puts("=" |> String.duplicate(60))

  name = config.name |> String.replace(~r/[^a-zA-Z0-9]/, "_") |> String.slice(0, 30)
  run_args = common_args ++ ["--name", "sweep_#{name}"] ++ config.args

  # Run as a separate OS process
  {output, exit_code} = System.cmd("mix", ["run" | run_args],
    stderr_to_stdout: true,
    env: [{"MIX_ENV", "dev"}]
  )

  # Extract metrics from output
  val_loss = case Regex.run(~r/Final val_loss: ([\d.]+)/, output) do
    [_, val] -> String.to_float(val)
    _ -> nil
  end

  train_loss = case Regex.run(~r/Final train_loss: ([\d.]+)/, output) do
    [_, val] -> String.to_float(val)
    _ -> nil
  end

  diversity = case Regex.run(~r/Action diversity: (\d+)\//, output) do
    [_, val] -> String.to_integer(val)
    _ -> 0
  end

  # Check for any button predictions
  has_buttons = output =~ ~r/pred=\s+[1-9]/

  # Check for collapse
  collapsed = output =~ "COLLAPSE" and diversity <= 1

  Output.puts("  val_loss: #{val_loss || "N/A"}")
  Output.puts("  diversity: #{diversity}")
  Output.puts("  buttons: #{has_buttons}")
  Output.puts("  exit: #{exit_code}")
  Output.puts("")

  %{
    name: config.name,
    val_loss: val_loss,
    train_loss: train_loss,
    diversity: diversity,
    has_buttons: has_buttons,
    collapsed: collapsed,
    exit_code: exit_code
  }
end)

# Summary table
Output.puts("\n" <> String.duplicate("=", 70))
Output.puts("  SWEEP RESULTS")
Output.puts(String.duplicate("=", 70))

headers = ["Config", "Val Loss", "Train Loss", "Diversity", "Buttons?", "Status"]
rows = Enum.map(results, fn r ->
  status = cond do
    r.exit_code != 0 -> "CRASH"
    r.collapsed -> "COLLAPSED"
    r.has_buttons -> "OK"
    true -> "neutral"
  end
  [
    r.name,
    if(r.val_loss, do: to_string(Float.round(r.val_loss, 4)), else: "N/A"),
    if(r.train_loss, do: to_string(Float.round(r.train_loss, 4)), else: "N/A"),
    to_string(r.diversity),
    if(r.has_buttons, do: "YES", else: "no"),
    status
  ]
end)

Output.puts_raw(Output.table(headers, rows))

# Best config
best = results
  |> Enum.filter(& &1.has_buttons)
  |> Enum.max_by(& &1.diversity, fn -> nil end)

if best do
  Output.puts("\n  Best: #{best.name} (diversity=#{best.diversity}, val_loss=#{best.val_loss})")
else
  non_crash = Enum.filter(results, & &1.exit_code == 0)
  if non_crash != [] do
    best_loss = Enum.min_by(non_crash, & (&1.val_loss || 999))
    Output.puts("\n  No config produced buttons. Best by val_loss: #{best_loss.name} (#{best_loss.val_loss})")
  else
    Output.puts("\n  All configs crashed!")
  end
end
