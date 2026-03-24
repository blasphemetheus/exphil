#!/usr/bin/env elixir
# Sweep focal_gamma and button_weight to find optimal mode collapse settings.
#
# Usage: mix run scripts/sweep_mode_collapse.exs [--replays DIR] [--max-files N] [--epochs N]
#
# Runs a grid of (focal_gamma, button_weight) combinations and reports
# per-button press-rate calibration for each. Results written to logs/sweep_results.txt.
#
# Default: 20 files, 5 epochs, MLP backbone

alias ExPhil.Training.Output

# Parse args
args = System.argv()

replays = case Enum.find_index(args, &(&1 == "--replays")) do
  nil -> "./replays/mewtwo"
  i -> Enum.at(args, i + 1)
end

max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> "20"
  i -> Enum.at(args, i + 1)
end

epochs = case Enum.find_index(args, &(&1 == "--epochs")) do
  nil -> "5"
  i -> Enum.at(args, i + 1)
end

# Grid
focal_gammas = [1.0, 2.0, 3.0]
button_weights = [1.0, 2.0, 3.0, 5.0]

IO.puts("\n=== Mode Collapse Parameter Sweep ===")
IO.puts("Replays: #{replays}, max_files: #{max_files}, epochs: #{epochs}")
IO.puts("Grid: focal_gamma=#{inspect(focal_gammas)} × button_weight=#{inspect(button_weights)}")
IO.puts("Total runs: #{length(focal_gammas) * length(button_weights)}\n")

results_path = "logs/sweep_results.txt"
File.mkdir_p!("logs")
File.write!(results_path, "focal_gamma,button_weight,val_loss,A_ratio,B_ratio,X_ratio,Y_ratio,Z_ratio,L_ratio,R_ratio\n")

for gamma <- focal_gammas, bw <- button_weights do
  name = "sweep_g#{gamma}_bw#{bw}"
  IO.puts("--- Running: focal_gamma=#{gamma}, button_weight=#{bw} ---")

  cmd = ~s(mix run scripts/train_from_replays.exs ) <>
    ~s(--epochs #{epochs} --max-files #{max_files} --hidden-sizes 64,64 ) <>
    ~s(--backbone mlp --batch-size 64 --replays #{replays} ) <>
    ~s(--focal-gamma #{gamma} --button-weight #{bw} ) <>
    ~s(--name #{name} --no-register 2>&1)

  output = :os.cmd(String.to_charlist(cmd)) |> to_string()
  log_path = "logs/#{name}.log"
  File.write!(log_path, output)

  # Extract last epoch's press rates
  lines = String.split(output, "\n")
  val_loss_line = Enum.find(lines, &String.contains?(&1, "Epoch #{epochs} complete"))
  val_loss = case val_loss_line && Regex.run(~r/val_loss=([\d.]+)/, val_loss_line) do
    [_, vl] -> vl
    _ -> "?"
  end

  # Find last set of pred/actual lines
  pred_lines = lines
    |> Enum.filter(&String.contains?(&1, "pred="))
    |> Enum.take(-8)

  ratios = Enum.map(pred_lines, fn line ->
    case Regex.run(~r/pred=\s*([\d.]+)%\s*actual=\s*([\d.]+)%/, line) do
      [_, pred, actual] ->
        {p, _} = Float.parse(pred)
        {a, _} = Float.parse(actual)
        if a > 0.5, do: Float.round(p / a, 2), else: 0.0
      _ -> 0.0
    end
  end)

  ratio_str = ratios |> Enum.map(&to_string/1) |> Enum.join(",")
  File.write!(results_path, "#{gamma},#{bw},#{val_loss},#{ratio_str}\n", [:append])

  IO.puts("  val_loss=#{val_loss}, ratios=#{inspect(ratios)}")
  IO.puts("  Log: #{log_path}")
end

IO.puts("\n=== Sweep complete ===")
IO.puts("Results: #{results_path}")
IO.puts("")
