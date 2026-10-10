# Build the expert-recovery index for the expert-distribution labeler
# (scripts/lib/expert_recovery_labeler.exs): every offstage, airborne,
# labelable frame of N expert FD Fox games -> feature vector + the input the
# expert gave there + hold flag.
#
#   mix run scripts/build_expert_recovery_index.exs --split checkpoints/coh_X/split.json \
#     --games 400 --out data/silent_fall/expert_recovery_index.bin
#
# Games come from the split's TRAIN list (the held-out validation games stay
# held out), FD only, own player = the Fox port (mirrors the training
# parse: Peppi.to_training_frames with remap_ports). Prints the hold share
# and the label mix so the set can be checked against the corpus (~0.76).
require Logger
Logger.configure(level: :warning)
# runnable without mix (elixir -pa _build/dev/lib/*/ebin) while a training beam is alive
for app <- [:nx, :jason], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")

alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, bad} = OptionParser.parse(System.argv(), strict: [split: :string, games: :integer, out: :string, stage: :integer, window: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
split = opts[:split] || raise("--split required")
out = opts[:out] || "data/silent_fall/expert_recovery_index.bin"
want = opts[:games] || 400
stage = opts[:stage] || 32
# --window wide (v4, 10-09): also grounded / above-stage states within L.near_edge() of the edge
# --window air (v5, 10-10): wide + the airborne no-jump drift within L.air_reach() of the edge
window =
  case opts[:window] do
    "wide" -> :wide
    "air" -> :air
    nil -> :offstage
    other -> raise("--window #{other}: expected wide | air")
  end

files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("train")
Output.banner("Expert recovery index")
Output.config([{"split", split}, {"train files", length(files)}, {"games wanted", want}, {"stage", stage}, {"window", window}, {"out", out}])

{rows, used} =
  Enum.reduce_while(files, {[], 0}, fn path, {acc, used} ->
    if used >= want do
      {:halt, {acc, used}}
    else
      case Peppi.metadata(path) do
        {:ok, %{stage: ^stage} = meta} ->
          own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
          opp = own && Enum.find(meta.players, &(&1.port != own.port))

          if own && opp do
            {:ok, replay} = Peppi.parse(path, player_port: own.port)
            frames =
              replay
              |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
              |> Enum.reject(&(&1.game_state.frame < 0))

            new = L.index_rows(frames, stage, window)
            if rem(used + 1, 25) == 0, do: Output.puts("  #{used + 1} games")
            {:cont, {[new | acc], used + 1}}
          else
            {:cont, {acc, used}}
          end

        _ ->
          {:cont, {acc, used}}
      end
    end
  end)

rows = rows |> Enum.reverse() |> Enum.concat()
n = length(rows)
holds = Enum.count(rows, &elem(&1, 2))
b_share = Enum.count(rows, fn {_, r, _} -> Enum.at(r, 5) > 0.5 end) / max(n, 1)
jump_share = Enum.count(rows, fn {_, r, _} -> Enum.at(r, 6) > 0.5 or Enum.at(r, 7) > 0.5 end) / max(n, 1)
up_share = Enum.count(rows, fn {_, r, _} -> Enum.at(r, 1) >= 0.75 end) / max(n, 1)

index = L.pack(rows) |> Map.put(:window, window) |> Map.put(:meta, %{games: used, split: split, stage: stage, window: window, built_at: DateTime.utc_now() |> DateTime.to_iso8601()})
L.save(index, out)

Output.puts("RESULT expert recovery index: #{used} games, #{n} #{window} rows, hold share #{Float.round(holds / max(n, 1), 3)}, " <>
  "B #{Float.round(100 * b_share, 1)} %, jump #{Float.round(100 * jump_share, 1)} %, stick up #{Float.round(100 * up_share, 1)} % -> #{out}")
