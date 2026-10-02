# Divergence survey after the Seed.mismatch slot fix: which matchups/stages drift, and when?
alias ExPhil.Data.Peppi
alias ExPhil.Sim.Seed
[n_arg] = System.argv()
split = "checkpoints/coh_base/split.json" |> File.read!() |> Jason.decode!()
files = split["validation"] ++ Enum.take(split["train"], String.to_integer(n_arg))
rows =
  for path <- files do
    {:ok, m} = Peppi.metadata(path)
    chars = m.players |> Enum.sort_by(& &1.port) |> Enum.map(& &1.character_name)
    ports = m.players |> Enum.map(& &1.port) |> Enum.sort()
    target = min(m.duration_frames - 200, 3000)
    res =
      try do
        {:ok, s} = Seed.from_replay(path, frame: target, tolerance: 0.01)
        case s.divergence do
          nil -> :clean
          d -> {:diverged, elem(d, 0), elem(d, 2)}
        end
      rescue
        e -> {:error, Exception.message(e) |> String.slice(0, 80)}
      end
    IO.puts("SURVEY stage=#{m.stage} chars=#{inspect(chars)} ports=#{inspect(ports)} to=#{target} -> #{inspect(res)}")
    %{stage: m.stage, chars: chars, res: res}
  end
tag = fn %{res: r} -> if r == :clean, do: :clean, else: elem(r, 0) end
IO.puts("SURVEY by stage: " <> inspect(rows |> Enum.group_by(& &1.stage) |> Map.new(fn {k, v} -> {k, Enum.frequencies_by(v, tag)} end)))
IO.puts("SURVEY by opponent: " <> inspect(rows |> Enum.group_by(fn r -> Enum.reject(r.chars, &(&1 == "Fox")) end) |> Map.new(fn {k, v} -> {k, Enum.frequencies_by(v, tag)} end)))
div_frames = for %{res: {:diverged, f, _}} <- rows, do: f
IO.puts("SURVEY first-divergence frames: #{inspect(Enum.sort(div_frames))}")
