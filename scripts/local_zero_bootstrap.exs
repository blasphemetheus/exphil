# Prepare an isolated training round. Run from the ExPhil repository root.
out = List.first(System.argv()) || "eval_runs/0915_local_zero/bootstrap"
File.mkdir_p!(Path.dirname(out))
File.mkdir!(out)
manifest = "eval_runs/0915_float_input/coverage03/mined.json" |> File.read!() |> JSON.decode!()
sources = manifest["entries"] |> Enum.map(& &1["slp"]) |> Enum.uniq() |> Enum.sort()
{train, held} = Enum.split_with(sources, &(Path.basename(&1) == "r1.slp"))
sha = fn p -> :crypto.hash(:sha256, File.read!(p)) |> Base.encode16(case: :lower) end

File.write!(
  Path.join(out, "split.json"),
  JSON.encode!(%{
    train_sources: train,
    heldout_sources: held,
    reaction_delay: 0,
    parser_semantics: "misc_as_live_parity_hurtbox_state_v2",
    parser_sha256: sha.("native/exphil_peppi/src/lib.rs"),
    source_hashes: Map.new(sources, &{&1, sha.(&1)}),
    recipe: %{
      epochs: 21,
      hidden_size: 64,
      window: 16,
      queue_depth: 1,
      head: "autoregressive",
      precision: "f32",
      recurrent_state: "zeros"
    }
  })
)

args = [
  "run",
  "scripts/dagger_drill.exs",
  "--expert",
  "multishine",
  "--fixture",
  "test/fixtures/replays/fox_multishine_closed_d1.slp",
  "--rollouts",
  Enum.join(train, ","),
  "--recurrent-state",
  "zeros",
  "--precision",
  "f32",
  "--initial-out",
  Path.join(out, "initial.bin"),
  "--hidden-size",
  "64",
  "--window",
  "16",
  "--action-delay",
  "0",
  "--multi-delay",
  "0",
  "--with-delay-id",
  "--queue-depth",
  "1",
  "--prev-action",
  "--prev-action-dropout",
  "0.0",
  "--head",
  "autoregressive",
  "--clean-loss",
  "--max-epochs",
  "21",
  "--target-loss",
  "0.0",
  "--out",
  Path.join(out, "candidate.bin")
]

File.write!(Path.join(out, "train_args.json"), JSON.encode!(args))
IO.puts("Prepared #{out}: #{length(train)} training games, #{length(held)} held-out games")
