# AI-generated; method and authorship: docs/guides/MAMBA_CARRY_RESOLUTION.md.
# Run through scripts/run_recurrent_validation.exs.
Code.require_file("lib/benchmark_sweep.exs", __DIR__)
Code.require_file("lib/mamba_precision_experiment.exs", __DIR__)
Code.compile_file("lib/exphil/embeddings/player.ex")
Code.compile_file("lib/exphil/embeddings/game/spatial.ex")

Code.require_file("lib/mamba_carry_resolution.exs", __DIR__)

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      out: :string,
      ticks: :integer,
      rows: :integer,
      capture: :string,
      mode: :string,
      capture_mode: :string,
      policy: :string,
      replay: :string,
      character: :string,
      stagger: :boolean,
      offset: :integer,
      capture_only: :boolean,
      require_pass: :boolean
    ]
  )

MambaCarryResolution.run(opts)
