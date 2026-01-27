defmodule ExPhil.MixProject do
  use Mix.Project

  def project do
    [
      app: :exphil,
      version: "0.1.0",
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),

      # Docs
      name: "ExPhil",
      description: "Elixir-based Melee AI for lower-tier characters",
      source_url: "https://github.com/yourusername/exphil",

      # Test coverage
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.html": :test,
        "coveralls.json": :test,
        "test.all": :test,
        "test.slow": :test,
        "test.fast": :test,
        "test.coverage": :test,
        muzak: :test
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {ExPhil.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # ML Core
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:exla, "~> 0.9"},
      {:polaris, "~> 0.1"},

      # Data Processing
      {:explorer, "~> 0.9"},
      {:yaml_elixir, "~> 2.9"},

      # Model Export (blasphemetheus fork with Axon 0.8+ fixes)
      # Use override to allow local development with path-based dep
      {:axon_onnx, github: "blasphemetheus/axon_onnx", branch: "runtime-fixes", override: true},
      # ONNX Runtime for inference (used in tests)
      {:ortex, "~> 0.1", only: [:dev, :test]},

      # Rust NIFs (for Peppi replay parsing)
      {:rustler, "~> 0.35"},

      # Python Interop (optional, for melee_bridge)
      {:pythonx, "~> 0.3", optional: true},

      # Telemetry & Metrics
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 1.0"},

      # Serialization
      {:jason, "~> 1.4"},
      {:msgpax, "~> 2.4"},  # For PyTorch Port communication
      {:req, "~> 0.5"},

      # Visualization (for training plots)
      {:vega_lite, "~> 0.1"},

      # Dev & Test
      {:kino, "~> 0.14", only: :dev},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:excoveralls, "~> 0.18", only: :test},
      {:mox, "~> 1.1", only: :test},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:muzak, "~> 1.1", only: :test}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "cmd --cd priv/python pip install -r requirements.txt"],
      "exphil.train": ["run scripts/train.exs"],
      "exphil.eval": ["run scripts/eval.exs"],
      "exphil.parse_replays": ["run scripts/parse_replays.exs"],

      # Test aliases for running different test categories
      # See docs/TESTING.md for full documentation
      # Default: fast unit tests only
      "test.fast": ["test"],
      "test.slow": ["test", "--include", "slow"],
      "test.all": [
        "test",
        "--include",
        "slow",
        "--include",
        "integration",
        "--include",
        "external"
      ],
      "test.integration": ["test", "--only", "integration"],
      "test.coverage": ["coveralls.html"],

      # Benchmark tests
      "test.benchmark": ["test", "--only", "benchmark"],
      "test.benchmark.update": ["cmd", "BENCHMARK_UPDATE=1 mix test --only benchmark"],

      # Snapshot tests
      "test.snapshot": ["test", "--only", "snapshot"],
      "test.snapshot.update": ["cmd", "SNAPSHOT_UPDATE=1 mix test --only snapshot"],

      # Mutation testing
      "test.mutate": ["muzak"],
      "test.mutate.quick": ["muzak", "--profile", "quick"]
    ]
  end
end
