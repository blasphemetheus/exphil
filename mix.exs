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
      source_url: "https://github.com/blasphemetheus/exphil",
      homepage_url: "https://github.com/blasphemetheus/exphil",
      docs: docs(),

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

  defp docs do
    [
      main: "ExPhil",
      extras: [
        "README.md",
        # Core Guides
        "docs/reference/ARCHITECTURE.md",
        "docs/guides/TRAINING.md",
        "docs/guides/TRAINING_CHEATSHEET.md",
        "docs/guides/TRAINING_FEATURES.md",
        "docs/guides/INFERENCE.md",
        "docs/guides/DOLPHIN.md",
        "docs/guides/SCRIPTS.md",
        # Reference
        "docs/reference/GOTCHAS.md",
        "docs/guides/TESTING.md",
        "docs/planning/GOALS.md",
        "docs/planning/PROJECT_ROADMAP.md",
        # Research & Planning
        "docs/research/RESEARCH.md",
        "docs/research/BITTER_LESSON_PLAN.md",
        "docs/planning/MEWTWO_TRAINING_PLAN.md",
        # Advanced Topics
        "docs/reference/EMBEDDING_DIMENSIONS.md",
        "docs/internals/GPU_OPTIMIZATIONS.md",
        "docs/internals/MAMBA_OPTIMIZATIONS.md",
        "docs/reference/SELF_PLAY_ARCHITECTURE.md",
        "docs/research/IC_TECH_FEATURE_BLOCK.md",
        # Implementation Notes
        "docs/internals/AXON_ORTHOGONAL_INIT_FIX.md",
        "docs/planning/TRAINING_IMPROVEMENTS.md",
        # Deployment
        "docs/operations/docker-workflow.md",
        "docs/operations/REPLAY_STORAGE.md",
        "docs/operations/RUNPOD_FILTER.md",
        "docs/operations/RCLONE_GDRIVE.md",
        # Meta
        "CHANGELOG.md",
        "LICENSE",
        "docs/guides/HEX_PUBLISHING.md"
      ],
      groups_for_extras: [
        "Getting Started": ~r/README/,
        "Core Guides": ~r/(ARCHITECTURE|TRAINING|INFERENCE|DOLPHIN|SCRIPTS)/,
        Reference: ~r/(GOTCHAS|TESTING|GOALS|PROJECT_ROADMAP)/,
        "Research & Planning": ~r/(RESEARCH|BITTER_LESSON|MEWTWO_TRAINING)/,
        "Advanced Topics": ~r/(EMBEDDING|GPU|MAMBA|SELF_PLAY|IC_TECH)/,
        "Implementation Notes": ~r/(AXON_ORTHOGONAL|TRAINING_IMPROVEMENTS)/,
        Deployment: ~r/(docker|REPLAY_STORAGE|RUNPOD|RCLONE)/,
        Meta: ~r/(CHANGELOG|LICENSE|HEX_PUBLISHING)/
      ],
      groups_for_modules: [
        Training: [
          ExPhil.Training.Imitation,
          ExPhil.Training.Config,
          ExPhil.Training.Data,
          ExPhil.Training.PPO,
          ExPhil.Training.Output,
          ExPhil.Training.EarlyStopping,
          ExPhil.Training.Checkpoint,
          ExPhil.Training.EMA,
          ExPhil.Training.LRFinder,
          ExPhil.Training.Augmentation
        ],
        Embeddings: [
          ExPhil.Embeddings,
          ExPhil.Embeddings.Game,
          ExPhil.Embeddings.Player,
          ExPhil.Embeddings.Controller,
          ExPhil.Embeddings.Primitives,
          ExPhil.Embeddings.KMeans
        ],
        Networks: [
          ExPhil.Networks.Policy,
          ExPhil.Networks.Value,
          ExPhil.Networks.ActorCritic,
          ExPhil.Networks.GatedSSM,
          ExPhil.Networks.Mamba,
          ExPhil.Networks.Attention,
          ExPhil.Networks.Recurrent,
          ExPhil.Networks.Hybrid
        ],
        Bridge: [
          ExPhil.Bridge.MeleePort,
          ExPhil.Bridge.AsyncRunner,
          ExPhil.Bridge.GameState,
          ExPhil.Bridge.Player,
          ExPhil.Bridge.ControllerState,
          ExPhil.Bridge.ControllerInput
        ],
        Agents: [
          ExPhil.Agents.Agent,
          ExPhil.Agents.Supervisor
        ],
        "Self-Play": [
          ExPhil.Training.SelfPlay.SelfPlayEnv,
          ExPhil.Training.SelfPlay.OpponentPool,
          ExPhil.Training.SelfPlay.LeagueTrainer
        ],
        Utilities: [
          ExPhil.Training.Utils,
          ExPhil.Training.GPUUtils,
          ExPhil.Training.Metrics,
          ExPhil.Training.Registry,
          ExPhil.Training.Help
        ]
      ],
      nest_modules_by_prefix: [
        ExPhil.Training,
        ExPhil.Embeddings,
        ExPhil.Networks,
        ExPhil.Bridge,
        ExPhil.Agents
      ]
    ]
  end

  defp deps do
    [
      # ML Core
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:exla, "~> 0.9"},
      {:polaris, "~> 0.1"},

      # ML Architecture Library (extracted generic architectures)
      # Uses local path for dev, GitHub for Docker builds
      {:edifice, edifice_dep()},

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

      # Timezone support (for Central time timestamps)
      {:tz, "~> 0.28"},
      # For PyTorch Port communication
      {:msgpax, "~> 2.4"},
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

  defp edifice_dep do
    if System.get_env("DOCKER_BUILD") do
      [github: "blasphemetheus/edifice"]
    else
      [path: "../edifice"]
    end
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
