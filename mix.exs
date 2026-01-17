defmodule ExPhil.MixProject do
  use Mix.Project

  def project do
    [
      app: :exphil,
      version: "0.1.0",
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),

      # Docs
      name: "ExPhil",
      description: "Elixir-based Melee AI for lower-tier characters",
      source_url: "https://github.com/yourusername/exphil"
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

      # Rust NIFs (for Peppi replay parsing)
      {:rustler, "~> 0.35"},

      # Python Interop (optional, for melee_bridge)
      {:pythonx, "~> 0.3", optional: true},

      # Telemetry & Metrics
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 1.0"},

      # JSON
      {:jason, "~> 1.4"},

      # Dev & Test
      {:kino, "~> 0.14", only: :dev},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "cmd --cd priv/python pip install -r requirements.txt"],
      "exphil.train": ["run scripts/train.exs"],
      "exphil.eval": ["run scripts/eval.exs"],
      "exphil.parse_replays": ["run scripts/parse_replays.exs"]
    ]
  end
end
