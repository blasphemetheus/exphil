# Muzak Mutation Testing Configuration
# See: https://hexdocs.pm/muzak/muzak.html
#
# Run with: mix muzak
# Run specific profile: mix muzak --profile ci

[
  # Default profile for local development
  default: [
    # Focus on core modules that are most critical
    only: [
      "lib/exphil/embeddings",
      "lib/exphil/training/config.ex",
      "lib/exphil/training/targets.ex"
    ],
    # Minimum percentage of mutations that must be caught
    min_coverage: 70.0,
    # Exclude slow tests to keep mutation runs fast
    test_command: "mix test --exclude slow --exclude integration --exclude external"
  ],

  # CI profile - more thorough
  ci: [
    only: [
      "lib/exphil/embeddings",
      "lib/exphil/training"
    ],
    min_coverage: 60.0,
    test_command: "mix test --exclude integration --exclude external"
  ],

  # Quick profile for rapid iteration
  quick: [
    only: [
      "lib/exphil/embeddings/player.ex"
    ],
    min_coverage: 50.0,
    test_command: "mix test test/exphil/embeddings/player_test.exs"
  ]
]
