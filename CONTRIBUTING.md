# Contributing to ExPhil

Thanks for your interest in contributing to ExPhil! This document covers the basics for getting started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/blasphemetheus/exphil.git
cd exphil

# Install Elixir dependencies
mix deps.get

# Run the interactive setup wizard
mix exphil.setup

# Run tests
mix test
```

### Prerequisites

- Elixir >= 1.18
- Erlang/OTP (compatible with your Elixir version)
- Rust toolchain (for Peppi NIF replay parser)
- Optional: CUDA/ROCm GPU for training with EXLA

### Companion Library

ExPhil depends on [Edifice](https://github.com/blasphemetheus/edifice) for generic ML architectures. For local development, clone it alongside ExPhil:

```
melee/
  exphil/    # This repo
  edifice/   # ML architecture library
```

## Running Tests

```bash
# Fast unit tests (default, ~2 min)
mix test

# Include slow tests
mix test.slow

# All tests including integration
mix test.all

# Run a specific test file
mix test test/exphil/embeddings/player_test.exs

# Run a specific test by line number
mix test test/exphil/embeddings/player_test.exs:42
```

See `docs/guides/TESTING.md` for full testing documentation.

## Project Structure

```
lib/exphil/
  embeddings/   # State embedding (player, game, controller)
  networks/     # Policy, value, backbone networks
  training/     # Imitation learning, PPO, data pipeline
  bridge/       # Dolphin/libmelee integration
  agents/       # Agent GenServer for live play
  rewards/      # Reward shaping
```

## Code Style

- Follow standard Elixir conventions (`mix format` before committing)
- Use `ExPhil.Training.Output` for all script output (timestamps, colors, progress bars)
- New CLI flags must be added to `@valid_flags` in `config.ex` and documented in `docs/guides/TRAINING.md`
- See `CLAUDE.md` for detailed coding standards and patterns

## Adding a New Backbone Architecture

1. Create a module in `lib/exphil/networks/` or use an existing Edifice architecture
2. Register it in `lib/exphil/networks/backbone.ex`
3. Add tests in `test/exphil/networks/`
4. Document in `docs/reference/architectures/`

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Run `mix test` to ensure nothing is broken
5. Run `mix format` to ensure consistent formatting
6. Open a pull request with a clear description

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Elixir/OTP version and OS

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
