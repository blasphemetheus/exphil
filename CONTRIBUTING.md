# Contributing to ExPhil

Thanks for your interest in contributing to ExPhil! This document covers the basics for getting started.

## Development Setup

```bash
# Clone all three into the same parent directory
git clone https://github.com/blasphemetheus/edifice.git
git clone https://github.com/blasphemetheus/libmelee_ex.git
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

ExPhil currently defaults to local sibling dependencies for
[Edifice](https://github.com/blasphemetheus/edifice) (generic ML architectures)
and [libmelee_ex](https://github.com/blasphemetheus/libmelee_ex) (Dolphin integration):

```
melee/
  exphil/    # This repo
  edifice/   # ML architecture library
  libmelee_ex/ # Native game bridge
```

Use `EDIFICE_PATH` and `LIBMELEE_EX_PATH` to select other local checkouts.
Reproducible remote defaults and compatible locked revisions are tracked in
[R4](docs/planning/REPO_IMPROVEMENTS.md#r4--dependencies-and-run-provenance);
the commands above do not yet pin a validated cross-repository revision set.

If training is active, use an independent development environment with separate
source, dependencies, and build output. Shared native dependency rebuilds and
source changes picked up by multi-stage launchers can disrupt training.

## Running Tests

```bash
# Fast unit tests (default, ~2 min)
mix test

# Include slow tests
mix test.slow

# Include slow, integration, and external tests (other exclusions remain)
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
- New training flags belong in the table in `lib/exphil/training/config/parser.ex`, with defaults and validation in Config. Accepted flags are derived from the table; regenerate the training flag reference with `ExPhil.Training.Config.FlagDocs.write!()` and run the flag parity tests.
- See `CLAUDE.md` for detailed coding standards and patterns

## Adding a New Backbone Architecture

1. Implement generic architectures in Edifice or use an existing Edifice architecture
2. Add the ExPhil defaults, build recipe, and output rule to `@backbone_specs` in `lib/exphil/training/config.ex`; bespoke adapters live in `lib/exphil/networks/policy/backbone.ex`
3. Add tests in `test/exphil/networks/`, including the spec/registry contract where applicable
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
