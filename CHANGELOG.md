# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ice Climbers added to supported characters
- Comprehensive ExDoc configuration with module groups
- Top-level ExPhil module with full documentation
- @spec annotations across bridge, agents, and training modules
- Hex publishing readiness roadmap (docs/HEX_PUBLISHING.md)

### Changed
- Default embedding mode now uses learned embeddings (288 dims vs 1204 one-hot)
- Updated architecture documentation to reflect current system

## [0.1.0] - 2026-01-30

### Added
- Initial release
- Imitation learning pipeline (behavioral cloning from Slippi replays)
- Multiple backbone architectures: MLP, LSTM, GRU, Mamba, Attention, Jamba
- Dolphin integration via libmelee Python bridge
- AsyncRunner for 60fps inference with decoupled frame reading
- Training presets for quick iteration and production training
- Frame delay augmentation for online play robustness
- K-means stick discretization for improved input precision
- Model EMA and cosine annealing with warm restarts
- Embedding caching for 2-3x training speedup
- PPO trainer infrastructure (self-play in progress)

### Target Characters
- Mewtwo
- Mr. Game & Watch
- Link
- Ganondorf
- Zelda
- Ice Climbers

[Unreleased]: https://github.com/blasphemetheus/exphil/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/blasphemetheus/exphil/releases/tag/v0.1.0
