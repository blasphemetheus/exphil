# Elixir Testing Research for ML Codebases

Research notes on ExUnit features, patterns, and tooling relevant to ExPhil/Edifice.
Used to inform our TESTING.md guide and test infrastructure improvements.

## Our Pain Point

Claude Code instances (and developers) make targeted changes to 1-2 files, then run the
full test suite (~2469 ExPhil tests, ~1100 Edifice tests). Both suites take minutes on
BinaryBackend. The infrastructure to run targeted subsets exists (tags, file paths) but
there's no clear mapping from "I changed X" to "run tests Y."

## ExUnit Features We Should Leverage

### `mix test --stale` (highest impact)

Tracks module dependencies via a manifest file. On first run, runs everything and creates
`.elixir_ls/test_stale_manifest`. On subsequent runs, only tests whose referenced modules
(transitively) changed since last `--stale` run are re-executed.

- Source: `mix test` docs — https://hexdocs.pm/mix/Mix.Tasks.Test.html
- Works by building a cross-reference of `alias`, `import`, `use`, function calls
- Transitive: if A depends on B depends on C, changing C reruns A's tests
- First run always runs everything (builds manifest)
- Resets when you run without `--stale` (full run rebuilds manifest)

**Caveat**: Doesn't track runtime-only dependencies (e.g., reading files, ETS). For
architecture registry changes, tests that use `Edifice.build(:name)` will be stale if
`edifice.ex` changes, which is correct.

### `mix test --failed`

Reruns only tests that failed on the last run. Great for iterating on fixes.
Stored in `.elixir_ls/test_failures_manifest`.

### `mix test --partitions N` + `MIX_TEST_PARTITION`

Splits test files round-robin into N groups. Each partition runs independently.
Usage: `MIX_TEST_PARTITION=1 mix test --partitions 4`

For CI: use GitHub Actions matrix strategy to run partitions in parallel.
Not useful for local dev (our bottleneck is compile + BinaryBackend, not parallelism).

### Tag System Deep Dive

Three levels of granularity:
- `@tag :name` — single test
- `@describetag :name` — all tests in a `describe` block
- `@moduletag :name` — all tests in a module

CLI filtering:
- `--only tag` — run ONLY tests with this tag (excludes everything else)
- `--include tag` — include tests with this tag (overrides exclude)
- `--exclude tag` — exclude tests with this tag

**Key insight**: `--only` is exclusive (ignores other tags), `--include` is additive.
So `--only backbone` runs ONLY backbone-tagged tests, while
`--include slow` adds slow tests to the default set.

### `--repeat-until-failure N`

Runs specified tests up to N times, stopping on first failure. Useful for catching
flaky tests: `mix test test/foo_test.exs:42 --repeat-until-failure 100`

### Custom Formatters

ExUnit supports custom formatters for test output. Useful options:
- `junit_formatter` — JUnit XML output for CI (GitHub Actions, Jenkins)
- Custom formatters can track per-test timing, identify slow tests automatically

## How Nx/Axon/Bumblebee Organize Tests

### Axon (reference ML project in Elixir)
- `test_helper.exs` checks `USE_TORCHX` and `USE_EXLA` env vars
- Excludes doctests when running with specific backends
- Tests organized by module: `test/axon/losses_test.exs`, `test/axon/layers_test.exs`
- No domain tags beyond slow/integration — relies on file paths

### Nx
- Distributed testing infrastructure (starts peer nodes)
- Excludes `:distributed` tag when peer setup fails
- Graceful degradation when features unavailable

**Takeaway**: Even the Nx ecosystem projects don't use domain tags. We'd be ahead of
the curve by adding them, and the payoff is high given our suite size.

## Watch Mode Tools

### mix_test_interactive (recommended)
- `{:mix_test_interactive, "~> 5.1", only: :dev, runtime: false}`
- Watches filesystem, reruns on change
- Interactive mode: change filters, run stale, run failed
- Can combine with `--stale` for maximum efficiency
- Source: https://github.com/randycoulman/mix_test_interactive

### mix test.watch (simpler alternative)
- `{:mix_test_watch, "~> 1.0", only: :dev, runtime: false}`
- Just watches and reruns, less interactive
- Source: https://github.com/lpil/mix-test.watch

### fswatch + mix test (DIY)
- `fswatch -o lib/ test/ | xargs -n1 -I{} mix test --stale`
- Simple, no deps, combines well with `--stale`

## Patterns for ML-Specific Testing

### Shape Tests (our most common pattern)
Test that model outputs have expected tensor shapes. These are fast (no training),
catch most structural bugs, and should be the majority of backbone tests.

### Numerical Stability Tests
Test with zero input, random input, extreme values. Check for NaN/Inf.
These are medium speed (need to init + predict once).

### Training Regression Tests
Actually train for a few steps, verify loss decreases. These are slow and should
be tagged `:slow` or `:regression`.

### Snapshot Tests
Compare embedding outputs against saved baselines. Catches unintentional changes
to numerical behavior. We already have this.

### Property-Based Tests
Generate random valid inputs, verify invariants hold. Slow but thorough.
We already have this with StreamData.

## Recommended Domain Tags

Based on our codebase structure:

### ExPhil
| Tag | Applies To | Test Count (est.) |
|-----|-----------|-------------------|
| `:backbone` | `test/exphil/networks/*_test.exs` (except policy, actor_critic) | ~400 |
| `:embedding` | `test/exphil/embeddings/*_test.exs` | ~300 |
| `:training` | `test/exphil/training/*_test.exs` | ~500 |
| `:policy` | `test/exphil/networks/policy_test.exs` | ~80 |
| `:config` | `test/exphil/training/config_test.exs` | ~300 |
| `:bridge` | `test/exphil/bridge/*`, `test/exphil_bridge/*` | ~50 |

### Edifice
| Tag | Applies To | Test Count (est.) |
|-----|-----------|-------------------|
| `:recurrent` | `test/edifice/recurrent/*_test.exs` | ~100 |
| `:ssm` | `test/edifice/ssm/*_test.exs` | ~100 |
| `:attention` | `test/edifice/attention/*_test.exs` | ~150 |
| `:vision` | `test/edifice/vision/*_test.exs` | ~80 |
| `:generative` | `test/edifice/generative/*_test.exs` | ~80 |

## CI Test Strategy

### Ideal Pipeline
```
PR opened:
  Job 1 (fast, ~30s): mix test --stale  (or test.smoke)
  Job 2 (medium, ~2m): mix test          (default, no slow)
  Job 3 (slow, ~5m):  mix test --include slow
  Job 4 (GPU, separate runner): mix test --only gpu

Merge to main:
  Full suite: mix test --include slow --include integration
```

### Partitioning for CI
With `--partitions`, split the slow suite across N workers:
```yaml
strategy:
  matrix:
    partition: [1, 2, 3, 4]
env:
  MIX_TEST_PARTITION: ${{ matrix.partition }}
steps:
  - run: mix test --partitions 4 --include slow
```

## Test Naming and Organization Conventions

### Current State
- 121 ExPhil test files, mostly following `lib/` mirror structure
- 251 Edifice test files, organized by architecture family
- Integration tests in `test/exphil/integration/`
- Property tests in `test/exphil/property_tests/`
- Benchmarks in `test/exphil/benchmarks/`

### What's Good
- File naming mirrors module paths (easy to find)
- Integration, property, benchmark tests are separate directories
- Tags for slow/integration/gpu are consistent

### What's Missing
- No domain tags (`:backbone`, `:embedding`, etc.)
- No `--stale` usage documented or aliased
- No "I changed X, run Y" mapping
- No smoke test subset
- CLAUDE.md doesn't instruct against running full suite
- Edifice has almost no tags (just `:slow` and timeouts)

## Script Testing

Our `scripts/` directory has 49 scripts. Most are standalone and don't have
corresponding test files. For scripts that orchestrate training/benchmarking,
a `--quick` or `--dry-run` flag would let us validate they parse args and
set up correctly without actually running the full workload.

## Sources

- [mix test docs (v1.19)](https://hexdocs.pm/mix/Mix.Tasks.Test.html)
- [ExUnit.Case docs](https://hexdocs.pm/ex_unit/ExUnit.Case.html)
- [mix_test_interactive](https://github.com/randycoulman/mix_test_interactive)
- [mix test.watch](https://github.com/lpil/mix-test.watch)
- [junit_formatter](https://github.com/victorolinasc/junit-formatter)
- [Faster test execution in Elixir](https://bartoszgorka.com/faster-test-execution-in-elixir)
- [ExUnit partitioning PR](https://github.com/elixir-lang/elixir/pull/9422)
- [TIL: mix test --stale](https://yiming.dev/blog/2018/07/10/til-mix-test-stale/)
