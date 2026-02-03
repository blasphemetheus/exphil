# Self-Play with Dolphin

This guide covers running self-play RL training with real Dolphin/Slippi games instead of mock environments.

## Prerequisites

### 1. Slippi/Dolphin Setup

Ensure you have Slippi Dolphin installed:

```bash
# The Slippi Launcher typically installs Dolphin to:
~/.config/Slippi Launcher/netplay/

# Or on some systems:
~/.local/share/Slippi Launcher/netplay/
```

### 2. Melee ISO

You need a Melee 1.02 NTSC ISO. The path will be used in configuration.

### 3. Python Environment

The bridge requires libmelee:

```bash
# From exphil directory
source .venv/bin/activate
pip install melee
```

## Quick Start

### Using LeagueTrainer

```elixir
# Start IEx
iex -S mix

# Create trainer with Dolphin
{:ok, trainer} = ExPhil.Training.SelfPlay.LeagueTrainer.new(
  mode: :simple_mix,
  game_type: :dolphin,
  dolphin_config: %{
    dolphin_path: System.get_env("HOME") <> "/.config/Slippi Launcher/netplay",
    iso_path: System.get_env("HOME") <> "/Games/melee.iso",
    character: "mewtwo",
    stage: "final_destination"
  },
  rollout_length: 2048,
  num_parallel_games: 1  # Start with 1 for testing
)

# Run training
ExPhil.Training.SelfPlay.LeagueTrainer.train(trainer, total_timesteps: 10_000)
```

### Using SelfPlayEnv Directly

```elixir
# Load a pretrained policy
{:ok, {model, params}} = ExPhil.Training.Imitation.load_checkpoint("checkpoints/imitation.axon")

# Create environment
{:ok, env} = ExPhil.Training.SelfPlay.SelfPlayEnv.new(
  p1_policy: {model, params},
  p2_policy: :cpu,  # or another {model, params}
  game_type: :dolphin,
  dolphin_config: %{
    dolphin_path: "~/.config/Slippi Launcher/netplay",
    iso_path: "~/melee.iso",
    character: "mewtwo",
    stage: "final_destination"
  }
)

# Collect experience
{:ok, env, experiences} = ExPhil.Training.SelfPlay.SelfPlayEnv.collect_steps(env, 1000)

# Don't forget to shutdown when done
ExPhil.Training.SelfPlay.SelfPlayEnv.shutdown(env)
```

## Manual Testing Checklist

### Test 1: Basic MeleePort Connection

Verify the Python bridge can connect to Dolphin:

```elixir
alias ExPhil.Bridge.MeleePort

{:ok, port} = MeleePort.start_link()

:ok = MeleePort.init_console(port, %{
  dolphin_path: "~/.config/Slippi Launcher/netplay",
  iso_path: "~/melee.iso",
  character: "mewtwo",
  stage: "final_destination"
})

# Should see Dolphin launch and navigate to game
# Wait for game to start, then:
{:ok, state} = MeleePort.step(port)

IO.inspect(state.frame)
IO.inspect(state.players)

# Cleanup
MeleePort.stop(port)
```

**Expected**: Dolphin launches, navigates through menus, starts a game. `state` contains player positions, damage, etc.

### Test 2: Controller Input

Verify the agent can control the character:

```elixir
alias ExPhil.Bridge.MeleePort

{:ok, port} = MeleePort.start_link()
:ok = MeleePort.init_console(port, %{...})

# Wait for game to start
for _ <- 1..300 do
  MeleePort.step(port, auto_menu: true)
  Process.sleep(16)
end

# Send some inputs - move right and jump
for _ <- 1..60 do
  MeleePort.send_controller(port, %{
    main_stick: %{x: 1.0, y: 0.5},  # Right
    buttons: %{y: true}              # Jump
  })
  {:ok, state} = MeleePort.step(port)
  IO.puts("Frame #{state.frame}: x=#{state.players[1].x}")
end

MeleePort.stop(port)
```

**Expected**: Character moves right and jumps. Position X increases.

### Test 3: Episode Reset

Verify games can restart after ending:

```elixir
alias ExPhil.Bridge.MeleePort

{:ok, port} = MeleePort.start_link()
:ok = MeleePort.init_console(port, %{...})

# Wait for game, then manually end it (SD or get KO'd)
IO.puts("Game starting... Please end the game to test reset")

# Poll until postgame
result = Stream.repeatedly(fn ->
  MeleePort.step(port, auto_menu: true)
end)
|> Enum.find(fn
  {:postgame, _} -> true
  _ -> false
end)

IO.puts("Got postgame, testing reset...")

# The auto_menu should navigate back to game
result = Stream.repeatedly(fn ->
  MeleePort.step(port, auto_menu: true)
end)
|> Stream.take(1800)  # 30 seconds max
|> Enum.find(fn
  {:ok, state} -> state.menu_state == 2  # IN_GAME
  _ -> false
end)

case result do
  {:ok, state} -> IO.puts("Reset successful! Frame: #{state.frame}")
  nil -> IO.puts("Reset timed out")
end

MeleePort.stop(port)
```

**Expected**: After game ends, menus are navigated automatically and a new game starts.

### Test 4: SelfPlayEnv Full Loop

```elixir
alias ExPhil.Training.SelfPlay.SelfPlayEnv

# Create mock policy (random actions)
model = Axon.input("state", shape: {nil, 1991})
|> Axon.dense(64, activation: :relu)
|> then(fn x ->
  buttons = Axon.dense(x, 8, name: "buttons")
  main_x = Axon.dense(x, 17, name: "main_x")
  main_y = Axon.dense(x, 17, name: "main_y")
  c_x = Axon.dense(x, 17, name: "c_x")
  c_y = Axon.dense(x, 17, name: "c_y")
  shoulder = Axon.dense(x, 5, name: "shoulder")
  value = Axon.dense(x, 1, name: "value")
  Axon.container({{buttons, main_x, main_y, c_x, c_y, shoulder}, value})
end)

{init_fn, _} = Axon.build(model)
params = init_fn.(Nx.template({1, 1991}, :f32), %{})

{:ok, env} = SelfPlayEnv.new(
  p1_policy: {model, params},
  p2_policy: :cpu,
  game_type: :dolphin,
  dolphin_config: %{
    dolphin_path: "~/.config/Slippi Launcher/netplay",
    iso_path: "~/melee.iso",
    character: "mewtwo",
    stage: "final_destination"
  }
)

# Collect 5 seconds of gameplay
IO.puts("Collecting 300 frames (5 seconds)...")
{:ok, env, experiences} = SelfPlayEnv.collect_steps(env, 300)

IO.puts("Collected #{length(experiences)} experiences")
IO.puts("First experience keys: #{inspect(Map.keys(hd(experiences)))}")

# Check rewards
rewards = Enum.map(experiences, & &1.reward)
IO.puts("Total reward: #{Enum.sum(rewards)}")

SelfPlayEnv.shutdown(env)
```

**Expected**: Collects 300 experiences with proper structure.

## Running Automated Integration Tests

```bash
# Set environment variables
export DOLPHIN_PATH="$HOME/.config/Slippi Launcher/netplay"
export MELEE_ISO="$HOME/Games/melee.iso"

# Optional: custom character/stage
export TEST_CHARACTER="mewtwo"
export TEST_STAGE="final_destination"

# Run dolphin-tagged tests
mix test --include dolphin

# Or run specific test file
mix test test/exphil/integration/dolphin_self_play_test.exs --include dolphin
```

## Configuration Reference

### dolphin_config Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `dolphin_path` | string | Yes | - | Path to Slippi Dolphin folder |
| `iso_path` | string | Yes | - | Path to Melee 1.02 ISO |
| `character` | string | No | "fox" | Character to select |
| `stage` | string | No | "final_destination" | Stage to select |
| `controller_port` | integer | No | 1 | P1 controller port |
| `opponent_port` | integer | No | 2 | P2 controller port |

### Character Names

Use lowercase with underscores:
- `mewtwo`, `ganondorf`, `link`, `zelda`, `game_and_watch`
- `fox`, `falco`, `marth`, `sheik`, `peach`, `captain_falcon`
- See libmelee documentation for full list

### Stage Names

Use lowercase with underscores:
- `final_destination`, `battlefield`, `yoshis_story`
- `fountain_of_dreams`, `pokemon_stadium`, `dreamland`

## Troubleshooting

### Dolphin doesn't launch

1. Check `dolphin_path` is correct
2. Ensure Dolphin has execute permissions
3. Check logs: `tail -f /tmp/melee_bridge.log`

### Connection timeout

1. Dolphin may need more time to initialize
2. Try increasing timeouts in test
3. Check if Dolphin is already running (close existing instances)

### Menu navigation fails

1. Ensure `auto_menu: true` is passed to `step/2`
2. Check character/stage names are valid
3. The menu helper may need multiple frames to navigate

### Reset times out

1. Increase timeout (default 30 seconds / 1800 frames)
2. Check if game is stuck in a menu state
3. Logs: `tail -f /tmp/melee_bridge.log`

### Performance issues

- Each Dolphin instance uses ~1 CPU core at 60fps
- Memory: ~200-300 MB per instance
- Start with `num_parallel_games: 1` and increase gradually

## Architecture Notes

```
┌─────────────────────────────────────────────────────────────────┐
│                    LeagueTrainer                                 │
│  - Manages training loop                                        │
│  - Handles checkpointing                                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SelfPlayEnv                                  │
│  - Wraps game environment                                       │
│  - Handles policy inference                                     │
│  - Collects experience                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MeleePort                                   │
│  - GenServer managing Python bridge                             │
│  - JSON protocol over stdio                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   melee_bridge.py                                │
│  - libmelee integration                                         │
│  - Menu navigation (menu_helper_simple)                         │
│  - Game state serialization                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Slippi Dolphin                                 │
│  - Emulator                                                     │
│  - Melee 1.02 ROM                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Reset Flow

When an episode ends:

1. `is_episode_done` returns `true` (stocks depleted or max frames)
2. `reset/1` is called on `SelfPlayEnv` or `GameRunner`
3. `reset_dolphin_game/1` polls `MeleePort.step(auto_menu: true)`
4. Python bridge's `menu_helper_simple` navigates postgame → char select → stage → game
5. When `step` returns `{:ok, game_state}`, reset is complete
6. New episode begins

## Known Limitations

1. **No parallel Dolphin yet**: `num_parallel_games > 1` with Dolphin is untested
2. **Frame timing variance**: Real Dolphin has hitches vs. perfect mock timing
3. **No headless mode**: Dolphin requires a display (or virtual framebuffer)
4. **Memory overhead**: ~200-300 MB per Dolphin instance

## Next Steps

After verifying manual tests work:

1. Run longer training sessions (10k+ timesteps)
2. Test with trained policy (not random)
3. Test episode resets across multiple games
4. Monitor reward curves
5. Compare to mock training results
