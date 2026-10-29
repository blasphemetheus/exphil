#!/usr/bin/env elixir
# ExPhil policy (port 1) vs vladfi1's slippi-ai agent (port 2) in ONE Dolphin.
#
# Host = our native Elixir harness (MeleePort/libmelee_ex drives Dolphin and
# both pipe controllers; AsyncRunner runs the port-1 policy exactly as
# play_dolphin_async.exs does). The slippi-ai agent rides the runner's
# Elixir-driven opponent hook (`dummy: {Mod, arg}`, the rich `step_game/3`
# contract PolicyOpponent uses): each in-game frame's GameState is forwarded
# over a length-prefixed JSON port to priv/python/slippi_ai_worker.py, which
# runs INSIDE the slippi-ai venv and returns the controller their unmodified
# `eval_lib.Agent.step` decoded. See docs/guides/DOLPHIN.md
# "slippi-ai head-to-head".
#
# Usage (from the repo root, inside `devenv shell`):
#
#   mix run scripts/play_vs_slippi_ai.exs \
#     --policy checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin \
#     --character fox --stage final_destination \
#     --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
#     --iso "$HOME/isos/melee.iso" \
#     --num-games 5 \
#     --replay-dir eval_runs/<session>/ \
#     --results eval_runs/<session>/results.json
#
# Script-specific flags (all optional):
#   --slippi-ai-model PATH    slippi-ai state pickle (default ~/git/slippi-ai/models/medium-v2)
#   --slippi-ai-python PATH   python inside the slippi-ai venv (default ~/git/slippi-ai/.venv/bin/python)
#   --slippi-ai-character C   their character (default fox; must be in the model's allowed set)
#   --slippi-ai-temperature T their sampling temperature (default 1.0, their harness default)
#   --slippi-ai-delay N       OVERRIDE their policy delay (default: the model's own, 21 for
#                             medium-v2). Untrained regime for their network — symmetry
#                             experiments only; see the DOLPHIN.md section.
#   --slippi-ai-seed N        their JAX/TF sampling seed (default 0; only matters for JAX agents)
#   --num-games N             finalized games to play before stopping (default 1)
#   --results PATH            results JSON (default <replay-dir>/results.json)
#
# Every play_dolphin_async.exs knob applies to OUR side unchanged
# (--reaction-delay / --frame-delay, --deterministic, --temperature,
# --policy-server, --headless, --local-delay, ...). The Dolphin frame delay
# the rung resolves to is also handed to the slippi-ai agent as its
# console_delay, so their total input->action delay stays at their policy's
# delay (21) regardless of our rung.

if "--policy-server" in System.argv() do
  System.put_env("EXLA_CPU_ONLY", "1")
end

require Logger

alias ExPhil.CLI
alias ExPhil.Bridge.MeleePort
alias ExPhil.Bridge.AsyncRunner
alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

flag_groups = [:verbosity, :checkpoint, :replay, :dolphin, :common]

opts =
  CLI.parse_args(System.argv(),
    flags: flag_groups,
    extra: [
      slippi_ai_model: :string,
      slippi_ai_python: :string,
      slippi_ai_character: :string,
      slippi_ai_temperature: :float,
      slippi_ai_delay: :integer,
      slippi_ai_seed: :integer,
      num_games: :integer,
      results: :string
    ],
    defaults: [character: "fox"]
  )

CLI.setup_verbosity(opts)

opts =
  Keyword.update(opts, :character, :fox, fn
    c when is_binary(c) -> String.to_atom(c)
    c -> c
  end)

opts =
  Keyword.update(opts, :stage, :final_destination, fn
    s when is_binary(s) -> String.to_atom(s)
    s -> s
  end)

CLI.maybe_show_help(opts, "play_vs_slippi_ai.exs", flag_groups, fn ->
  IO.puts("""

  ExPhil policy (port 1) vs slippi-ai agent (port 2), one Dolphin.

    --slippi-ai-model PATH      default ~/git/slippi-ai/models/medium-v2
    --slippi-ai-python PATH     default ~/git/slippi-ai/.venv/bin/python
    --slippi-ai-character C     default fox
    --slippi-ai-temperature T   default 1.0
    --slippi-ai-delay N         override their policy delay (symmetry experiments only)
    --num-games N               default 1
    --results PATH              default <replay-dir>/results.json
  """)
end)

CLI.require_options!(opts, [:policy, :dolphin, :iso])

slippi_ai_root = Path.expand("~/git/slippi-ai")

sa = %{
  model: Path.expand(opts[:slippi_ai_model] || Path.join(slippi_ai_root, "models/medium-v2")),
  python: Path.expand(opts[:slippi_ai_python] || Path.join(slippi_ai_root, ".venv/bin/python")),
  character: String.downcase(opts[:slippi_ai_character] || "fox"),
  temperature: opts[:slippi_ai_temperature] || 1.0,
  delay_override: opts[:slippi_ai_delay],
  root: slippi_ai_root,
  worker_script: Path.join([File.cwd!(), "priv", "python", "slippi_ai_worker.py"])
}

for {what, path} <- [{"slippi-ai model", sa.model}, {"slippi-ai python", sa.python}, {"worker script", sa.worker_script}] do
  unless File.exists?(path) do
    Output.error("#{what} not found: #{path}")
    System.halt(2)
  end
end

num_games = opts[:num_games] || 1

# INVARIANTS item 12: one delay knob, resolved through HarnessRung exactly as
# play_dolphin_async.exs does; the LatencyProbe measures it at game start.
{:ok, %{config: rung_cfg}} = ExPhil.Training.Checkpoint.load_policy(opts[:policy])

rung =
  case ExPhil.Eval.HarnessRung.resolve(:async_runner, opts, rung_cfg) do
    {:ok, r} ->
      r

    {:error, msg} ->
      Output.error(msg)
      System.halt(1)
  end

if rung.source == :frame_delay_alias,
  do: Output.warning("--frame-delay is deprecated; this run == --reaction-delay #{rung.reaction_delay}")

opts = Keyword.put(opts, :frame_delay, rung.knob)

Output.banner("ExPhil vs slippi-ai (one Dolphin)")

# GOTCHA #84: replays are the canonical evidence; never let them die in the
# temp User dir.
opts =
  if is_nil(opts[:replay_dir]) do
    default_dir = "eval_runs/#{Date.utc_today() |> Calendar.strftime("%m%d")}_vs_slippi_ai"
    File.mkdir_p!(default_dir)
    Output.puts("No --replay-dir given — replays will be kept in #{default_dir}/")
    Keyword.put(opts, :replay_dir, default_dir)
  else
    File.mkdir_p!(opts[:replay_dir])
    opts
  end

results_path = opts[:results] || Path.join(opts[:replay_dir], "results.json")

temperature =
  cond do
    opts[:buttons_temperature] || opts[:main_temperature] ||
      opts[:c_temperature] || opts[:shoulder_temperature] ->
      %{
        buttons: opts[:buttons_temperature] || opts[:temperature] || 1.0,
        main: opts[:main_temperature] || opts[:temperature] || 1.0,
        c: opts[:c_temperature] || opts[:temperature] || 1.0,
        shoulder: opts[:shoulder_temperature] || opts[:temperature] || 1.0
      }

    true ->
      opts[:temperature] || 1.0
  end

Output.config([
  {"ExPhil policy (port #{opts[:port]})", opts[:policy]},
  {"slippi-ai model (port #{opts[:opponent_port]})", sa.model},
  {"slippi-ai character", sa.character},
  {"slippi-ai temperature", sa.temperature},
  {"slippi-ai delay override", sa.delay_override || "none (model's own)"},
  {"Dolphin", opts[:dolphin]},
  {"ISO", opts[:iso]},
  {"Character", opts[:character]},
  {"Stage", opts[:stage]},
  {"Reaction delay (ours)",
   "#{rung.reaction_delay} (latency #{rung.expected_latency}; Dolphin frame delay #{rung.knob}; from #{rung.source})"},
  {"Deterministic", opts[:deterministic]},
  {"Temperature", inspect(temperature)},
  {"Games", num_games},
  {"Replay dir", opts[:replay_dir]},
  {"Results", results_path}
])

# ============================================================================
# slippi-ai worker: owns the python Port (one process owns a port; the frame
# loop and the runner are different processes, so route every call here).
# ============================================================================
defmodule SlippiAiWorker do
  use GenServer
  require Logger

  @call_timeout 2_000
  @init_timeout 300_000

  def start_link(cfg), do: GenServer.start_link(__MODULE__, cfg, name: __MODULE__)

  @doc "Forward one in-game GameState; returns the bridge input map (string keys) or nil."
  def step(game_state) do
    GenServer.call(__MODULE__, {:step, game_state}, @call_timeout)
  catch
    :exit, reason ->
      Logger.warning("[SlippiAiWorker] step failed: #{inspect(reason)}")
      nil
  end

  def info, do: GenServer.call(__MODULE__, :info)
  def stop, do: GenServer.call(__MODULE__, :stop, 15_000)

  @impl true
  def init(cfg) do
    env =
      [
        {~c"PYTHONUNBUFFERED", ~c"1"},
        {~c"CUDA_VISIBLE_DEVICES", ~c""},
        {~c"TF_CPP_MIN_LOG_LEVEL", ~c"2"}
      ] ++
        for var <- ["NIX_PYTHONPATH", "PYTHONPATH"], val = System.get_env(var), val != nil do
          {String.to_charlist(var), String.to_charlist(val)}
        end

    port =
      Port.open({:spawn_executable, cfg.python}, [
        :binary,
        :exit_status,
        {:packet, 4},
        {:args, [cfg.worker_script]},
        {:cd, cfg.root},
        {:env, env}
      ])

    init_req = %{
      cmd: "init",
      model_path: cfg.model,
      port: cfg.port,
      opponent_port: cfg.opponent_port,
      console_delay: cfg.console_delay,
      character: cfg.character,
      sample_temperature: cfg.temperature,
      policy_delay_override: cfg.delay_override,
      seed: cfg.seed
    }

    Port.command(port, Jason.encode!(init_req))

    case recv(port, @init_timeout) do
      {:ok, %{"ok" => true} = resp} ->
        {:ok, %{port: port, info: resp, steps: 0, step_ms_sum: 0.0}}

      other ->
        {:stop, {:slippi_ai_init_failed, other}}
    end
  end

  @impl true
  def handle_call({:step, game_state}, _from, state) do
    Port.command(state.port, Jason.encode!(%{cmd: "step", game_state: plain_game_state(game_state)}))

    case recv(state.port, @call_timeout) do
      {:ok, %{"ok" => true, "input" => input} = resp} ->
        ms = Map.get(resp, "step_ms", 0.0)
        {:reply, input, %{state | steps: state.steps + 1, step_ms_sum: state.step_ms_sum + ms}}

      {:ok, %{"error" => err}} ->
        Logger.warning("[SlippiAiWorker] step error: #{err}")
        {:reply, nil, state}

      {:error, reason} ->
        Logger.error("[SlippiAiWorker] worker unreachable: #{inspect(reason)}")
        {:reply, nil, state}
    end
  end

  def handle_call(:info, _from, state) do
    mean = if state.steps > 0, do: state.step_ms_sum / state.steps, else: 0.0
    {:reply, Map.merge(state.info, %{"steps" => state.steps, "mean_step_ms" => mean}), state}
  end

  def handle_call(:stop, _from, state) do
    Port.command(state.port, Jason.encode!(%{cmd: "stop"}))
    _ = recv(state.port, 10_000)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("[SlippiAiWorker] python worker exited with status #{status}")
    {:stop, {:slippi_ai_worker_exited, status}, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  defp recv(port, timeout) do
    receive do
      {^port, {:data, payload}} ->
        case Jason.decode(payload) do
          {:ok, map} -> {:ok, map}
          {:error, _} -> {:error, {:bad_json, String.slice(payload, 0, 200)}}
        end

      {^port, {:exit_status, status}} ->
        {:error, {:worker_exited, status}}
    after
      timeout -> {:error, :timeout}
    end
  end

  # ExPhil.Bridge.GameState -> JSON-able map (string port keys; the worker
  # rebuilds a melee.GameState from exactly these fields).
  defp plain_game_state(gs) do
    %{
      frame: gs.frame,
      stage: gs.stage,
      menu_state: gs.menu_state,
      distance: gs.distance,
      fod_platform_left: gs.fod_platform_left,
      fod_platform_right: gs.fod_platform_right,
      players: Map.new(gs.players || %{}, fn {p, v} -> {Integer.to_string(p), plain(v)} end),
      projectiles: Enum.map(gs.projectiles || [], &plain/1)
    }
  end

  defp plain(%_{} = s), do: s |> Map.from_struct() |> plain()
  defp plain(%{} = m), do: Map.new(m, fn {k, v} -> {k, plain(v)} end)
  defp plain(l) when is_list(l), do: Enum.map(l, &plain/1)
  defp plain(t) when is_tuple(t), do: t |> Tuple.to_list() |> plain()
  defp plain(v), do: v
end

# ============================================================================
# Match book: per-game result tracking in a public ETS table the frame loop
# writes and the script reads at the end.
# ============================================================================
defmodule MatchBook do
  @table :slippi_ai_match_book

  def create(meta) do
    :ets.new(@table, [:named_table, :set, :public])
    :ets.insert(@table, {:meta, meta})
    :ets.insert(@table, {:games, []})
    :ets.insert(@table, {:current, nil})
    :ok
  end

  @doc "Record one in-game frame; finalizes the previous game on a frame regression."
  def observe(gs, exphil_port, slippi_port) do
    snap = snapshot(gs, exphil_port, slippi_port)

    case :ets.lookup(@table, :current) do
      [{:current, %{frame: last} = cur}] when gs.frame < last ->
        finalize(cur)
        :ets.insert(@table, {:current, snap})

      _ ->
        :ets.insert(@table, {:current, snap})
    end

    :ok
  end

  def finalize_current do
    case :ets.lookup(@table, :current) do
      [{:current, %{} = cur}] ->
        finalize(cur)
        :ets.insert(@table, {:current, nil})

      _ ->
        :ok
    end
  end

  def write(path) do
    [{:meta, meta}] = :ets.lookup(@table, :meta)
    [{:games, games}] = :ets.lookup(@table, :games)
    games = Enum.reverse(games)

    summary = %{
      games: length(games),
      exphil_wins: Enum.count(games, &(&1.winner == "exphil")),
      slippi_ai_wins: Enum.count(games, &(&1.winner == "slippi_ai")),
      incomplete: Enum.count(games, &(not &1.complete))
    }

    File.write!(path, Jason.encode!(%{meta: meta, summary: summary, games: games}, pretty: true))
    summary
  end

  defp snapshot(gs, ep, sp) do
    e = gs.players[ep]
    s = gs.players[sp]

    %{
      frame: gs.frame,
      exphil: %{stocks: e && e.stock, percent: e && e.percent},
      slippi_ai: %{stocks: s && s.stock, percent: s && s.percent}
    }
  end

  defp finalize(%{exphil: e, slippi_ai: s, frame: frame}) do
    es = e.stocks || 0
    ss = s.stocks || 0

    {winner, complete} =
      cond do
        es == 0 and ss > 0 -> {"slippi_ai", true}
        ss == 0 and es > 0 -> {"exphil", true}
        es == 0 and ss == 0 -> {"draw", true}
        # Game left early (LRAS / timeout / session stop): rank by stocks,
        # then percent, and say so.
        es > ss -> {"exphil", false}
        ss > es -> {"slippi_ai", false}
        (e.percent || 0.0) < (s.percent || 0.0) -> {"exphil", false}
        (s.percent || 0.0) < (e.percent || 0.0) -> {"slippi_ai", false}
        true -> {"draw", false}
      end

    [{:games, games}] = :ets.lookup(@table, :games)

    record = %{
      index: length(games) + 1,
      winner: winner,
      complete: complete,
      exphil: e,
      slippi_ai: s,
      last_frame: frame,
      duration_s: Float.round(max(frame, 0) / 60.0, 1),
      recorded_at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    :ets.insert(@table, {:games, [record | games]})
    IO.puts("[Match] game #{record.index}: #{winner}#{if complete, do: "", else: " (incomplete)"} — " <>
      "exphil #{es} stocks / #{round(e.percent || 0.0)}%, slippi-ai #{ss} stocks / #{round(s.percent || 0.0)}%, #{record.duration_s}s")
  end
end

# ============================================================================
# The opponent module AsyncRunner drives (rich contract: new/1 + step_game/3).
# ============================================================================
defmodule SlippiAiOpponent do
  alias ExPhil.Bridge.ControllerState

  defstruct [:exphil_port]

  def new(opts), do: %__MODULE__{exphil_port: Keyword.fetch!(opts, :exphil_port)}

  def step_game(game_state, self_port, %__MODULE__{} = s) do
    MatchBook.observe(game_state, s.exphil_port, self_port)

    case SlippiAiWorker.step(game_state) do
      %{"main_stick" => ms, "c_stick" => cs, "buttons" => b} = input ->
        controller = %ControllerState{
          main_stick: %{x: ms["x"] * 1.0, y: ms["y"] * 1.0},
          c_stick: %{x: cs["x"] * 1.0, y: cs["y"] * 1.0},
          l_shoulder: (input["shoulder"] || 0.0) * 1.0,
          r_shoulder: 0.0,
          button_a: b["a"] == true,
          button_b: b["b"] == true,
          button_x: b["x"] == true,
          button_y: b["y"] == true,
          button_z: b["z"] == true,
          button_l: b["l"] == true,
          button_r: b["r"] == true,
          button_d_up: b["d_up"] == true
        }

        {controller, s}

      _ ->
        {nil, s}
    end
  end
end

# ============================================================================
# Step 1: slippi-ai worker (TF import + their warmup, ~3-20 s) BEFORE Dolphin
# ============================================================================
Output.step(1, 5, "Starting slippi-ai worker (#{Path.basename(sa.model)})")

{:ok, _worker} =
  SlippiAiWorker.start_link(%{
    python: sa.python,
    worker_script: sa.worker_script,
    root: sa.root,
    model: sa.model,
    port: opts[:opponent_port],
    opponent_port: opts[:port],
    # Their console_delay = the Dolphin frame delay our rung sets; their
    # internal queue absorbs the rest so total == their policy delay.
    console_delay: rung.knob,
    character: sa.character,
    temperature: sa.temperature,
    delay_override: sa.delay_override,
    seed: opts[:slippi_ai_seed] || 0
  })

worker_info = SlippiAiWorker.info()

Output.success(
  "slippi-ai agent ready: type=#{worker_info["agent_type"]} policy_delay=#{worker_info["policy_delay"]} " <>
    "console_delay=#{worker_info["console_delay"]} queue=#{worker_info["effective_queue_delay"]} " <>
    "(#{Float.round(worker_info["warmup_s"] * 1.0, 1)}s)"
)

MatchBook.create(%{
  exphil_policy: opts[:policy],
  exphil_port: opts[:port],
  exphil_character: opts[:character],
  exphil_reaction_delay: rung.reaction_delay,
  exphil_expected_latency: rung.expected_latency,
  exphil_local_delay: opts[:local_delay] || 0,
  exphil_deterministic: opts[:deterministic] || false,
  exphil_temperature: temperature,
  slippi_ai_model: sa.model,
  slippi_ai_port: opts[:opponent_port],
  slippi_ai_character: sa.character,
  slippi_ai_agent_type: worker_info["agent_type"],
  slippi_ai_policy_delay: worker_info["policy_delay"],
  slippi_ai_console_delay: worker_info["console_delay"],
  slippi_ai_temperature: sa.temperature,
  stage: opts[:stage],
  dolphin_frame_delay: rung.knob,
  replay_dir: opts[:replay_dir],
  started_at: DateTime.utc_now() |> DateTime.to_iso8601()
})

# ============================================================================
# Step 2: our agent
# ============================================================================
Output.step(2, 5, "Loading ExPhil agent")

agent_opts = [
  deterministic: opts[:deterministic],
  temperature: temperature,
  deterministic_buttons: opts[:deterministic_buttons] || false,
  mode_of_n: opts[:mode_of_n],
  critic_path: opts[:critic],
  critic_k: opts[:critic_k],
  press_threshold: opts[:press_threshold],
  release_threshold: opts[:release_threshold],
  jump_debounce: opts[:jump_debounce],
  frame_delay: opts[:frame_delay],
  harness: :async_runner,
  reaction_delay: rung.reaction_delay,
  delay_id: opts[:delay_id_override],
  allow_untrained_delay_id: opts[:delay_id_override] != nil,
  ablate_prev_action: opts[:ablate_prev_action] || false,
  leace_eraser: opts[:leace_eraser],
  steer_vector: opts[:steer_vector],
  steer_alpha: opts[:steer_alpha] || 1.0,
  style_id: opts[:style_id],
  style_tag: opts[:style_tag],
  player_registry: opts[:player_registry],
  uncertainty_log: opts[:uncertainty_log],
  stateful_step: opts[:stateful_step] || false,
  stateful_resync: opts[:stateful_resync],
  stateful_resync_phase: opts[:stateful_resync_phase] || 0,
  stateful_resync_stagger: opts[:stateful_resync_stagger] || false,
  af_convention: if(opts[:live_af], do: :libmelee, else: :parsed)
]

agent =
  if opts[:policy_server] do
    unless Node.alive?() do
      raise """
      --policy-server needs the session VM booted distributed. Launch with:
        ELIXIR_ERL_OPTIONS="-name session_$$@127.0.0.1 -setcookie exphil_policy_local" \\
          mix run scripts/play_vs_slippi_ai.exs --policy-server ...
      """
    end

    Node.set_cookie(:exphil_policy_local)

    unless Node.connect(:"exphil_policy@127.0.0.1") do
      raise "policy server not reachable — start it: mix run scripts/policy_server.exs"
    end

    :global.sync()

    {:ok, agent, %{warmup_ms: ms, cached: cached}} =
      ExPhil.Agents.PolicyServer.checkout(opts[:policy], agent_opts)

    Output.success("Agent checked out from policy server (warmup #{ms}ms, cached=#{cached})")
    agent
  else
    {:ok, agent} = Agent.start_link([policy_path: opts[:policy]] ++ agent_opts)
    agent
  end

config = Agent.get_config(agent)
Output.success("Agent loaded (temporal=#{config.temporal}#{if config.temporal, do: ", #{config.backbone} w#{config.window_size}", else: ""})")

# ============================================================================
# Step 3: Dolphin (port 1 = us, port 2 = external dummy controller = slippi-ai)
# ============================================================================
Output.step(3, 5, "Initializing Dolphin")

{:ok, bridge} = MeleePort.start_link()

:persistent_term.put({:exphil, :warmup_done}, false)

bridge_config = %{
  menu_ready_check: fn -> :persistent_term.get({:exphil, :warmup_done}, true) end,
  dolphin_path: opts[:dolphin],
  iso_path: opts[:iso],
  controller_port: opts[:port],
  opponent_port: opts[:opponent_port],
  character: opts[:character],
  stage: opts[:stage],
  online_delay: opts[:frame_delay],
  gfx_backend: System.get_env("EXPHIL_GFX"),
  # The slippi-ai agent drives port 2 through the external-dummy controller;
  # the menu helper picks its character at the CSS (cpu_level 0 = human port).
  dummy_mode: "external",
  dummy_character: sa.character,
  dummy_cpu_level: 0,
  no_audio: opts[:no_audio] || false,
  headless: opts[:headless] || false,
  emulation_speed: opts[:emulation_speed],
  blocking_input: opts[:blocking_input] || nil,
  console_timeout: opts[:console_timeout],
  replay_dir: opts[:replay_dir],
  slippi_port: opts[:slippi_port],
  memory_card: if(opts[:nametag], do: :folder, else: true),
  nametag: opts[:nametag],
  frozen_stadium: opts[:frozen_stadium],
  # Hold the bot's START press for a few seconds after each game so the
  # game-count monitor can stop the session at the CSS instead of a few
  # frames into an unwanted extra game.
  postgame_delay: opts[:postgame_delay] || 5
}

case MeleePort.init_console(bridge, bridge_config, 60_000) do
  {:ok, info} ->
    Output.success("Dolphin initialized (our controller on port #{info.controller_port}, slippi-ai on port #{opts[:opponent_port]})")

  :ok ->
    Output.success("Dolphin initialized")

  {:error, reason} ->
    Output.error("Failed to initialize Dolphin: #{inspect(reason)}")
    System.halt(1)
end

# ============================================================================
# Step 4: JIT warmup (backgrounded; CSS interlock holds our pick until done)
# ============================================================================
Output.step(4, 5, "JIT warmup (backgrounded)")

Task.start(fn ->
  try do
    case Agent.warmup(agent) do
      {:ok, ms} -> Output.success("JIT warmup complete (#{ms}ms)")
      {:error, reason} -> Output.warning("Warmup failed: #{inspect(reason)}")
    end
  after
    :persistent_term.put({:exphil, :warmup_done}, true)
  end
end)

# ============================================================================
# Step 5: runner
# ============================================================================
Output.step(5, 5, "Starting async runner (#{num_games} game#{if num_games == 1, do: "", else: "s"})")

{:ok, runner} =
  AsyncRunner.start_link(
    agent: agent,
    bridge: bridge,
    player_port: opts[:port],
    auto_menu: not opts[:no_auto_menu],
    on_game_end: if(num_games == 1, do: :stop, else: :restart),
    dummy: {SlippiAiOpponent, [exphil_port: opts[:port]]},
    no_frame_fatal_streak: 600,
    pace_hz: opts[:pace_hz] || if(opts[:headless], do: 60, else: 0),
    lras: opts[:console_timeout] != 0,
    local_delay: opts[:local_delay] || 0,
    expected_latency: rung.expected_latency,
    allow_latency_mismatch: opts[:allow_latency_mismatch] || false
  )

Output.success("Runner started — Ctrl+C to abort (replays finalize only on game end)")

defmodule GameCountMonitor do
  def run(runner, target, interval_ms \\ 500) do
    Process.sleep(interval_ms)
    stats = AsyncRunner.get_stats(runner)

    tick = Process.get(:ticks, 0) + 1
    Process.put(:ticks, tick)

    if rem(tick, 10) == 0 and stats.elapsed_ms > 0 do
      IO.puts(
        "[Stats] #{Float.round(stats.elapsed_ms / 1000, 1)}s | #{Float.round(stats.fps, 1)} fps | " <>
          "inferences #{stats.inferences_run} | games #{stats.games_played}/#{target}"
      )
    end

    if stats.games_played >= target, do: :done, else: run(runner, target, interval_ms)
  end
end

try do
  GameCountMonitor.run(runner, num_games)
  Output.success("#{num_games} game(s) finished")
catch
  :exit, _ -> Output.puts("Exiting...")
end

Output.puts("Cleaning up...")

case AsyncRunner.stop(runner) do
  {:ok, final} ->
    Output.puts("  Runner: #{final.frames} frames, #{final.inferences} inferences, #{final.games} games, " <>
      "#{Map.get(final, :stale_sends, 0)} stale sends")

  _ ->
    :ok
end

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> Output.puts("  (bridge cleanup timed out, Dolphin may still be running)")
end

MatchBook.finalize_current()
summary = MatchBook.write(results_path)

info = SlippiAiWorker.info()
Output.puts("  slippi-ai worker: #{info["steps"]} steps, mean #{Float.round(info["mean_step_ms"] * 1.0, 2)} ms/step")
try do
  SlippiAiWorker.stop()
catch
  :exit, _ -> :ok
end

if opts[:policy_server] do
  ExPhil.Agents.PolicyServer.release(agent)
else
  GenServer.stop(agent)
end

Output.divider()
Output.section("Results")
Output.puts("  exphil #{summary.exphil_wins} – #{summary.slippi_ai_wins} slippi-ai (#{summary.games} games, #{summary.incomplete} incomplete)")
Output.puts("  JSON:    #{results_path}")
Output.puts("  Replays: #{opts[:replay_dir]}")
