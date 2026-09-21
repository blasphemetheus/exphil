defmodule ExPhil.Bridge.SimBatch.Core do
  @moduledoc """
  In-process sim batch over the C API NIF (`ExPhil.Bridge.SimNif`), with the
  same surface as `ExPhil.Bridge.SimPort` so `ExPhil.Sim.Env` can dispatch to
  either: `start/1`, `frames/1`, `step/2`, `reinit/2`, `reset/2`, `save/3`,
  `restore/4`, `observe/1`, `stop/1`. No Python, no pipes: a step is one
  NIF call with raw rows both ways; frames are decoded by `SimRows` from
  `priv/sim/layouts.json` (the same descriptors the worker sends), so a
  `GameState` from here is bit-identical to one from the Port path.

  Savestates: `save/3` returns the blob; with `keep: true` it is also kept
  in this struct's cache under a `state_id`, and `restore/4` accepts
  `{:id, state_id}` — the SimPort contract, minus the transfer cost.

  This is a plain struct, updated functionally; keep it in one process.
  """

  alias ExPhil.Bridge.{ControllerState, SimRows, SimState}

  defstruct [:ref, :n, :layouts, :frames, :terminals, :own_port, :players, :stage, :lib, :data_root, cache: %{}, next_id: 0]

  @characters %{
    "mario" => 0, "fox" => 1, "falcon" => 2, "captainfalcon" => 2, "donkeykong" => 3, "dk" => 3,
    "kirby" => 4, "bowser" => 5, "link" => 6, "sheik" => 7, "ness" => 8, "peach" => 9,
    "iceclimbers" => 10, "popo" => 10, "nana" => 11, "pikachu" => 12, "samus" => 13, "yoshi" => 14,
    "jigglypuff" => 15, "mewtwo" => 16, "luigi" => 17, "marth" => 18, "zelda" => 19,
    "younglink" => 20, "drmario" => 21, "doc" => 21, "falco" => 22, "pichu" => 23,
    "gameandwatch" => 24, "gnw" => 24, "ganondorf" => 25, "ganon" => 25, "roy" => 26
  }
  @stages %{"fountain_of_dreams" => 2, "pokemon_stadium" => 3, "yoshis_story" => 8, "dream_land_n64" => 28, "battlefield" => 31, "final_destination" => 32}

  # MslButton bits (api.h)
  @bits %{"A" => 0x0100, "B" => 0x0200, "X" => 0x0400, "Y" => 0x0800, "Z" => 0x0010, "L" => 0x0040, "R" => 0x0020, "D_UP" => 0x0008}

  @doc """
  Options: `:batch_size` (1), `:stage`, `:players`, `:seed`, `:stocks`,
  `:max_frame`, `:own_port` (1), `:root` (EXPHIL_SIM_ROOT or ~/git/msl-main).
  """
  def start(opts \\ []) do
    root = Keyword.get(opts, :root) || System.get_env("EXPHIL_SIM_ROOT") || Path.expand("~/git/msl-main")
    lib = Path.join(root, "build/melee_core/python/libmelee_core.so")
    n = Keyword.get(opts, :batch_size, 1)

    with true <- File.exists?(lib) || {:error, {:sim_library_missing, lib}},
         {:ok, ref} <- ExPhil.Bridge.SimNif.open(lib, Path.join(root, "data"), n) do
      batch = %__MODULE__{ref: ref, n: n, layouts: layouts!(), own_port: Keyword.get(opts, :own_port, 1), lib: lib, data_root: Path.join(root, "data")}
      reinit(batch, Map.new(Keyword.take(opts, [:stage, :players, :seed, :stocks, :max_frame, :batch_size])))
    else
      {:error, _} = e -> e
      other -> {:error, other}
    end
  end

  @doc "Current frames (one GameState per env)."
  def frames(%__MODULE__{frames: f}), do: {:ok, f}

  @doc "Re-configure every env (stage, players, seed…) and reset. `batch_size` must match the batch."
  def reinit(%__MODULE__{} = b, req) when is_map(req) do
    req = Map.new(req, fn {k, v} -> {to_string(k), v} end)

    b =
      if req["batch_size"] && req["batch_size"] != b.n do
        # a different batch size = a new C batch (the savestate cache carries over)
        {:ok, ref} = ExPhil.Bridge.SimNif.open(b.lib, b.data_root, req["batch_size"])
        %{b | ref: ref, n: req["batch_size"]}
      else
        b
      end

    if false,
      do: raise(ArgumentError, "SimBatch batch_size is fixed at #{b.n} (asked #{req["batch_size"]}); start a new batch")

    {:ok, default_bin} = ExPhil.Bridge.SimNif.match_config_default(b.ref)
    default = SimRows.decode(b.layouts["match_config"], default_bin)
    players = req["players"] || [%{"character" => "fox"}, %{"character" => "fox", "costume" => 1}]
    seed0 = req["seed"] || 0

    configs =
      for i <- 0..(b.n - 1), into: <<>> do
        cfg =
          default
          |> Map.put("stage", stage_id(req["stage"] || "final_destination"))
          |> Map.put("random_seed", seed0 + i)
          |> Map.put("num_players", length(players))
          |> Map.put("stocks", req["stocks"] || default["stocks"])
          |> Map.put("max_frame", req["max_frame"] || default["max_frame"])
          |> Map.put("players", Enum.map(players, &player_config(&1, default)) ++ List.duplicate(Enum.at(default["players"], 0), 4 - length(players)))

        SimRows.encode(b.layouts["match_config"], cfg)
      end

    with {:ok, obs} <- ExPhil.Bridge.SimNif.reset(b.ref, configs, <<>>) do
      frames = decode_frames(b, obs)
      {:ok, frames, %{b | frames: frames, terminals: nil, players: players, stage: req["stage"]}}
    end
  end

  @doc "Advance one frame. `controllers`: per env, per player `ControllerState` | sim float row | nil."
  def step(%__MODULE__{} = b, controllers \\ nil) do
    inputs =
      case controllers do
        nil -> :binary.copy(<<0::size(32 * 8)>>, b.n)
        list -> for per_env <- list, into: <<>>, do: encode_input(per_env)
      end

    with {:ok, {obs, term}} <- ExPhil.Bridge.SimNif.step(b.ref, inputs) do
      frames = decode_frames(b, obs)
      terminals = SimRows.decode_rows(b.layouts["terminal"], term, b.n)
      {:ok, frames, terminals, %{b | frames: frames, terminals: terminals}}
    end
  end

  @doc "Re-observe every env without stepping."
  def observe(%__MODULE__{} = b) do
    with {:ok, {obs, term}} <- ExPhil.Bridge.SimNif.observe(b.ref) do
      frames = decode_frames(b, obs)
      terminals = SimRows.decode_rows(b.layouts["terminal"], term, b.n)
      {:ok, frames, terminals, %{b | frames: frames, terminals: terminals}}
    end
  end

  @doc "Reset all envs (or `env_ids`) with the current config."
  def reset(%__MODULE__{} = b, env_ids \\ nil) do
    req = %{"stage" => b.stage || "final_destination", "players" => b.players}
    _ = env_ids
    reinit(b, req)
  end

  @doc "Serialize env `i`. `keep: true` also caches it: returns `{:ok, blob, state_id, batch}`."
  def save(%__MODULE__{} = b, env \\ 0, opts \\ []) do
    with {:ok, blob} <- ExPhil.Bridge.SimNif.save(b.ref, env) do
      if Keyword.get(opts, :keep, false) do
        id = b.next_id + 1
        {:ok, blob, id, %{b | cache: Map.put(b.cache, id, blob), next_id: id}}
      else
        {:ok, blob, b}
      end
    end
  end

  @doc "Cache a blob; returns `{:ok, state_id, batch}`."
  def upload(%__MODULE__{} = b, blob) when is_binary(blob) do
    id = b.next_id + 1
    {:ok, id, %{b | cache: Map.put(b.cache, id, blob), next_id: id}}
  end

  @doc "Restore env `i` from a blob or `{:id, state_id}`. `frames: false` skips the re-observe."
  def restore(%__MODULE__{} = b, env, state, opts \\ []) do
    blob =
      case state do
        {:id, id} -> Map.fetch!(b.cache, id)
        bin when is_binary(bin) -> bin
      end

    with {:ok, :ok} <- ExPhil.Bridge.SimNif.restore(b.ref, env, blob) do
      if Keyword.get(opts, :frames, true), do: observe(b) |> then(fn {:ok, f, _t, b2} -> {:ok, f, b2} end), else: {:ok, b}
    end
  end

  def stop(%__MODULE__{}), do: :ok

  # ---------------------------------------------------------------------------

  defp decode_frames(b, obs), do: b.layouts["gamestate"] |> SimRows.decode_rows(obs, b.n) |> Enum.map(&SimState.to_game_state(&1, own_port: b.own_port))

  defp layouts!() do
    path = Path.join(:code.priv_dir(:exphil) |> to_string(), "sim/layouts.json")
    path |> File.read!() |> Jason.decode!()
  end

  defp stage_id(s) when is_integer(s), do: s
  defp stage_id(s) when is_binary(s), do: Map.fetch!(@stages, String.downcase(s))
  defp stage_id(s) when is_atom(s), do: stage_id(Atom.to_string(s))

  defp character_id(c) when is_integer(c), do: c
  defp character_id(c) when is_atom(c), do: character_id(Atom.to_string(c))
  defp character_id(c) when is_binary(c), do: Map.fetch!(@characters, c |> String.downcase() |> String.replace(~r/[^a-z0-9]/, ""))

  defp player_config(p, default) do
    p = Map.new(p, fn {k, v} -> {to_string(k), v} end)
    base = Enum.at(default["players"], 0) || %{}

    base
    |> Map.put("character", character_id(p["character"]))
    |> Map.put("costume", p["costume"] || 0)
    |> Map.put("start_percent", p["start_percent"] || 0)
    |> Map.put("facing", p["facing"] || 0)
    |> Map.put("team", p["team_id"] || p["team"] || -1)
    |> Map.put("controller_port", p["controller_port"] || -1)
    |> Map.put("handicap", p["handicap"] || 9)
  end

  # ControllerState / float row -> raw MslInput player (8 bytes), like the sim's
  # msl_python_controller_inputs: axis = rint(x*160 - 80), l = rint(shoulder*140), r = 0.
  defp encode_input(per_env) do
    players = Enum.map(per_env, &raw_player/1) ++ List.duplicate(<<0::64>>, 4 - length(per_env))
    IO.iodata_to_binary(Enum.take(players, 4))
  end

  defp raw_player(nil), do: <<0::16, 0::8, 0::8, 0::8, 0::8, 0, 0>>
  defp raw_player(%ControllerState{} = c), do: raw_player(SimState.controller_to_row(c))

  defp raw_player(%{} = row) do
    bt = Map.get(row, :buttons) || Map.get(row, "buttons") || %{}
    bits = Enum.reduce(@bits, 0, fn {name, bit}, acc -> if truthy(Map.get(bt, String.to_atom(name)) || Map.get(bt, name)), do: Bitwise.bor(acc, bit), else: acc end)
    l = Map.get(row, :shoulder) || Map.get(row, "shoulder") || 0.0
    <<bits::unsigned-16-little, axis(row, :main_stick_x)::signed-8, axis(row, :main_stick_y)::signed-8, axis(row, :c_stick_x)::signed-8, axis(row, :c_stick_y)::signed-8, trigger(l)::unsigned-8, 0::8>>
  end

  defp axis(row, key) do
    v = Map.get(row, key) || Map.get(row, Atom.to_string(key)) || 0.5
    round(v * 160.0 - 80.0) |> max(-80) |> min(80)
  end

  defp trigger(v), do: round(v * 140.0) |> max(0) |> min(140)

  defp truthy(v) when is_boolean(v), do: v
  defp truthy(v) when is_integer(v), do: v != 0
  defp truthy(_), do: false
end
