defmodule ExPhil.Sim.Trace do
  @moduledoc """
  Writer for melee-sim-light's `MSLTRACE1` viewer format
  (`tools/viewer/TRACE_FORMAT.md`), so any rollout we produce can be
  PLAYED in the sim's HTML viewer with real fighter models and hitboxes:
  open the viewer (`make viewer` in the sim checkout), choose the
  `.msltrace.json` file.

  Two sources:
    * `from_game_states/2` — full `ExPhil.Bridge.GameState` frames (from the
      sim loop): every player field is exact.
    * `from_slim/2` — the compact `examples.jsonl` frames the regret viewer
      exports (`a1/a2/x1/y1/x2/y2/pct2/g1`): action frame is reconstructed as
      frames-since-transition, shield/stocks/jumps use defaults. Good enough
      to animate; exact traces come from the full path.

  Every frame is written as a keyframe (the format allows early keyframes),
  which keeps the writer trivial and the files small at 90 frames.
  """

  @player_fields ~w(charId actionId actionFrame x y facing grounded percent shield stocks jumps hitlag hitstun hurtbox reflect fastfall shielding inHitstun powershield dead)
  @input_fields ~w(buttons mainX mainY cX cY l r)
  @item_fields ~w(alive typeId state owner x y vx vy facing damage timer spawnId misc0 misc1 misc2)

  @char_ids %{"fox" => 1, "falco" => 22, "marth" => 18, "mewtwo" => 16, "sheik" => 7, "captainfalcon" => 2, "peach" => 9, "jigglypuff" => 15, "samus" => 13, "ganondorf" => 25, "link" => 6, "zelda" => 19, "iceclimbers" => 10, "mario" => 0, "luigi" => 17, "drmario" => 21, "pikachu" => 12, "yoshi" => 14, "donkeykong" => 3, "kirby" => 4, "bowser" => 5, "ness" => 8, "younglink" => 20, "roy" => 26, "pichu" => 23, "gameandwatch" => 24, "mrgamewatch" => 24}

  @doc "Sim-internal character id from a Peppi character name (\"Fox\", \"Captain Falcon\", …); Fox when unknown."
  def char_id(name) when is_binary(name), do: Map.get(@char_ids, name |> String.downcase() |> String.replace(~r/[^a-z0-9]/, ""), 1)
  def char_id(_), do: 1

  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @shield MapSet.new(178..182)

  @doc "Trace from full GameStates (players 1 and 2). Options: `:stage` (32), `:chars` ([1, 1]), `:seed`, `:label`."
  def from_game_states(states, opts \\ []) do
    rows =
      states
      |> Enum.with_index()
      |> Enum.map(fn {gs, i} ->
        [0, i, nil, Enum.map([1, 2], fn port -> player_row(gs.players[port], Keyword.get(opts, :chars, [1, 1]) |> Enum.at(port - 1)) end)]
      end)

    envelope(rows, states |> Enum.map(& &1.frame) |> List.first(), opts)
  end

  defp player_row(p, char) do
    [
      char, p.action, p.action_frame || 0, p.x, p.y, p.facing, b(p.on_ground), p.percent, p.shield_strength || 60.0, p.stock, p.jumps_left || 0,
      0, p.hitstun_frames_left || 0, 0, 0, 0, b(MapSet.member?(@shield, p.action)), b(MapSet.member?(@hitstun, p.action)), 0, b(p.action <= 10)
    ]
  end

  @doc """
  Trace from the regret viewer's slim frames (`[%{"a1","a2","x1","y1","x2","y2","pct2","g1"}, ...]`
  or the array-encoded form). Options as `from_game_states/2`.
  """
  def from_slim(frames, opts \\ []) do
    frames = Enum.map(frames, &normalize_slim/1)

    {rows, _} =
      frames
      |> Enum.with_index()
      |> Enum.map_reduce({nil, 0, nil, 0}, fn {f, i}, {la1, af1, la2, af2} ->
        af1 = if f.a1 == la1, do: af1 + 1, else: 0
        af2 = if f.a2 == la2, do: af2 + 1, else: 0

        row = [
          0, i, nil,
          [
            slim_player(1, f.a1, af1, f.x1, f.y1, f.g1, 0.0, Keyword.get(opts, :chars, [1, 1]) |> Enum.at(0), facing_from(frames, i, :x1)),
            slim_player(2, f.a2, af2, f.x2, f.y2, f.g2, f.pct2, Keyword.get(opts, :chars, [1, 1]) |> Enum.at(1), facing_from(frames, i, :x2))
          ]
        ]

        {row, {f.a1, af1, f.a2, af2}}
      end)

    envelope(rows, Keyword.get(opts, :start_frame, 0), opts)
  end

  defp slim_player(_port, action, af, x, y, grounded, pct, char, facing) do
    [char, action, af, x, y, facing, b(grounded), pct, 60.0, 4, 2, 0, 0, 0, 0, 0, b(MapSet.member?(@shield, action)), b(MapSet.member?(@hitstun, action)), 0, b(action <= 10)]
  end

  # slim frames carry no facing; infer from horizontal motion, default right
  defp facing_from(frames, i, key) do
    cur = Map.get(Enum.at(frames, i), key)
    prev = if i > 0, do: Map.get(Enum.at(frames, i - 1), key), else: cur
    cond do
      cur - prev > 0.05 -> 1
      cur - prev < -0.05 -> -1
      true -> 1
    end
  end

  defp normalize_slim(%{"a1" => _} = f), do: %{a1: f["a1"], a2: f["a2"], x1: f["x1"], y1: f["y1"], x2: f["x2"], y2: f["y2"], pct2: f["pct2"], g1: f["g1"] == true or f["g1"] == 1, g2: f["y2"] != nil and f["y2"] <= 0.01}
  defp normalize_slim([a1, a2, x1, y1, x2, y2, pct2, g1 | _]), do: %{a1: a1, a2: a2, x1: x1, y1: y1, x2: x2, y2: y2, pct2: pct2, g1: g1 == 1 or g1 == true, g2: y2 <= 0.01}

  defp envelope(rows, start_frame, opts) do
    chars = Keyword.get(opts, :chars, [1, 1])

    %{
      format: "MSLTRACE1",
      schemaVersion: 1,
      producer: %{name: "exphil", version: nil},
      createdAt: DateTime.utc_now() |> DateTime.to_iso8601(),
      match: %{
        stageId: Keyword.get(opts, :stage, 32),
        numPlayers: 2,
        isTeams: false,
        players: [%{port: 1, charId: Enum.at(chars, 0), teamId: 0}, %{port: 2, charId: Enum.at(chars, 1), teamId: 1}],
        startFrame: 0,
        start: %{mode: "exphil-rollout", traceFrame: 0, simFrameId: start_frame || 0, randomSeed: Keyword.get(opts, :seed, 0)}
      },
      inputs: %{encoding: "sparse-delta-v1", keyframeInterval: 60, fields: @input_fields, players: [[[0, 0, [0, 0, 0, 0, 0, 0, 0]]], [[0, 0, [0, 0, 0, 0, 0, 0, 0]]]]},
      frames: %{encoding: "sparse-delta-v1", keyframeInterval: 60, fields: ["frame", "randomSeed", "players"], playerFields: @player_fields, rows: rows},
      items: %{encoding: "sparse-delta-v1", keyframeInterval: 60, fields: @item_fields, rows: []},
      metadata: %{provenance: %{source: Keyword.get(opts, :label, "exphil rollout")}}
    }
  end

  @doc "Write a trace map to `path` (compact JSON)."
  def write!(trace, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(trace))
    path
  end

  defp b(true), do: 1
  defp b(_), do: 0
end
