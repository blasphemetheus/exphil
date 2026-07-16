defmodule ExPhil.Eval.ScenarioScan do
  @moduledoc """
  Pathology-moment detectors for the scenario evaluation suite (task #18).

  Scans parsed replays for *handoff candidates*: frames where a known
  pathology situation exists (opponent behind, tech chase, edgeguard,
  getup, idle deadlock). The scenario runner replays recorded inputs up to
  such a frame and hands port 1 to the policy — so a candidate must be a
  moment where "what does the policy do next?" is a well-posed question.

  Detectors are pure functions over slim frame lists so they are
  unit-testable with synthetic sequences; `load/1` produces that shape from
  a .slp. Action-state sets mirror `ExPhil.Interp.ReplayStats` so numbers
  stay comparable across the interp toolkit.

  Port convention: P1 = policy, P2 = opponent (probe-game convention).
  """

  alias ExPhil.Data.Peppi

  @types [:opponent_behind, :tech_chase, :edgeguard, :getup, :idle_deadlock]

  @idle 14
  # Knockdown lifecycle (ReplayStats): bound/wait/getup/roll/tech families
  @kd_entries [183, 191, 199, 200, 201]
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  # P1 knocked down WITHOUT teching = a getup decision is coming
  @p1_knockdown [183, 191]

  # Final Destination geometry
  @fd_edge 85.5

  # Defaults (tunable per call)
  @default_min_frame 300
  @default_gap 240

  def types, do: @types

  @doc """
  Load a replay into the slim frame shape the detectors consume.

  Returns `{:ok, %{frames: [%{frame:, p1:, p2:}], meta: ReplayMeta}}` where
  each player is `player_summary/1`, or `{:error, reason}` for unparseable
  files (stub/corrupt replays exist in the wild — GOTCHAS #58). Frames keep
  the replay's own in-game frame numbers (starting at -123); frames missing
  either port are dropped.
  """
  def load(path) do
    case Peppi.parse(Path.expand(path)) do
      {:ok, replay} ->
        frames =
          replay.frames
          |> Enum.filter(fn f -> f.players[1] != nil and f.players[2] != nil end)
          |> Enum.map(fn f ->
            %{
              frame: f.frame_number,
              p1: player_summary(f.players[1]),
              p2: player_summary(f.players[2])
            }
          end)

        {:ok, %{frames: frames, meta: replay.metadata}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Slim per-player view used by both the scanner and the scenario runner
  (works for `Peppi.PlayerFrame` and live `ExPhil.Bridge.Player` alike).
  """
  def player_summary(p) do
    %{
      x: p.x * 1.0,
      y: p.y * 1.0,
      action: trunc(p.action || 0),
      facing: p.facing,
      on_ground: p.on_ground,
      stock: p.stock,
      percent: (p.percent || 0) * 1.0
    }
  end

  @doc """
  Run every detector over `frames` and return curated candidates:
  `[%{type:, frame:, note:}]`, sorted by frame.

  ## Options
    - `:min_frame` — earliest in-game frame to accept (default #{@default_min_frame};
      leaves a real prefix to replay)
    - `:max_frac` — latest acceptable point as a fraction of the last frame
      (default 2/3; leaves room for the response window)
    - `:gap` — minimum frames between same-type candidates (default #{@default_gap})
    - `:types` — subset of `types/0` to run
  """
  def scan(frames, opts \\ []) do
    min_frame = Keyword.get(opts, :min_frame, @default_min_frame)
    max_frac = Keyword.get(opts, :max_frac, 2 / 3)
    gap = Keyword.get(opts, :gap, @default_gap)
    wanted = Keyword.get(opts, :types, @types)

    last = if frames == [], do: 0, else: List.last(frames).frame
    max_frame = trunc(last * max_frac)

    wanted
    |> Enum.flat_map(fn type -> detect(type, frames) end)
    |> Enum.filter(fn c -> c.frame >= min_frame and c.frame <= max_frame end)
    |> Enum.group_by(& &1.type)
    |> Enum.flat_map(fn {_type, cands} -> space_out(Enum.sort_by(cands, & &1.frame), gap) end)
    |> Enum.sort_by(& &1.frame)
  end

  @doc "Run a single detector. Returns `[%{type:, frame:, note:}]` (unbounded)."
  def detect(:opponent_behind, frames), do: opponent_behind(frames)
  def detect(:tech_chase, frames), do: tech_chase(frames)
  def detect(:edgeguard, frames), do: edgeguard(frames)
  def detect(:getup, frames), do: getup(frames)
  def detect(:idle_deadlock, frames), do: idle_deadlock(frames)

  # ==========================================================================
  # Detectors
  # ==========================================================================

  @doc """
  P2 within 30 x-units BEHIND P1's facing, both grounded, P1 not in
  hitstun — persisting `persist` frames (default 12). Candidate at onset.
  """
  def opponent_behind(frames, persist \\ 12) do
    flags =
      Enum.map(frames, fn %{p1: p1, p2: p2} ->
        dx = p2.x - p1.x

        p1.on_ground and p2.on_ground and dx != 0 and abs(dx) <= 30.0 and
          sign(dx) != p1.facing and not MapSet.member?(@hitstun, p1.action)
      end)

    for i <- persistent_onsets(flags, persist) do
      f = Enum.at(frames, i)
      dx = Float.round(f.p2.x - f.p1.x, 1)

      %{
        type: :opponent_behind,
        frame: f.frame,
        note: "dx=#{dx} p1_facing=#{f.p1.facing} p1_action=#{f.p1.action}"
      }
    end
  end

  @doc """
  P2 enters the knockdown lifecycle (ReplayStats entry classes: missed-tech
  bound 183/191 or tech 199/200/201) from outside it. Candidate at the
  entry frame — the tech-chase read starts here.
  """
  def tech_chase(frames) do
    frames
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [a, b] ->
      b.p2.action in @kd_entries and not MapSet.member?(@lifecycle, a.p2.action)
    end)
    |> Enum.map(fn [_, b] ->
      %{
        type: :tech_chase,
        frame: b.frame,
        note: "entry=#{entry_name(b.p2.action)} p2_x=#{Float.round(b.p2.x, 1)} dist=#{Float.round(abs(b.p2.x - b.p1.x), 1)}"
      }
    end)
  end

  @doc """
  P2 offstage on FD (|x| > #{@fd_edge} or y < -10) while P1 is onstage,
  persisting `persist` frames (default 20). Candidate at onset.
  """
  def edgeguard(frames, persist \\ 20) do
    flags =
      Enum.map(frames, fn %{p1: p1, p2: p2} ->
        p2_off = abs(p2.x) > @fd_edge or p2.y < -10.0
        p1_on = abs(p1.x) <= @fd_edge and p1.y > -5.0
        p2_off and p1_on and p2.stock > 0
      end)

    for i <- persistent_onsets(flags, persist) do
      f = Enum.at(frames, i)

      %{
        type: :edgeguard,
        frame: f.frame,
        note: "p2=(#{Float.round(f.p2.x, 1)},#{Float.round(f.p2.y, 1)}) p1_x=#{Float.round(f.p1.x, 1)}"
      }
    end
  end

  @doc """
  P1 (the policy) enters a missed-tech knockdown (183/191). A getup
  decision follows — the pathology is stalling in down-wait.

  Knockdowns within ~5 units of the FD edge are excluded: the downed
  character slides off the stage into a ledge grab (observed at x=-84.3),
  which is a ledge scenario, not a getup decision.
  """
  def getup(frames, max_x \\ @fd_edge - 5) do
    frames
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [a, b] ->
      b.p1.action in @p1_knockdown and abs(b.p1.x) <= max_x and
        not MapSet.member?(@lifecycle, a.p1.action)
    end)
    |> Enum.map(fn [_, b] ->
      %{
        type: :getup,
        frame: b.frame,
        note: "entry=#{b.p1.action} p1=(#{Float.round(b.p1.x, 1)},#{Float.round(b.p1.y, 1)})"
      }
    end)
  end

  @doc """
  Both ports in Wait (action #{@idle}) for >= `min_run` consecutive frames
  (default 90). Candidate 90 frames INTO the run — the deadlock is
  established, and breaking it is the policy's job.
  """
  def idle_deadlock(frames, min_run \\ 90) do
    flags = Enum.map(frames, fn %{p1: p1, p2: p2} -> p1.action == @idle and p2.action == @idle end)

    flags
    |> Enum.with_index()
    |> Enum.chunk_by(fn {f, _} -> f end)
    |> Enum.filter(fn [{f, _} | _] = chunk -> f and length(chunk) >= min_run end)
    |> Enum.map(fn [{_, start} | _] = chunk ->
      f = Enum.at(frames, start + min_run - 1)

      %{
        type: :idle_deadlock,
        frame: f.frame,
        note: "run=#{length(chunk)}f dist=#{Float.round(abs(f.p2.x - f.p1.x), 1)}"
      }
    end)
  end

  # ==========================================================================
  # Helpers
  # ==========================================================================

  # Indices where a flag run of at least `persist` true frames begins.
  defp persistent_onsets(flags, persist) do
    flags
    |> Enum.with_index()
    |> Enum.chunk_by(fn {f, _} -> f end)
    |> Enum.filter(fn [{f, _} | _] = chunk -> f and length(chunk) >= persist end)
    |> Enum.map(fn [{_, i} | _] -> i end)
  end

  # Keep candidates at least `gap` frames apart (list pre-sorted by frame).
  defp space_out(cands, gap) do
    Enum.reduce(cands, [], fn c, acc ->
      case acc do
        [prev | _] when c.frame - prev.frame < gap -> acc
        _ -> [c | acc]
      end
    end)
    |> Enum.reverse()
  end

  defp entry_name(183), do: "missed_bound_u"
  defp entry_name(191), do: "missed_bound_d"
  defp entry_name(199), do: "tech_in_place"
  defp entry_name(200), do: "tech_roll_f"
  defp entry_name(201), do: "tech_roll_b"

  defp sign(x) when x < 0, do: -1
  defp sign(_), do: 1
end
