defmodule ExPhil.Eval.StateStreamDiff do
  @moduledoc """
  Reconciles the PARSED and LIVE state streams for the same game (task #8,
  GOTCHAS #81).

  A policy is trained on features parsed out of a `.slp` by Peppi, but at
  inference it receives features read through the libmelee bridge. Those two
  readings of the same frame do not agree, and nothing in the training loss
  can reveal it — the loss is computed entirely in parsed space. The observed
  consequence: identical weights scored 99.3% button agreement on parsed
  fixture states and collapsed to a constant (B=100%, never shines) on live
  states.

  This module derives the exact mapping from a *pair* — a `.slp` and the
  recorder's own per-frame trace of that same run
  (`MULTISHINE_TRACE=1`, see `test/fixtures/statestream/README.md`).

  ## What it does

  1. **Align.** Frame numbering differs between the two (a replay starts at
     -123 during countdown; the trace starts at f0), so alignment is anchored
     on the first entry into an unambiguous action — jumpsquat by default —
     rather than assuming f0 maps to parsed frame 0.
  2. **Verify the alignment** on fields that must agree if the frames really
     correspond (`action`, `on_ground`, `y`). A bad anchor shows up as a low
     agreement score instead of silently producing a wrong mapping.
  3. **Emit the per-action mapping** of `parsed action_frame` to
     `live action_frame`, plus whichever other fields shift.

  ## Why the mapping is a per-action table and not a formula

  Two tempting rules both fail against the fixtures:

    * *"live af is 1-based frames since the action was entered"* — fails
      because `action_frame` FREEZES on repeated frames (act 323 sits at
      live af 11 while the frame counter keeps climbing).
    * *"parsed af is the same counter, 0-based"* — fails because the value
      Peppi reports on the first frame of an action is 0 for some actions
      (24, 29, 42, 323, 366) and 1 for others (360, 361, 365).

  What IS stable is the per-action offset: for a given action id, the
  difference `live_af - parsed_af` is constant. Hence a table.

  ## Usage

      {:ok, report} = StateStreamDiff.diff(slp_path, trace_path)
      report.mapping[24].delta  #=> 1

  `scripts/diff_state_streams.exs` is the CLI printer over this module.
  """

  alias ExPhil.Constants
  alias ExPhil.Data.Peppi

  @trace_re ~r/f(-?\d+)\s+act=(-?\d+)\s+af=(-?\d+)\s+gnd=(\w+)\s+y=(-?[\d.e+-]+)\s+vy=(-?[\d.e+-]+)/

  # The trace prints y rounded to 2dp, so exact equality is not available.
  @default_y_tolerance 0.01

  @type trace_row :: %{
          f: integer(),
          act: integer(),
          af: integer(),
          gnd: boolean(),
          y: float(),
          vy: float()
        }

  @type parsed_row :: %{
          f: integer(),
          act: integer(),
          af: integer(),
          gnd: boolean(),
          y: float(),
          vy: float()
        }

  @type action_mapping :: %{
          delta: integer() | nil,
          deltas: [integer()],
          n: non_neg_integer(),
          parsed_af: Range.t(),
          live_af: Range.t(),
          consistent?: boolean()
        }

  @doc """
  Full reconciliation for one pair.

  Options:
    * `:port` — player port to compare (default 1)
    * `:anchor_action` — action id to align on (default `Constants.jumpsquat/0`)
    * `:y_tolerance` — max |Δy| still counted as agreement (default #{@default_y_tolerance})

  Returns `{:ok, report}` where report has `:offset`, `:frames_compared`,
  `:agreement`, `:mapping`, `:inconsistent_actions` and `:shifted_fields`.
  """
  @spec diff(Path.t(), Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def diff(slp_path, trace_path, opts \\ []) do
    port = Keyword.get(opts, :port, 1)

    with {:ok, parsed} <- parse_replay(slp_path, port),
         {:ok, trace} <- parse_trace(trace_path),
         {:ok, offset} <- align(parsed, trace, opts) do
      {:ok, build_report(parsed, trace, offset, opts)}
    end
  end

  @doc """
  Parse the recorder's live trace log.

  The log is raw recorder stdout, so it also contains build noise; only
  `[trace]` lines are considered.
  """
  @spec parse_trace(Path.t()) :: {:ok, [trace_row()]} | {:error, term()}
  def parse_trace(path) do
    case File.read(path) do
      {:ok, contents} ->
        rows =
          contents
          |> String.split("\n")
          |> Enum.filter(&String.contains?(&1, "[trace]"))
          |> Enum.flat_map(&parse_trace_line/1)

        if rows == [], do: {:error, {:no_trace_lines, path}}, else: {:ok, rows}

      {:error, reason} ->
        {:error, {:trace_unreadable, path, reason}}
    end
  end

  defp parse_trace_line(line) do
    case Regex.run(@trace_re, line) do
      [_, f, act, af, gnd, y, vy] ->
        [
          %{
            f: String.to_integer(f),
            act: String.to_integer(act),
            af: String.to_integer(af),
            gnd: gnd == "true",
            y: to_float(y),
            vy: to_float(vy)
          }
        ]

      nil ->
        []
    end
  end

  defp to_float(s) do
    case Float.parse(s) do
      {v, _} -> v
      :error -> 0.0
    end
  end

  @doc """
  Parse one player's per-frame state out of a `.slp` via Peppi.
  """
  @spec parse_replay(Path.t(), pos_integer()) :: {:ok, [parsed_row()]} | {:error, term()}
  def parse_replay(path, port \\ 1) do
    case Peppi.parse(path) do
      {:ok, replay} ->
        rows =
          replay.frames
          |> Enum.flat_map(fn frame ->
            case frame.players[port] do
              nil ->
                []

              p ->
                [
                  %{
                    f: frame.frame_number,
                    act: trunc(p.action),
                    af: trunc(p.action_frame),
                    gnd: p.on_ground,
                    y: p.y,
                    vy: p.speed_y_self || 0.0
                  }
                ]
            end
          end)

        if rows == [], do: {:error, {:no_frames_for_port, path, port}}, else: {:ok, rows}

      {:error, reason} ->
        {:error, {:replay_unparseable, path, reason}}
    end
  end

  @doc """
  Frame-number offset such that `parsed_frame = trace_frame + offset`.

  Anchored on the first *entry* into `:anchor_action` (a transition into it,
  not merely a frame where it holds) on both sides. Frame 0 is deliberately
  not used as the anchor: the two streams cover different amounts of menu and
  countdown, so their frame numbering has no reason to coincide.
  """
  @spec align([parsed_row()], [trace_row()], keyword()) :: {:ok, integer()} | {:error, term()}
  def align(parsed, trace, opts \\ []) do
    anchor = Keyword.get(opts, :anchor_action, Constants.jumpsquat())

    with {:ok, p_frame} <- first_entry_frame(parsed, anchor, :parsed),
         {:ok, t_frame} <- first_entry_frame(trace, anchor, :trace) do
      {:ok, p_frame - t_frame}
    end
  end

  defp first_entry_frame(rows, action, side) do
    rows
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.find(fn [a, b] -> a.act != action and b.act == action end)
    |> case do
      [_, b] -> {:ok, b.f}
      nil -> {:error, {:anchor_not_found, side, action}}
    end
  end

  @doc """
  Per-action `action_frame` mapping over aligned frames.

  Only frames where `action` already agrees are used — a frame where the two
  streams disagree about which action is active says nothing about the af
  convention. `delta` is `live_af - parsed_af`; it is `nil` when an action
  shows more than one delta (which would mean the offset is not a per-action
  constant, and the table is not the whole story).
  """
  @spec mapping([{trace_row(), parsed_row()}]) :: %{integer() => action_mapping()}
  def mapping(pairs) do
    pairs
    # Only frames whose action already agrees are informative, and af < 0 is a
    # sentinel ("no action frame") rather than a counter value.
    |> Enum.filter(fn {t, p} -> t.act == p.act and t.af >= 0 and p.af >= 0 end)
    |> Enum.group_by(fn {t, _} -> t.act end)
    |> Map.new(fn {act, rows} ->
      deltas = rows |> Enum.map(fn {t, p} -> t.af - p.af end) |> Enum.uniq() |> Enum.sort()
      parsed_afs = Enum.map(rows, fn {_, p} -> p.af end)
      live_afs = Enum.map(rows, fn {t, _} -> t.af end)

      {act,
       %{
         delta: if(length(deltas) == 1, do: hd(deltas), else: nil),
         deltas: deltas,
         n: length(rows),
         parsed_af: Enum.min(parsed_afs)..Enum.max(parsed_afs),
         live_af: Enum.min(live_afs)..Enum.max(live_afs),
         consistent?: length(deltas) == 1
       }}
    end)
  end

  @doc """
  Pair up aligned frames as `{trace_row, parsed_row}`.
  """
  @spec pairs([parsed_row()], [trace_row()], integer()) :: [{trace_row(), parsed_row()}]
  def pairs(parsed, trace, offset) do
    by_frame = Map.new(parsed, &{&1.f, &1})

    Enum.flat_map(trace, fn t ->
      case by_frame[t.f + offset] do
        nil -> []
        p -> [{t, p}]
      end
    end)
  end

  defp build_report(parsed, trace, offset, opts) do
    tol = Keyword.get(opts, :y_tolerance, @default_y_tolerance)
    pairs = pairs(parsed, trace, offset)
    n = length(pairs)

    agreement = %{
      action: ratio(pairs, n, fn {t, p} -> t.act == p.act end),
      on_ground: ratio(pairs, n, fn {t, p} -> t.gnd == p.gnd end),
      y: ratio(pairs, n, fn {t, p} -> abs(t.y - p.y) <= tol end),
      action_frame: ratio(pairs, n, fn {t, p} -> t.af == p.af end)
    }

    map = mapping(pairs)

    %{
      offset: offset,
      frames_compared: n,
      agreement: agreement,
      mapping: map,
      inconsistent_actions:
        map
        |> Enum.reject(fn {_, m} -> m.consistent? end)
        |> Enum.map(&elem(&1, 0))
        |> Enum.sort(),
      shifted_fields: shifted_fields(agreement),
      y_tolerance: tol
    }
  end

  defp ratio(_pairs, 0, _fun), do: 0.0
  defp ratio(pairs, n, fun), do: Enum.count(pairs, fun) / n

  # A field "shifts" when the two streams disagree on it anywhere. Listed so a
  # future recorder or Peppi change that starts shifting `y` or `on_ground`
  # surfaces here instead of silently degrading a policy.
  defp shifted_fields(agreement) do
    agreement
    |> Enum.reject(fn {_field, ratio} -> ratio == 1.0 end)
    |> Enum.map(&elem(&1, 0))
    |> Enum.sort()
  end

  @doc """
  Convert a parsed `action_frame` into the live convention using a mapping.

  Returns the af unchanged when the action is absent from the table — callers
  that need to know should check `Map.has_key?/2` themselves, since an unknown
  action is "no evidence", not "no shift".
  """
  @spec to_live_af(%{integer() => action_mapping()}, integer(), integer()) :: integer()
  def to_live_af(mapping, action, parsed_af) do
    case mapping[action] do
      %{delta: d} when is_integer(d) -> parsed_af + d
      _ -> parsed_af
    end
  end
end
