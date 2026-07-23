defmodule ExPhil.Eval.GapLedger do
  @moduledoc """
  The shared currency of the data flywheel (DATA_FLYWHEEL_DESIGN_2026-07-23).

  Everything in stage A (noticing gaps) produces, and everything in stage B
  (acquiring data) consumes, one file: `gaps.json`. A *gap* is a moment or
  bucket where the bot's training data is missing something — surfaced by a
  detector, a coverage diff, an uncertainty spike, or a manual bookmark.

  This module is the pure load/merge/append/dedupe core (JSON on disk); the
  producer/consumer scripts (`auto_bookmarks.exs`, `coach_report.exs`,
  `coverage_ledger.exs`, `scan_bookmarks.exs`, `build_seed_dir.exs`) all
  route through it so ids and status transitions stay consistent.

  ## Gap shape (string keys — JSON round-trips)

      %{
        "id"       => "g_<8hex>",   # stable hash of {source, slp, frame, type}
        "source"   => "manual|detector|coverage|uncertainty|coach",
        "type"     => "neutral_loss|dropped_punish|<bucket key>|...",
        "slp"      => "path" | nil, # coverage gaps have no single moment
        "frame"    => 4180 | nil,   # drill handoff frame
        "note"     => "human-readable evidence",
        "status"   => "new|mined|drilled|verified",
        "created"  => "2026-07-23",
        "evidence" => %{}           # source-specific extras
      }

  ## Dedupe & status

  Gaps are identified by `id`. `append/2` keeps the FIRST-seen gap's
  `status` (a later re-detection must not reset a `drilled` gap back to
  `new`); everything else on a duplicate is dropped. Status is advanced by
  the tools that do the work (`build_seed_dir` -> "drilled", a passing
  scenario battery -> "verified"), never guessed here.
  """

  @statuses ~w(new mined drilled verified)
  @sources ~w(manual detector coverage uncertainty coach)

  @doc "An empty ledger."
  def new, do: %{"gaps" => []}

  @doc """
  Load a ledger from `path`. Returns `new/0` if the file is absent so
  producers can append unconditionally.
  """
  def load(path) do
    case File.read(path) do
      {:ok, bin} ->
        case Jason.decode(bin) do
          {:ok, %{"gaps" => gaps} = m} when is_list(gaps) -> m
          {:ok, _} -> new()
          {:error, _} -> new()
        end

      {:error, _} ->
        new()
    end
  end

  @doc """
  Stable 8-hex id for a gap from its identity tuple. Coverage gaps (no slp)
  still get a stable id from `{source, nil, frame_or_type, type}`.
  """
  def gap_id(source, slp, frame, type) do
    digest =
      :crypto.hash(:sha256, :erlang.term_to_binary({source, slp, frame, type}))
      |> Base.encode16(case: :lower)
      |> binary_part(0, 8)

    "g_" <> digest
  end

  @doc """
  Normalize a loose attribute map (atom or string keys) into a full gap map
  with a computed id and defaults. `:created` defaults to today (UTC);
  pass it explicitly for deterministic tests.
  """
  def entry(attrs, opts \\ []) do
    a = stringify(attrs)
    source = a["source"] || "detector"
    slp = a["slp"]
    frame = a["frame"]
    type = a["type"] || "unknown"

    %{
      "id" => a["id"] || gap_id(source, slp, frame, type),
      "source" => source,
      "type" => to_string(type),
      "slp" => slp,
      "frame" => frame,
      "note" => a["note"] || "",
      "status" => a["status"] || "new",
      "created" => a["created"] || opts[:created] || Date.to_iso8601(today(opts)),
      "evidence" => a["evidence"] || %{}
    }
  end

  @doc """
  Append `entries` (each a loose attr map or a full gap) to `ledger`,
  deduping by id. On collision the existing gap wins (preserves its
  status/created); genuinely new gaps are appended in order. Returns the
  updated ledger and the count actually added.
  """
  def append(ledger, entries, opts \\ []) do
    existing = ledger["gaps"] || []
    seen = MapSet.new(existing, & &1["id"])

    {rev_new, _seen} =
      Enum.reduce(entries, {[], seen}, fn raw, {acc, seen} ->
        gap = entry(raw, opts)

        if MapSet.member?(seen, gap["id"]) do
          {acc, seen}
        else
          {[gap | acc], MapSet.put(seen, gap["id"])}
        end
      end)

    added = Enum.reverse(rev_new)
    {%{ledger | "gaps" => existing ++ added}, length(added)}
  end

  @doc "Set `status` on the gap with `id` (no-op if absent or invalid status)."
  def set_status(ledger, id, status) when status in @statuses do
    gaps =
      Enum.map(ledger["gaps"] || [], fn g ->
        if g["id"] == id, do: Map.put(g, "status", status), else: g
      end)

    %{ledger | "gaps" => gaps}
  end

  def set_status(ledger, _id, _status), do: ledger

  @doc "Write `ledger` to `path` (pretty JSON), creating parent dirs."
  def save(ledger, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(ledger, pretty: true))
    ledger
  end

  @doc "Gaps filtered by status."
  def by_status(ledger, status), do: Enum.filter(ledger["gaps"] || [], &(&1["status"] == status))

  @doc "Valid source/status vocabularies (for CLI validation)."
  def sources, do: @sources
  def statuses, do: @statuses

  # -- helpers ---------------------------------------------------------------

  defp stringify(map) do
    Map.new(map, fn {k, v} -> {to_string(k), v} end)
  end

  defp today(opts) do
    case opts[:today] do
      %Date{} = d -> d
      iso when is_binary(iso) -> Date.from_iso8601!(iso)
      _ -> Date.utc_today()
    end
  end
end
