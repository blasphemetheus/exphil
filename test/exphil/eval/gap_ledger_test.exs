defmodule ExPhil.Eval.GapLedgerTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.GapLedger

  @today "2026-07-23"

  describe "gap_id/4" do
    test "is stable and 8-hex-suffixed" do
      id = GapLedger.gap_id("detector", "a.slp", 4180, "neutral_loss")
      assert id == GapLedger.gap_id("detector", "a.slp", 4180, "neutral_loss")
      assert String.starts_with?(id, "g_")
      assert String.length(id) == 10
    end

    test "differs on any identity field" do
      base = GapLedger.gap_id("detector", "a.slp", 4180, "neutral_loss")
      refute base == GapLedger.gap_id("manual", "a.slp", 4180, "neutral_loss")
      refute base == GapLedger.gap_id("detector", "b.slp", 4180, "neutral_loss")
      refute base == GapLedger.gap_id("detector", "a.slp", 4181, "neutral_loss")
      refute base == GapLedger.gap_id("detector", "a.slp", 4180, "dropped_punish")
    end

    test "coverage gaps (nil slp) still get a stable id" do
      id = GapLedger.gap_id("coverage", nil, nil, "d15-30|zmid")
      assert id == GapLedger.gap_id("coverage", nil, nil, "d15-30|zmid")
    end
  end

  describe "entry/2" do
    test "normalizes atom keys and fills defaults" do
      g = GapLedger.entry(%{source: "detector", slp: "a.slp", frame: 100, type: :neutral_loss}, today: @today)
      assert g["type"] == "neutral_loss"
      assert g["status"] == "new"
      assert g["created"] == @today
      assert g["evidence"] == %{}
      assert String.starts_with?(g["id"], "g_")
    end

    test "respects an explicit id/status" do
      g = GapLedger.entry(%{"id" => "g_custom01", "status" => "drilled", "type" => "x"}, today: @today)
      assert g["id"] == "g_custom01"
      assert g["status"] == "drilled"
    end
  end

  describe "append/3" do
    test "adds new gaps and reports the count" do
      {ledger, added} =
        GapLedger.append(GapLedger.new(), [
          %{source: "detector", slp: "a.slp", frame: 100, type: :neutral_loss},
          %{source: "detector", slp: "a.slp", frame: 500, type: :dropped_punish}
        ], today: @today)

      assert added == 2
      assert length(ledger["gaps"]) == 2
    end

    test "dedupes by id and preserves the first-seen status" do
      seed = %{source: "detector", slp: "a.slp", frame: 100, type: :neutral_loss}
      {ledger, _} = GapLedger.append(GapLedger.new(), [seed], today: @today)
      ledger = GapLedger.set_status(ledger, hd(ledger["gaps"])["id"], "drilled")

      # Re-detecting the same gap must NOT reset it to "new".
      {ledger, added} = GapLedger.append(ledger, [seed], today: @today)
      assert added == 0
      assert length(ledger["gaps"]) == 1
      assert hd(ledger["gaps"])["status"] == "drilled"
    end
  end

  describe "set_status/3 & by_status/2" do
    test "advances status and filters" do
      {ledger, _} =
        GapLedger.append(GapLedger.new(), [
          %{source: "detector", slp: "a.slp", frame: 100, type: :x},
          %{source: "detector", slp: "a.slp", frame: 200, type: :y}
        ], today: @today)

      target = hd(ledger["gaps"])["id"]
      ledger = GapLedger.set_status(ledger, target, "mined")
      assert length(GapLedger.by_status(ledger, "mined")) == 1
      assert length(GapLedger.by_status(ledger, "new")) == 1
    end

    test "ignores invalid status" do
      {ledger, _} = GapLedger.append(GapLedger.new(), [%{type: :x, frame: 1}], today: @today)
      id = hd(ledger["gaps"])["id"]
      assert GapLedger.set_status(ledger, id, "bogus") == ledger
    end
  end

  describe "load/1" do
    test "returns an empty ledger for a missing file" do
      assert GapLedger.load("/nonexistent/gaps.json") == GapLedger.new()
    end
  end
end
