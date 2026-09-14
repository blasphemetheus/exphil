defmodule ExPhil.Eval.EarlyTeacherGateTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.EarlyTeacherGate

  defp report do
    %{
      "cases" =>
        for name <- [
              "neutral:2228",
              "neutral:2566",
              "sustain:900",
              "recovery:4",
              "recovery:75",
              "recovery:146"
            ] do
          %{
            "case" => name,
            "rows" =>
              for index <- 0..117 do
                %{"index" => index, "tf_argmax_correct" => true, "target_probability" => 0.99}
              end
          }
        end
    }
  end

  test "requires every case, every target, and confident correct early actions" do
    assert EarlyTeacherGate.check(report()).ready
    refute EarlyTeacherGate.check(%{}).ready
    [first | rest] = report()["cases"]
    refute EarlyTeacherGate.check(%{"cases" => rest}).ready
    refute EarlyTeacherGate.check(%{"cases" => [first, first | rest]}).ready
    [row | rows] = first["rows"]

    for changed <- [
          %{row | "target_probability" => 0.94},
          %{row | "target_probability" => nil},
          %{row | "target_probability" => 2.0},
          %{row | "tf_argmax_correct" => false}
        ] do
      refute EarlyTeacherGate.check(%{"cases" => [%{first | "rows" => [changed | rows]} | rest]}).ready
    end

    refute EarlyTeacherGate.check(%{"cases" => [%{first | "rows" => rows} | rest]}).ready
  end

  defp clip(name, history, targets, opts \\ []) do
    bad = Keyword.get(opts, :bad_index)

    %{
      "case" => name,
      "history" => history,
      "targets" => targets,
      "rows" =>
        for index <- 0..(targets - 1) do
          %{
            "index" => index,
            "tf_argmax_correct" => index != bad,
            "target_probability" => 0.99
          }
        end
    }
  end

  test "gates the report's own case list, cold and warm, with per-case target counts" do
    names = ["4_cold:4", "4_warm:4", "502_cold:502", "502_warm:502"]

    report = %{
      "gate_cases" => names,
      "cases" => [
        %{"case" => "canonical", "targets" => 7077, "rows" => []},
        clip("4_cold:4", "cold", 118),
        clip("4_warm:4", "warm", 118),
        clip("502_cold:502", "cold", 358),
        clip("502_warm:502", "warm", 358)
      ]
    }

    result = EarlyTeacherGate.check(report)
    assert result.ready
    assert result.gate_cases == names
    assert Enum.map(result.cases, & &1.history) == ["cold", "warm", "cold", "warm"]
    # the canonical fixture is never a gated case
    refute Enum.any?(result.cases, &(&1.case == "canonical"))

    # a warm clip's early miss fails the whole gate; a miss after the first
    # 18 supervised targets does not
    swap = fn entry -> %{report | "cases" => Enum.map(report["cases"], &if(&1["case"] == entry["case"], do: entry, else: &1))} end
    refute EarlyTeacherGate.check(swap.(clip("4_warm:4", "warm", 118, bad_index: 3))).ready
    assert EarlyTeacherGate.check(swap.(clip("502_warm:502", "warm", 358, bad_index: 200))).ready

    # a case listed for gating but missing from the report, or short of its
    # declared targets, fails
    refute EarlyTeacherGate.check(%{report | "cases" => Enum.drop(report["cases"], -1)}).ready
    short = clip("502_cold:502", "cold", 358)
    refute EarlyTeacherGate.check(swap.(%{short | "rows" => tl(short["rows"])})).ready
    refute EarlyTeacherGate.check(%{report | "gate_cases" => []}).ready
  end
end
