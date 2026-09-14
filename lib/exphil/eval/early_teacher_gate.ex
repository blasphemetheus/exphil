defmodule ExPhil.Eval.EarlyTeacherGate do
  @moduledoc """
  Fixed early-window readiness gate; never replaces closed-loop validation.

  Every gated case must have all of its supervised targets present (indexed
  0..n-1, where `n` is the case's `targets`) and, over the first 18
  supervised targets, a correct teacher-forced conditional argmax on every
  one with a joint target probability of at least 0.95.

  The case list comes from the report's `gate_cases` (the fit script lists
  every recorded clip it measured, cold and warm alike); a report without
  it falls back to the original six cold cases. A case without `targets`
  is the original 118-target window.
  """
  @legacy_cases [
    "neutral:2228",
    "neutral:2566",
    "sustain:900",
    "recovery:4",
    "recovery:75",
    "recovery:146"
  ]
  @legacy_targets 118
  @early 18
  @minimum_probability 0.95

  def check(report) do
    gated = report["gate_cases"] || @legacy_cases
    cases = report["cases"] || []
    teachers = Enum.filter(cases, &(&1["case"] in gated))
    complete = gated != [] and Enum.sort(Enum.map(teachers, & &1["case"])) == Enum.sort(gated)

    results =
      Enum.map(teachers, fn entry ->
        rows = entry["rows"] || []
        targets = entry["targets"] || @legacy_targets
        early = Enum.filter(rows, &(&1["index"] in 0..(@early - 1)))

        valid =
          is_integer(targets) and targets >= @early and length(rows) == targets and
            Enum.sort(Enum.map(rows, & &1["index"])) == Enum.to_list(0..(targets - 1))

        minimum = Enum.map(early, & &1["target_probability"]) |> Enum.min(fn -> 0.0 end)
        correct = Enum.count(early, &(&1["tf_argmax_correct"] == true))

        probabilities_valid =
          Enum.all?(early, fn row ->
            probability = row["target_probability"]
            is_number(probability) and probability >= 0 and probability <= 1
          end)

        %{
          case: entry["case"],
          history: entry["history"] || "cold",
          targets: targets,
          ready:
            valid and probabilities_valid and correct == @early and is_number(minimum) and
              minimum >= @minimum_probability,
          early_correct: correct,
          minimum_target_probability: minimum,
          valid_rows: valid
        }
      end)

    %{
      ready: complete and Enum.all?(results, & &1.ready),
      complete_cases: complete,
      gate_cases: gated,
      cases: results
    }
  end
end
