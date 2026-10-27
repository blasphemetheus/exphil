# AI-generated; authorship: docs/guides/MAMBA_CONTROLLER_CHOICES.md.
defmodule MambaChoiceMetrics do
  @components [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

  def record(candidate, reference, row) do
    changes =
      Enum.filter(@components, fn key ->
        values(candidate[key], row) != values(reference[key], row)
      end)

    margins =
      Map.new(@components, fn key ->
        logits = values(reference.logits[key], row)

        margin =
          if key == :buttons do
            logits |> Enum.map(&abs/1) |> Enum.min()
          else
            [first, second | _] = Enum.sort(logits, :desc)
            first - second
          end

        {key, margin}
      end)

    max_logit_error =
      for key <- @components,
          {a, b} <-
            Enum.zip(values(candidate.logits[key], row), values(reference.logits[key], row)),
          reduce: 0.0 do
        max_error -> max(max_error, abs(a - b))
      end

    details =
      Map.new(changes, fn key ->
        candidate_probs = probabilities(values(candidate.logits[key], row), key)
        reference_probs = probabilities(values(reference.logits[key], row), key)

        {key,
         %{
           candidate_choice: values(candidate[key], row),
           reference_choice: values(reference[key], row),
           candidate_probabilities: candidate_probs,
           reference_probabilities: reference_probs,
           max_probability_change:
             Enum.zip_with(candidate_probs, reference_probs, &abs(&1 - &2)) |> Enum.max()
         }}
      end)

    %{
      changed_components: changes,
      changed_details: details,
      reference_margins: margins,
      max_logit_error: max_logit_error
    }
  end

  defp probabilities(logits, :buttons),
    do:
      Enum.map(logits, fn x ->
        if x >= 0, do: 1 / (1 + :math.exp(-x)), else: :math.exp(x) / (1 + :math.exp(x))
      end)

  defp probabilities(logits, _) do
    maximum = Enum.max(logits)
    exps = Enum.map(logits, &:math.exp(&1 - maximum))
    sum = Enum.sum(exps)
    Enum.map(exps, &(&1 / sum))
  end

  defp values(tensor, row),
    do: tensor |> Nx.slice_along_axis(row, 1, axis: 0) |> Nx.to_flat_list()

  def summarize(records) do
    changed = Enum.filter(records, &(&1.changed_components != []))

    %{
      decisions: length(records),
      changed_decisions: length(changed),
      ordinary_changes: Enum.count(changed, &(not &1.resync)),
      resync_changes: Enum.count(changed, & &1.resync),
      ordinary_decisions: Enum.count(records, &(not &1.resync)),
      resync_decisions: Enum.count(records, & &1.resync),
      by_component:
        Map.new(@components, fn key ->
          {key, Enum.count(changed, &(key in &1.changed_components))}
        end),
      max_logit_error: records |> Enum.map(& &1.max_logit_error) |> Enum.max(),
      minimum_reference_margin:
        Map.new(@components, fn key ->
          {key, records |> Enum.map(& &1.reference_margins[key]) |> Enum.min()}
        end),
      changed_examples: Enum.take(changed, 30)
    }
  end
end
