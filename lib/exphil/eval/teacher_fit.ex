defmodule ExPhil.Eval.TeacherFit do
  @moduledoc "Frozen teacher-forced action likelihoods; not a closed-loop success metric."
  @heads [:main_x, :main_y, :c_x, :c_y, :shoulder]

  def row(logits, target) do
    button_losses =
      Enum.zip(logits.buttons, target.buttons)
      |> Enum.map(fn {logit, label} ->
        max(logit, 0) - logit * label + :math.log(1 + :math.exp(-abs(logit)))
      end)

    predicted_buttons = Enum.map(logits.buttons, &if(&1 >= 0, do: 1, else: 0))
    button_matches = Enum.zip_with(predicted_buttons, target.buttons, &Kernel.==/2)

    categorical =
      Map.new(@heads, fn head ->
        values = Map.fetch!(logits, head)
        maximum = Enum.max(values)
        log_partition = maximum + :math.log(Enum.sum(Enum.map(values, &:math.exp(&1 - maximum))))
        prediction = values |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)

        {head,
         %{
           correct: prediction == target[head],
           predicted: prediction,
           nll: log_partition - Enum.at(values, target[head])
         }}
      end)

    nll =
      Enum.sum(button_losses) + Enum.sum(Enum.map(categorical, fn {_, value} -> value.nll end))

    %{
      nll: nll,
      target_probability: :math.exp(-nll),
      buttons_correct: Enum.all?(button_matches),
      b_correct: Enum.at(button_matches, 1),
      x_correct: Enum.at(button_matches, 2),
      tf_argmax_correct:
        Enum.all?(button_matches) and Enum.all?(categorical, fn {_, value} -> value.correct end),
      target: target,
      predicted_buttons: predicted_buttons,
      button_probabilities:
        Enum.map(logits.buttons, fn logit ->
          if logit >= 0,
            do: 1 / (1 + :math.exp(-logit)),
            else: :math.exp(logit) / (1 + :math.exp(logit))
        end),
      categorical: categorical
    }
  end

  def summarize([]), do: %{count: 0}

  def summarize(rows) do
    count = length(rows)

    rates =
      Map.new([:buttons_correct, :b_correct, :x_correct, :tf_argmax_correct], fn key ->
        {key, Enum.count(rows, &Map.fetch!(&1, key)) / count}
      end)

    Map.merge(rates, %{
      count: count,
      mean_joint_nll: Enum.sum(Enum.map(rows, & &1.nll)) / count,
      mean_target_probability: Enum.sum(Enum.map(rows, & &1.target_probability)) / count,
      categorical_accuracy:
        Map.new(@heads, fn head ->
          {head, Enum.count(rows, & &1.categorical[head].correct) / count}
        end)
    })
  end
end
