defmodule ExPhil.ResyncSpread do
  @moduledoc "Benchmark-only scheduling and carry validation; no deployment defaults change."

  def prepare(group, history) do
    # Rebuild every row once before assigning phases. Advancing the counters below
    # can then only make the first replay earlier, never extend stale carry.
    for agent <- group.agents do
      :sys.replace_state(agent, &set_counts(&1, group.mode, List.duplicate(30, group.rows)))
    end

    for call <- group.calls, do: call.(history)
    %{group | bases: Enum.map(group.agents, &:sys.get_state/1)}
  end

  def restore(group, schedule) do
    phases =
      for row <- 0..(group.rows - 1),
          do: if(schedule == :spread, do: div(row * 31, group.rows), else: 0)

    for {{agent, base}, row} <- Enum.with_index(Enum.zip(group.agents, group.bases)) do
      counts = if group.mode == :batch, do: phases, else: [Enum.at(phases, row)]
      :sys.replace_state(agent, fn _ -> set_counts(base, group.mode, counts) end)
    end
  end

  defp set_counts(state, :batch, counts),
    do: %{state | batch: %{state.batch | resync_steps: counts}}

  defp set_counts(state, _, [count | _]), do: %{state | steps_since_resync: count}

  def validate(groups, history) do
    actors = Enum.find(groups, &(&1.mode == :actors))
    batch = Enum.find(groups, &(&1.mode == :batch))

    for schedule <- [:aligned, :spread] do
      restore(actors, schedule)
      restore(batch, schedule)

      events =
        for tick <- 0..61 do
          before = :sys.get_state(hd(batch.agents)).batch.resync_steps
          for call <- actors.calls ++ batch.calls, do: call.(history + tick)
          b = :sys.get_state(hd(batch.agents)).batch
          expected = Enum.map(before, &if(&1 >= 30, do: 0, else: &1 + 1))
          true = b.resync_steps == expected

          errors =
            for {agent, row} <- Enum.with_index(actors.agents) do
              state = :sys.get_state(agent)
              true = state.steps_since_resync == Enum.at(expected, row)

              for {key, value} <- state.trunk_state do
                actual = Nx.slice_along_axis(b.trunk_state[key], row, 1, axis: 0)

                error =
                  actual |> Nx.subtract(value) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

                scale = value |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() |> max(1.0)
                # This trained checkpoint already differs between batch-eight
                # and batch-one in the aligned baseline. Preserve the errors;
                # do not turn a timing experiment into an unsupported parity claim.
                true = Nx.to_number(Nx.any(Nx.is_nan(actual))) == 0
                true = Nx.to_number(Nx.any(Nx.is_infinity(actual))) == 0

                %{
                  row: row,
                  key: key,
                  max_absolute_error: error,
                  tensor_scale: scale,
                  within_elementwise_tolerance:
                    Nx.to_number(Nx.all_close(actual, value, atol: 1.0e-4, rtol: 1.0e-4)) == 1
                }
              end
            end

          %{
            tick: tick,
            rows: for({0, row} <- Enum.with_index(expected), do: row),
            carry_errors: List.flatten(errors)
          }
        end

      for row <- 0..(actors.rows - 1) do
        ticks = for %{tick: tick, rows: rows} <- events, row in rows, do: tick
        [first, second] = ticks
        true = first <= 30 and second - first == 31
      end

      %{schedule: schedule, events: events, replay_cadence_validated: true}
    end
  end
end
