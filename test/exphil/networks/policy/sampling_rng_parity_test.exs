# Isolate the optimization against the same sampler using Nx's reference RNG.
# Keeping all decode branches identical makes changes in random bits/key order
# observable at the public action API, including conditional AR logits.
File.read!("lib/exphil/networks/policy/sampling.ex")
|> String.replace(
  "defmodule ExPhil.Networks.Policy.Sampling do",
  "defmodule ExPhil.Networks.Policy.RolledSamplingReference do"
)
|> String.replace("  alias Edifice.Utils.SamplingRandom\n", "")
|> String.replace("SamplingRandom.uniform(", "Nx.Random.uniform(")
|> Code.compile_string("sampling_rolled_reference_test.ex")

defmodule ExPhil.Networks.Policy.SamplingRngParityTest do
  use ExUnit.Case, async: false
  alias ExPhil.Networks.Policy.{Heads, RolledSamplingReference, Sampling}

  defp assert_action(actual, expected) do
    for field <- [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder] do
      assert Nx.to_binary(actual[field]) == Nx.to_binary(expected[field]), "#{field} differs"
    end

    if actual[:stick_commit],
      do: assert(Nx.to_binary(actual.stick_commit) == Nx.to_binary(expected.stick_commit))

    for {field, value} <- actual.logits do
      assert Nx.to_number(Nx.all_close(value, expected.logits[field], atol: 1.0e-5, rtol: 1.0e-5)) ==
               1
    end
  end

  test "independent samples, temperatures and mode-of-N retain seeded actions" do
    for rows <- [1, 8] do
      {logits, _} = Nx.Random.normal(Nx.Random.key(93), shape: {rows, 17})

      outputs =
        {Nx.slice_along_axis(logits, 0, 8, axis: 1), logits, logits, logits, logits,
         Nx.slice_along_axis(logits, 0, 5, axis: 1)}

      predict = fn _, _ -> outputs end

      modes =
        [
          [],
          [deterministic_buttons: true],
          [temperature: %{buttons: 0.7, main: 1.3, c: 0.8, shoulder: 1.1}]
        ] ++
          if(rows == 1, do: [[mode_of_n: 4]], else: [])

      for mode <- modes, seed <- 0..7 do
        opts = Keyword.put(mode, :key, Nx.Random.key(seed))

        assert_action(
          Sampling.sample(%{}, predict, Nx.tensor(0), opts),
          RolledSamplingReference.sample(%{}, predict, Nx.tensor(0), opts)
        )
      end
    end
  end

  for events <- [false, true] do
    test "AR seeded parity with events/duration=#{events}" do
      for rows <- [1, 8] do
        prev_b = Nx.broadcast(0.0, {rows, 8})
        prev_s = Nx.broadcast(Nx.tensor(2, type: :s64), {rows, 4})
        age = Nx.broadcast(Nx.tensor(3, type: :s64), {rows})

        extra =
          if unquote(events),
            do: [
              button_events_prev: Axon.input("prev_buttons", shape: {nil, 8}),
              stick_events_prev: Axon.input("prev_sticks", shape: {nil, 4}),
              event_context: true,
              stick_release: true,
              stick_duration: 5
            ],
            else: []

        model =
          Heads.build_autoregressive_head(
            Axon.input("trunk", shape: {nil, 8}),
            [axis_buckets: 16, shoulder_buckets: 4, residual_size: 8, component_hidden: 8] ++
              extra
          )

        {init, _} = Axon.build(model)
        inputs = Map.merge(Heads.tf_templates(rows), %{"trunk" => Nx.template({rows, 8}, :f32)})

        inputs =
          if unquote(events),
            do:
              Map.merge(inputs, %{
                "prev_buttons" => prev_b,
                "prev_sticks" => prev_s,
                "prev_age" => age
              }),
            else: inputs

        params = init.(inputs, Axon.ModelState.empty())
        # Exercise actual conditional dependencies, not the zero-init embedding special case.
        data =
          Map.new(params.data, fn {name, layer} ->
            if String.ends_with?(name, "_embed") do
              {k, _} =
                Nx.Random.normal(Nx.Random.key(:erlang.phash2(name)),
                  shape: Nx.shape(layer["kernel"])
                )

              {name, Map.put(layer, "kernel", Nx.multiply(k, 0.1))}
            else
              {name, layer}
            end
          end)

        params = %{params | data: data}
        {features, _} = Nx.Random.normal(Nx.Random.key(18), shape: {rows, 8})

        context =
          if unquote(events),
            do: [
              event_prev_buttons: prev_b,
              event_prev_sticks: prev_s,
              event_prev_age: age,
              stick_commit: Nx.remainder(Nx.iota({rows}, type: :s64), 3)
            ],
            else: []

        modes =
          [
            [],
            [
              deterministic_buttons: true,
              press_threshold: 0.6,
              release_threshold: 0.4,
              prev_buttons: prev_b
            ]
          ] ++
            if(rows == 1, do: [[mode_of_n: 4]], else: [])

        for mode <- modes, seed <- 0..7 do
          opts = [key: Nx.Random.key(seed), temperature: 0.8] ++ mode ++ context

          assert_action(
            Sampling.sample_autoregressive_from_features(params, features, opts),
            RolledSamplingReference.sample_autoregressive_from_features(params, features, opts)
          )
        end
      end
    end
  end
end
