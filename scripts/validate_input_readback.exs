alias ExPhil.Bridge.ControllerState
alias ExPhil.Eval.ScenarioInputTiming

{opts, _, []} =
  OptionParser.parse(System.argv(), strict: [input: :string, profile: :string, out: :string])

profile =
  case Keyword.fetch!(opts, :profile) do
    "pipe_v1" -> :pipe_v1
    "pipe_v2" -> :pipe_v2
  end

data = opts |> Keyword.fetch!(:input) |> File.read!() |> Jason.decode!(keys: :atoms)
buttons = [:a, :b, :x, :y, :z, :l, :r, :d_up]

inputs =
  Map.new(data.rows, fn row ->
    recorded = row.recorded

    input = %{
      main_stick: %{x: recorded.main_stick_x, y: recorded.main_stick_y},
      c_stick: %{x: recorded.c_stick_x, y: recorded.c_stick_y},
      shoulder: recorded.l_trigger,
      buttons:
        Map.new(buttons, &{&1, Map.fetch!(recorded, String.to_existing_atom("button_#{&1}"))})
    }

    controller = %{ControllerState.from_input(input) | r_shoulder: recorded.r_trigger}
    {row.frame, controller}
  end)

trace = Enum.map(data.rows, &Map.take(&1, [:frame, :sent, :issued]))
result = ScenarioInputTiming.verify(trace, inputs, 0, profile: profile)
File.write!(Keyword.fetch!(opts, :out), Jason.encode!(result, pretty: true), [:exclusive])
IO.inspect(result)
unless result.valid, do: System.halt(1)
