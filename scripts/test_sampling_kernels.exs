# Bare-VM targeted tests, through run_recurrent_validation.exs.
ExUnit.start(exclude: [:slow, :integration, :external], timeout: 180_000)

for file <- [
      "test/exphil/networks/policy/sampling_rng_parity_test.exs",
      "test/exphil/networks/policy/autoregressive_head_test.exs",
      "test/exphil/networks/policy/sampling_mode_of_n_test.exs",
      "test/exphil/networks/policy/button_events_test.exs",
      "test/exphil/networks/policy/event_context_test.exs",
      "test/exphil/networks/policy/stick_duration_test.exs"
    ] do
  Code.require_file(file)
end
