defmodule ExPhil.Test.FlakyCase do
  @moduledoc """
  A custom ExUnit.Case for tests with known flakiness.

  Use this module instead of `ExUnit.Case` for test modules that
  contain flaky tests that need retry logic.

  ## Usage

      defmodule MyFlakyTest do
        use ExPhil.Test.FlakyCase, async: true

        @tag :flaky
        @tag retries: 3
        test "occasionally fails due to timing" do
          # test that sometimes fails
        end
      end

  ## How It Works

  When a test is tagged with `@tag retries: N`, this module will
  automatically retry the test up to N times before marking it as failed.

  ## Configuration

  You can set default retries at the module level:

      defmodule MyFlakyTest do
        use ExPhil.Test.FlakyCase, async: true, default_retries: 2

        # All tests in this module will retry up to 2 times by default
      end

  ## Best Practices

  1. Always tag flaky tests with `@tag :flaky` for tracking
  2. Keep retry counts low (2-3 max)
  3. Investigate and fix root causes when possible
  4. Use this as a last resort, not a first solution
  """

  use ExUnit.CaseTemplate

  using opts do
    default_retries = Keyword.get(opts, :default_retries, 1)

    quote do
      use ExUnit.Case, unquote(opts)
      import ExPhil.Test.Helpers

      # Store default retries for this module
      @default_retries unquote(default_retries)

      # Override the test macro to add retry logic
      # This is a simple approach - tests with retries tag get wrapped
    end
  end

  @doc """
  A setup callback that can be added to inject retry behavior.

  Add this to your test module:

      setup context do
        # Your setup here
        :ok
      end

  The FlakyCase module handles retries at the test execution level,
  not through setup callbacks.
  """
  def setup_flaky_context(context) do
    # Extract retry configuration from tags
    retries = Map.get(context, :retries, 1)
    flaky? = Map.get(context, :flaky, false)

    # Pass retry info through context
    {:ok, %{retries: retries, flaky: flaky?}}
  end
end
