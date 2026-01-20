defmodule ExPhil.Test.Mocks do
  @moduledoc """
  Mox mock definitions for testing.

  This module defines mock modules for behaviours used in ExPhil.
  Mocks allow tests to isolate components by replacing dependencies
  with controlled implementations.

  ## Usage

  1. Define a behaviour for the component you want to mock
  2. Add the mock definition here
  3. In tests, use Mox.stub/3 or Mox.expect/3 to configure mock behavior

  ## Example

      defmodule MyTest do
        use ExUnit.Case, async: true
        import Mox

        setup :verify_on_exit!

        test "uses mock replay parser" do
          expect(ExPhil.ReplayParserMock, :parse, fn path ->
            {:ok, %{frames: [], metadata: %{path: path}}}
          end)

          # Code that uses the parser via Application.get_env(:exphil, :replay_parser)
        end
      end
  """

  # Define mocks for behaviours
  # Mox.defmock(ExPhil.ReplayParserMock, for: ExPhil.Data.ReplayParser.Behaviour)

  # Note: Add more mocks here as behaviours are defined in the codebase.
  # For now, Mox is available for use when behaviours are added.
end
