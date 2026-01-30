defmodule ExPhil.CLITest do
  use ExUnit.Case, async: true

  alias ExPhil.CLI

  describe "parse_args/2" do
    test "parses verbosity flags" do
      opts = CLI.parse_args(["--quiet"], flags: [:verbosity])
      assert opts[:quiet] == true
      assert opts[:verbose] == false

      opts = CLI.parse_args(["--verbose"], flags: [:verbosity])
      assert opts[:quiet] == false
      assert opts[:verbose] == true

      opts = CLI.parse_args(["-q"], flags: [:verbosity])
      assert opts[:quiet] == true
    end

    test "parses replay flags" do
      opts = CLI.parse_args(
        ["--replays", "/path/to/replays", "--max-files", "100"],
        flags: [:replay]
      )

      assert opts[:replays] == "/path/to/replays"
      assert opts[:max_files] == 100
    end

    test "parses checkpoint flags" do
      opts = CLI.parse_args(
        ["--checkpoint", "model.axon", "--policy", "policy.bin"],
        flags: [:checkpoint]
      )

      assert opts[:checkpoint] == "model.axon"
      assert opts[:policy] == "policy.bin"
    end

    test "uses defaults when flags not provided" do
      opts = CLI.parse_args([], flags: [:replay, :training])

      assert opts[:replays] == "./replays"
      assert opts[:batch_size] == 64
      assert opts[:epochs] == 10
    end

    test "allows default overrides" do
      opts = CLI.parse_args(
        [],
        flags: [:training],
        defaults: [batch_size: 512, epochs: 100]
      )

      assert opts[:batch_size] == 512
      assert opts[:epochs] == 100
    end

    test "parses extra script-specific flags" do
      opts = CLI.parse_args(
        ["--temporal", "--backbone", "mamba"],
        flags: [:verbosity],
        extra: [temporal: :boolean, backbone: :string]
      )

      assert opts[:temporal] == true
      assert opts[:backbone] == "mamba"
    end

    test "captures positional arguments" do
      opts = CLI.parse_args(
        ["--quiet", "file1.slp", "file2.slp"],
        flags: [:verbosity]
      )

      assert opts[:_positional] == ["file1.slp", "file2.slp"]
    end

    test "parses short aliases" do
      opts = CLI.parse_args(
        ["-r", "/replays", "-b", "256"],
        flags: [:replay, :training]
      )

      assert opts[:replays] == "/replays"
      assert opts[:batch_size] == 256
    end
  end

  describe "parse_raw/2" do
    test "parses string args" do
      args = ["--replays", "/path/to/replays"]
      opts = CLI.parse_raw(args, [])

      assert opts[:replays] == "/path/to/replays"
    end

    test "parses integer args" do
      args = ["--batch-size", "512", "--max-files", "50"]
      opts = CLI.parse_raw(args, [])

      assert opts[:batch_size] == 512
      assert opts[:max_files] == 50
    end

    test "parses float args with scientific notation" do
      args = ["--lr", "3e-4"]
      opts = CLI.parse_raw(args, [])

      assert_in_delta opts[:learning_rate], 0.0003, 0.00001
    end

    test "parses boolean flags" do
      args = ["--quiet", "--detailed"]
      opts = CLI.parse_raw(args, [])

      assert opts[:quiet] == true
      assert opts[:detailed] == true
    end

    test "preserves defaults for missing args" do
      args = ["--quiet"]
      opts = CLI.parse_raw(args, [batch_size: 64, replays: "./replays"])

      assert opts[:batch_size] == 64
      assert opts[:replays] == "./replays"
      assert opts[:quiet] == true
    end
  end

  describe "parsing helpers" do
    test "has_flag?/2 detects flag presence" do
      assert CLI.has_flag?(["--quiet", "--verbose"], "--quiet")
      refute CLI.has_flag?(["--verbose"], "--quiet")
    end

    test "get_arg_value/2 gets value after flag" do
      args = ["--replays", "/path", "--batch-size", "512"]

      assert CLI.get_arg_value(args, "--replays") == "/path"
      assert CLI.get_arg_value(args, "--batch-size") == "512"
      assert CLI.get_arg_value(args, "--missing") == nil
    end

    test "get_arg_value/2 returns nil for flag at end" do
      args = ["--replays"]
      assert CLI.get_arg_value(args, "--replays") == nil
    end

    test "get_arg_value/2 returns nil when next arg is a flag" do
      args = ["--quiet", "--verbose"]
      assert CLI.get_arg_value(args, "--quiet") == nil
    end

    test "parse_int_list_arg/4 parses comma-separated integers" do
      args = ["--hidden-sizes", "512,256,128"]
      opts = CLI.parse_int_list_arg([], args, "--hidden-sizes", :hidden_sizes)

      assert opts[:hidden_sizes] == [512, 256, 128]
    end

    test "parse_atom_list_arg/4 parses comma-separated atoms" do
      args = ["--characters", "mewtwo,ganondorf"]
      opts = CLI.parse_atom_list_arg([], args, "--characters", :characters)

      assert opts[:characters] == [:mewtwo, :ganondorf]
    end
  end

  describe "verbosity_level/1" do
    test "returns 0 for quiet" do
      assert CLI.verbosity_level(quiet: true) == 0
    end

    test "returns 2 for verbose" do
      assert CLI.verbosity_level(verbose: true) == 2
    end

    test "returns 1 for default" do
      assert CLI.verbosity_level([]) == 1
    end
  end

  describe "flags_for_groups/1" do
    test "returns flags for specified groups" do
      flags = CLI.flags_for_groups([:verbosity])

      names = Enum.map(flags, & &1.name)
      assert :quiet in names
      assert :verbose in names
      refute :replays in names
    end

    test "returns flags for multiple groups" do
      flags = CLI.flags_for_groups([:verbosity, :replay])

      names = Enum.map(flags, & &1.name)
      assert :quiet in names
      assert :replays in names
      assert :max_files in names
    end
  end

  describe "help_text/1" do
    test "generates help text for flag groups" do
      help = CLI.help_text([:verbosity])

      assert help =~ "--quiet"
      assert help =~ "--verbose"
      assert help =~ "Suppress non-essential output"
    end

    test "includes short aliases" do
      help = CLI.help_text([:verbosity])

      assert help =~ "-q"
      assert help =~ "-v"
    end

    test "includes defaults" do
      help = CLI.help_text([:replay])

      assert help =~ "./replays"
    end
  end

  describe "require_options!/2" do
    test "passes when all required options present" do
      opts = [checkpoint: "model.axon", replays: "/path"]

      assert CLI.require_options!(opts, [:checkpoint, :replays]) == :ok
    end

    test "raises when required option missing" do
      opts = [replays: "/path"]

      assert_raise ArgumentError, ~r/Missing required options.*--checkpoint/, fn ->
        CLI.require_options!(opts, [:checkpoint, :replays])
      end
    end
  end

  describe "require_one_of!/2" do
    test "passes when at least one option present" do
      opts = [checkpoint: "model.axon"]

      assert CLI.require_one_of!(opts, [:checkpoint, :policy]) == :ok
    end

    test "raises when none of the options present" do
      opts = [replays: "/path"]

      assert_raise ArgumentError, ~r/Requires at least one of/, fn ->
        CLI.require_one_of!(opts, [:checkpoint, :policy])
      end
    end
  end
end
