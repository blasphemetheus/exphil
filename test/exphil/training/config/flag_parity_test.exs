defmodule ExPhil.Training.Config.FlagParityTest do
  @moduledoc """
  INVARIANTS.md item 2: a training flag must be accepted (`@valid_flags`),
  parsed (`Parser`), and — for value flags — defaulted (`defaults/0`),
  identically. These are four hand-maintained lists today; this test is
  the drift detector until the flag table generates them.

  Found by the 2026-09-09 audit that motivated it: `--num-heads` accepted
  but never parsed (Trainer's private 2/32 table silently won over 4/64),
  `--log-file` documented but rejected, `--transition-weight` plumbed but
  flagless, seven legacy mode aliases parsed but rejected at the door.

  The parser pipeline is not introspectable at runtime, so this reads
  `parser.ex` as text — a deliberate trade: a regex over source is still
  one place that fails loudly on drift.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config

  alias ExPhil.Training.Config.Parser

  # Phase B (2026-09-09): the parser is TABLE-DRIVEN and Config's
  # @valid_flags is derived from Parser.flags/0, so accepted <=> parsed is
  # structural. The tests below now pin the remaining hand-maintained
  # lists (defaults, docs) against the table, plus keep the two-way check
  # as a tripwire in case someone re-introduces a hand list.
  @meta_flags ~w(--preset --config --verbose --quiet)

  defp parser_flags do
    table = Enum.map(Parser.flag_table(), fn {flag, key, _type} -> {flag, key} end)
    special = Parser.flags() |> Enum.reject(fn f -> f in Enum.map(table, &elem(&1, 0)) end) |> Enum.map(&{&1, nil})
    Enum.uniq_by(table ++ special, &elem(&1, 0))
  end

  test "every @valid_flags entry has a parser line (or is a known meta flag)" do
    parsed = parser_flags() |> Enum.map(&elem(&1, 0)) |> MapSet.new()

    orphans =
      Config.valid_flags()
      |> Enum.reject(&(&1 in @meta_flags))
      |> Enum.reject(&MapSet.member?(parsed, &1))

    assert orphans == [],
           "accepted-and-IGNORED flags (in @valid_flags, no parser line): #{inspect(orphans)}"
  end

  test "every parsed flag is in @valid_flags (otherwise it is rejected at the door)" do
    valid = MapSet.new(Config.valid_flags())
    rejected = parser_flags() |> Enum.map(&elem(&1, 0)) |> Enum.reject(&MapSet.member?(valid, &1))

    assert rejected == [],
           "parsed but NOT in @valid_flags (train.exs aborts on them): #{inspect(rejected)}"
  end

  test "every parsed value key has a defaults/0 entry (negation flags exempt)" do
    defaults = Config.defaults() |> Keyword.keys() |> MapSet.new()

    missing =
      parser_flags()
      |> Enum.map(&elem(&1, 1))
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()
      # --no-X flags set a :no_x key that is folded into :x downstream
      |> Enum.reject(&String.starts_with?(Atom.to_string(&1), "no_"))
      # explicitly nil-by-absence keys, audited 2026-09-09
      |> Enum.reject(&(&1 in [:fail_fast, :hide_errors, :attention_every]))
      |> Enum.reject(&MapSet.member?(defaults, &1))

    assert missing == [], "parsed keys with no defaults/0 entry: #{inspect(missing)}"
  end

  test "the documented train.exs flags are all valid" do
    md = File.read!("docs/guides/TRAINING.md")

    # Scope to the train.exs sections: rows whose flag is not one of the
    # other scripts' (PPO / drill / eval) tables. Those tables document
    # flags of scripts with their own parsers; this test only pins the
    # Config-owned surface. Keep this list honest — it is the allowlist.
    other_scripts =
      ~w(--player-port --hidden --solver --integration-steps --expand-ratio
         --attention-interval --multi-delay --pipeline-offset --queue-depth
         --with-delay-id --snippet-frames --probe-basin --probe-entries
         --reject-at --select-by --num-games --game-type --max-episode-frames
         --track-elo --ppo-epochs --clip-epsilon --gae-lambda)

    valid = MapSet.new(Config.valid_flags())

    phantom =
      Regex.scan(~r/^\| `(--[\w-]+)/m, md)
      |> Enum.map(fn [_, f] -> f end)
      |> Enum.uniq()
      |> Enum.reject(&(&1 in other_scripts))
      |> Enum.reject(&MapSet.member?(valid, &1))

    assert phantom == [], "documented in TRAINING.md but rejected by train.exs: #{inspect(phantom)}"
  end

  test "--num-heads / --head-dim parse and default to the documented 4/64" do
    opts = Config.parse_args(["--num-heads", "8", "--head-dim", "32", "--backbone", "gru"])
    assert opts[:num_heads] == 8
    assert opts[:head_dim] == 32
    assert Config.defaults()[:num_heads] == 4
    assert Config.defaults()[:head_dim] == 64
  end
end
