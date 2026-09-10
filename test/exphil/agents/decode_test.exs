defmodule ExPhil.Agents.DecodeTest do
  @moduledoc "INVARIANTS.md item 7: one decode builder, one style resolver."
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Decode

  @state %{
    deterministic: false,
    temperature: 0.7,
    deterministic_buttons: nil,
    mode_of_n: 3,
    press_threshold: 0.6,
    release_threshold: 0.4,
    last_action: %{buttons: :prev_tensor}
  }

  test "state values flow through; per-call opts override; nil buttons default false" do
    o = Decode.sample_opts(@state, [])
    assert o[:temperature] == 0.7
    assert o[:mode_of_n] == 3
    assert o[:deterministic_buttons] == false
    assert o[:press_threshold] == 0.6
    assert o[:prev_buttons] == :prev_tensor

    o2 = Decode.sample_opts(@state, temperature: 1.0, mode_of_n: nil, select_n: 4, select_fn: :f)
    assert o2[:temperature] == 1.0
    assert o2[:mode_of_n] == nil
    assert o2[:select_n] == 4
    assert o2[:select_fn] == :f
  end

  test "every path gets the same keys (the drift the incremental path had)" do
    keys = Decode.sample_opts(@state, []) |> Keyword.keys() |> Enum.sort()

    assert keys ==
             Enum.sort([
               :deterministic, :temperature, :deterministic_buttons, :mode_of_n,
               :select_n, :select_fn, :press_threshold, :release_threshold, :prev_buttons
             ])
  end

  test "resolve_style_id: explicit id wins; tag resolves via registry; unknown -> 0" do
    dir = Path.join(System.tmp_dir!(), "exphil_decode_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    path = Path.join(dir, "players.json")
    File.write!(path, Jason.encode!(%{"version" => 1, "max_players" => 4, "players" => ["MS", "INFP"], "unknown_strategy" => "nil"}))

    assert Decode.resolve_style_id(style_id: 7, style_tag: "INFP", player_registry: path) == 7
    assert Decode.resolve_style_id(style_tag: "INFP", player_registry: path) == 1
    # registry path may come from the agent's stored field (reconfigure)
    assert Decode.resolve_style_id([style_tag: "MS"], path) == 0 or Decode.resolve_style_id([style_tag: "MS"], path) == 0
    assert Decode.resolve_style_id(style_tag: "NOPE", player_registry: path) == 0
    assert Decode.resolve_style_id([]) == 0
  end
end

defmodule ExPhil.Agents.DecodeStructTest do
  @moduledoc "Item 7 phase B: the typed struct is the contract."
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Decode

  @fields %{deterministic: false, temperature: 0.8, deterministic_buttons: nil, mode_of_n: 2,
            press_threshold: 0.6, release_threshold: 0.4, last_action: %{buttons: :pb}}

  test "from_state builds a validated struct; opts derives from it" do
    d = Decode.from_state(@fields)
    assert %Decode{temperature: 0.8, mode_of_n: 2, deterministic_buttons: false} = d
    o = Decode.opts(d, @fields, select_n: 3)
    assert o[:select_n] == 3 and o[:prev_buttons] == :pb and o[:temperature] == 0.8
  end

  test "state.decode wins over loose fields once built (the point of the struct)" do
    d = Decode.from_state(@fields)
    # loose field drifts; the struct is what decisions read
    stale = Map.merge(@fields, %{temperature: 5.0, decode: d})
    assert Decode.sample_opts(stale, [])[:temperature] == 0.8
  end

  test "invalid configurations fail at construction" do
    assert_raise ArgumentError, fn -> Decode.from_state(%{@fields | temperature: 0}) end
    assert_raise ArgumentError, fn -> Decode.from_state(%{@fields | press_threshold: 0.3, release_threshold: 0.5}) end
    assert_raise ArgumentError, fn -> Decode.from_state(%{@fields | mode_of_n: 0}) end
  end
end
