defmodule ExPhil.Eval.MultishineBenchmarkTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.MultishineBenchmark, as: Benchmark

  defp rows(actions) do
    Enum.with_index(actions, fn action, index ->
      %{
        frame: index,
        action: action,
        stock: 4,
        hitstun: if(action == 75, do: 3, else: 0),
        grounded: action not in [25, 365, 75]
      }
    end)
  end

  test "reentry requires a full cycle, excluding hitstun from readiness timing" do
    report = Benchmark.score(rows([360, 75, 75, 14, 360, 24, 365, 360]))
    assert report.completed_cycles == 1
    assert report.recovery.completed == 1
    assert report.recovery.ready_to_cycle_frames == [4]
    assert report.recovery.disruption_to_cycle_frames == [6]
    assert hd(report.recovery.episodes).cause == :hitstun
    assert report.self_initiated_onsets == 0
    assert report.recent_hit_onsets == 2
  end

  test "isolated shine never counts as recovered" do
    report = Benchmark.score(rows([75, 14, 360]))
    assert report.recovery.completed == 0
    assert report.recovery.censored == 1
    assert hd(report.recovery.episodes).outcome == :end_of_replay
  end

  test "gaps cannot join chains or recovery episodes" do
    source = rows([75, 14, 360, 24, 365, 360])
    source = List.update_at(source, 5, &%{&1 | frame: 99})
    report = Benchmark.score(source)
    assert report.completed_cycles == 0
    assert hd(report.recovery.episodes).outcome == :frame_gap
  end

  test "stock loss and repeat interruption censor rather than disappear" do
    source = rows([75, 14, 75, 14, 0]) |> List.update_at(4, &%{&1 | stock: 3})
    report = Benchmark.score(source)
    assert report.deaths == 1
    assert Enum.map(report.recovery.episodes, & &1.outcome) == [:interrupted_again, :death]
  end

  test "loop breaks and empty hops are distinct from completed cycles" do
    report = Benchmark.score(rows([360, 24, 25, 14, 360, 24, 365, 360]))
    assert report.endings.empty_hop == 1
    assert report.recovery.completed == 1
    assert hd(report.recovery.episodes).cause == :loop_break
  end

  test "empty replay fails rather than reporting a successful zero" do
    assert_raise ArgumentError, fn -> Benchmark.score([]) end
  end

  test "jumpsquat without an observed aerial shine is not recovery" do
    report = Benchmark.score(rows([75, 14, 360, 24, 360]))
    assert report.completed_cycles == 0
    assert report.recovery.completed == 0
  end

  test "teacher audit exposes fallback labels during hitstun and skips gaps" do
    neutral = ExPhil.Bridge.ControllerState.neutral()

    source =
      rows([14, 14])
      |> Enum.map(fn row ->
        Map.put(row, :player, %{action: 14, action_frame: 0, on_ground: true, controller: neutral})
      end)
      |> List.update_at(0, &%{&1 | hitstun: 4})

    expert = %ExPhil.Agents.MultishineExpert{table: %{}}
    audit = Benchmark.teacher_audit(source, expert)
    assert audit.counts == %{fallback: 1}
    assert audit.disagreement_count == 1
    assert audit.fallback_during_hitstun == 1

    assert Benchmark.teacher_audit(List.update_at(source, 1, &%{&1 | frame: 50}), expert).counts ==
             %{}
  end

  @tag :nif
  test "real technique fixture scores above the established chain floor and audits the teacher" do
    path = "test/fixtures/replays/fox_multishine_closed_d1.slp"
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)
    rows = Benchmark.rows(replay, 1)
    report = Benchmark.score(rows)
    assert report.max_chain >= 50
    assert report.completed_cycles >= 50
    expert = ExPhil.Agents.MultishineExpert.from_fixture(path)
    audit = Benchmark.teacher_audit(rows, expert)
    assert audit.counts.table > 100
    assert is_integer(audit.disagreement_count)
  end

  @tag :nif
  @tag :tmp_dir
  test "frozen manifest rejects mutated content and duplicate recordings", %{tmp_dir: directory} do
    fixture = Path.expand("test/fixtures/replays/fox_multishine_closed_d1.slp")
    policy = Path.join(directory, "policy.bin")
    File.write!(policy, "test policy identity")
    hash = fn path -> :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower) end

    run = %{
      id: "reference-stand-1",
      scenario: "stand",
      port: 1,
      policy: policy,
      replay: fixture,
      policy_sha256: hash.(policy),
      replay_sha256: hash.(fixture)
    }

    manifest = %{
      version: 1,
      teacher_fixture: fixture,
      teacher_fixture_sha256: hash.(fixture),
      protocol: %{
        runner: "async",
        frame_delay: 3,
        delay_id: 2,
        temperature: 1.0,
        buttons_temperature: 1.0,
        stage: "FD",
        seconds: 60
      },
      runs: [run]
    }

    path = Path.join(directory, "manifest.json")
    File.write!(path, Jason.encode!(manifest))
    assert length(Benchmark.run_manifest(path).runs) == 1
    File.write!(path, Jason.encode!(%{manifest | runs: [run, %{run | id: "duplicate"}]}))
    assert_raise ArgumentError, ~r/Duplicate replay/, fn -> Benchmark.run_manifest(path) end
    File.write!(policy, "changed")
    assert_raise ArgumentError, ~r/frozen manifest/, fn -> Benchmark.run_manifest(path) end
  end
end
