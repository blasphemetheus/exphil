defmodule ExPhil.Bridge.SimBatchTest do
  @moduledoc "NIF backend vs Python worker backend: same frames, same terminals (mix test --include external)."
  use ExUnit.Case, async: false
  alias ExPhil.Sim.{Drill, Env}
  @moduletag :external

  @opts [stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], batch_size: 1, length: 64, seed: 3]

  test "reset and 60 scripted steps are identical across backends; save/restore by id" do
    {:ok, port} = Env.start(:port, @opts)
    {:ok, nif} = Env.start(:nif, @opts)
    {:ok, [p]} = Env.frames(port)
    {:ok, [n]} = Env.frames(nif)
    assert p == n
    neutral = Drill.neutral()
    dash = %{neutral | main_stick: %{x: 1.0, y: 0.5}}
    jump = %{neutral | button_y: true}

    for t <- 1..60 do
      c = [[if(t < 30, do: dash, else: jump), neutral]]
      {:ok, [a], [ta]} = Env.step(port, c)
      {:ok, [b], [tb]} = Env.step(nif, c)
      assert a == b and ta == tb, "diverged at step #{t}"
    end

    {:ok, blob, id} = Env.save(nif, 0, keep: true)
    {:ok, _, _} = Env.step(nif)
    {:ok, [r]} = Env.restore(nif, 0, {:id, id})
    {:ok, [r2]} = Env.restore(nif, 0, blob)
    assert r.frame == r2.frame and byte_size(blob) > 100_000
    Env.stop(port)
    Env.stop(nif)
  end

  test "batch of 8 steps independently" do
    {:ok, nif} = Env.start(:nif, Keyword.put(@opts, :batch_size, 8))
    neutral = Drill.neutral()
    ctrl = for i <- 0..7, do: [%{neutral | main_stick: %{x: (if rem(i, 2) == 0, do: 1.0, else: 0.0), y: 0.5}}, neutral]
    for _ <- 1..150, do: {:ok, _, _} = Env.step(nif, ctrl)
    {:ok, frames} = Env.frames(nif)
    xs = Enum.map(frames, & &1.players[1].x)
    assert length(frames) == 8 and Enum.at(xs, 0) > Enum.at(xs, 1)
    Env.stop(nif)
  end
end
