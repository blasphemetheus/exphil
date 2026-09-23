defmodule ExPhil.Sim.RecoveryProbe do
  @moduledoc """
  TAS a recovery: from a savestate, run a grid of scripted recoveries for one
  port in a batch and report which make it back (ledge grab, or grounded
  outside the dead actions, stock kept). The opponent holds neutral. This is
  the ground truth `ExPhil.Melee.Checkmate` is calibrated against.

  Grid (~527 plans): drift only; optional double jump then Fire Fox (8
  timings x 10 angles); Illusion (5 timings); air dodge (4 timings x 5
  angles); shine stalls (1/2/4 shines) then Fire Fox.
  """
  alias ExPhil.Sim.{Drill, Env}

  @ledge_actions [252, 253]
  @dead_actions Enum.to_list(0..12)

  @doc "`[{name, events}]` for a fighter that must travel in `toward` (+1 right, -1 left)."
  def plans(toward) do
    n = Drill.neutral()
    stick = fn deg -> r = deg * :math.pi() / 180; %{x: 0.5 + 0.5 * toward * :math.cos(r), y: 0.5 + 0.5 * :math.sin(r)} end
    drift = %{n | main_stick: stick.(0)}
    hold = fn from, k, c -> for t <- from..(from + k - 1), do: {t, c} end
    firefox = fn at, deg -> [{at, %{n | button_b: true, main_stick: %{x: 0.5, y: 1.0}}}] ++ hold.(at + 1, 50, %{n | main_stick: stick.(deg)}) end
    illusion = fn at -> [{at, %{n | button_b: true, main_stick: stick.(0)}}] end
    dodge = fn at, deg -> [{at, %{n | button_l: true, l_shoulder: 1.0, main_stick: stick.(deg)}}] end
    jump = fn at -> [{at, %{drift | button_y: true}}] end
    shines = fn at, k -> for i <- 0..(k - 1), do: {at + 6 * i, %{n | button_b: true, main_stick: %{x: 0.5, y: 0.0}}} end
    j_ev = fn j -> if j, do: jump.(j), else: [] end

    plans =
      [{"drift only", []}] ++
        for(j <- [nil, 0, 4, 8, 16, 30], u <- [0, 10, 20, 30, 40, 50, 60, 70], deg <- [-22.5, 0, 11.25, 22.5, 33.75, 45, 56.25, 67.5, 78.75, 90], j == nil or u > j,
          do: {"jump #{inspect(j)} firefox #{u} @#{deg}", j_ev.(j) ++ firefox.(u, deg)}) ++
        for(j <- [nil, 0, 8, 16, 30], s <- [0, 10, 20, 30, 45], j == nil or s > j, do: {"jump #{inspect(j)} illusion #{s}", j_ev.(j) ++ illusion.(s)}) ++
        for(j <- [nil, 0, 8, 16], d <- [2, 10, 20, 30], deg <- [0, 22.5, 45, 67.5, 90], j == nil or d > j, do: {"jump #{inspect(j)} dodge #{d} @#{deg}", j_ev.(j) ++ dodge.(d, deg)}) ++
        for(j <- [nil, 0], s <- [0, 10], k <- [1, 2, 4], u <- [30, 50], deg <- [22.5, 45, 67.5], j == nil or s > j,
          do: {"jump #{inspect(j)} shine x#{k} @#{s} firefox +#{u} @#{deg}", j_ev.(j) ++ shines.(s, k) ++ firefox.(s + 6 * k + u - 30, deg)})

    {plans, drift}
  end

  @doc """
  Run every plan from savestate `ref` (a blob or `{:id, id}` already uploaded)
  on `sim` (batch of `batch` envs). `p0` is the port's player state at the
  start. Returns `[{name, outcome}]`, outcome `{:ledge, t} | {:landed, t, x, y} | {:died, t} | nil`.
  """
  def run(sim, ref, port, p0, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, 240)
    batch = Keyword.fetch!(opts, :batch)
    n = Drill.neutral()
    toward = if p0.x < 0, do: 1.0, else: -1.0
    {plans, drift} = plans(toward)
    program = fn events -> ev = Map.new(events); List.to_tuple(for t <- 0..(horizon - 1), do: Map.get(ev, t, drift)) end
    idle = program.([])

    plans
    |> Enum.chunk_every(batch)
    |> Enum.flat_map(fn chunk ->
      for i <- 0..(batch - 1), do: {:ok, _} = Env.restore(sim, i, ref, frames: false)
      progs = Enum.map(chunk, fn {_, ev} -> program.(ev) end) ++ List.duplicate(idle, batch - length(chunk))

      outcome =
        Enum.reduce(0..(horizon - 1), Enum.map(chunk, fn _ -> nil end), fn t, acc ->
          ctrls = Enum.map(progs, fn p -> c = elem(p, t); if port == 1, do: [c, n], else: [n, c] end)
          {:ok, states, _} = Env.step(sim, ctrls)

          Enum.zip_with(acc, states, fn
            nil, gs ->
              p = gs.players[port]
              cond do
                p.stock < p0.stock -> {:died, t}
                p.action in @ledge_actions -> {:ledge, t}
                p.on_ground and p.action not in @dead_actions -> {:landed, t, Float.round(p.x * 1.0, 1), Float.round(p.y * 1.0, 1)}
                true -> nil
              end
            done, _ -> done
          end)
        end)

      Enum.zip(Enum.map(chunk, &elem(&1, 0)), outcome)
    end)
  end

  def made_it?({:ledge, _}), do: true
  def made_it?({:landed, _, _, _}), do: true
  def made_it?(_), do: false
end
