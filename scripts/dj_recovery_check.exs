# DJ-recovery check for the steering A/B (gates --steer-alpha 1.0 as a live
# default): does full projection removal of the shield direction ALSO kill
# recovery double jumps? (The alpha=1.0 probe showed DJ-per-air-stint 0.0%;
# neutral DJs were pathological, but offstage DJs are load-bearing.)
#
#   mix run scripts/dj_recovery_check.exs <label:replay.slp> [more ...]
#
# e.g. mix run scripts/dj_recovery_check.exs \
#        baseline:probes/steer_ab/baseline/Game_x.slp \
#        alpha10:probes/steer_ab/alpha10/Game_y.slp
#
# Per replay (P1 = the policy): offstage recovery situations = contiguous
# airborne stretches with |x| > 88 or y < -6 (gaps <= 10f merged, >= 15f
# long). For each: DJ used (action 27/28 JumpAerialF/B inside the stretch),
# outcome (death = action <= 0x0A or stock loss during/within 90f after;
# otherwise returned). NO-MIX: one beam.

alias ExPhil.Data.Peppi

edge_x = 88.0
gap_merge = 10
min_len = 15
post_window = 90

args = System.argv()
if args == [], do: raise("usage: mix run scripts/dj_recovery_check.exs label:replay.slp ...")

for arg <- args do
  {label, path} =
    case String.split(arg, ":", parts: 2) do
      [l, p] -> {l, p}
      [p] -> {Path.basename(p), p}
    end

  {:ok, replay} = Peppi.parse(path)

  p1 =
    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(& &1.game_state.players[1])

  n = length(p1)

  offstage? = fn pl ->
    not (pl.on_ground || false) and (abs(pl.x || 0.0) > edge_x or (pl.y || 0.0) < -6.0)
  end

  # Indices of offstage frames -> merged stretches
  off_idx =
    p1
    |> Enum.with_index()
    |> Enum.filter(fn {pl, _i} -> offstage?.(pl) end)
    |> Enum.map(&elem(&1, 1))

  stretches =
    off_idx
    |> Enum.reduce([], fn i, acc ->
      case acc do
        [{s, e} | rest] when i - e <= gap_merge -> [{s, i} | rest]
        _ -> [{i, i} | acc]
      end
    end)
    |> Enum.reverse()
    |> Enum.filter(fn {s, e} -> e - s + 1 >= min_len end)

  frames_arr = List.to_tuple(p1)

  results =
    Enum.map(stretches, fn {s, e} ->
      seg = for i <- s..e, do: elem(frames_arr, i)

      dj_frames = Enum.count(seg, &(&1.action in [27, 28]))
      dj_used = dj_frames > 0

      stock_start = (elem(frames_arr, s)).stock
      post_end = min(e + post_window, n - 1)
      post = for i <- e..post_end, do: elem(frames_arr, i)

      died =
        Enum.any?(seg ++ post, fn pl ->
          pl.action <= 0x0A or (pl.stock || stock_start) < stock_start
        end)

      %{start: s, len: e - s + 1, dj_used: dj_used, dj_frames: dj_frames, died: died}
    end)

  situations = length(results)
  dj_count = Enum.count(results, & &1.dj_used)
  deaths = Enum.count(results, & &1.died)
  returns = situations - deaths
  dj_and_returned = Enum.count(results, &(&1.dj_used and not &1.died))

  IO.puts("\n== #{label}: #{Path.basename(path)} (#{n} frames)")
  IO.puts("  recovery situations (offstage >=#{min_len}f): #{situations}")
  IO.puts("  DJ used during recovery: #{dj_count}/#{situations}")
  IO.puts("  outcomes: returned #{returns}, died #{deaths} (DJ'd and returned: #{dj_and_returned})")

  for r <- results do
    IO.puts(
      "    @f#{r.start} len=#{r.len} dj=#{if r.dj_used, do: "YES(#{r.dj_frames}f)", else: "no"} -> #{if r.died, do: "DIED", else: "returned"}"
    )
  end
end
