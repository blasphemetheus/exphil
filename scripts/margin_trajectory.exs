# Margin trajectory over a snapshot series, on a COMMON yardstick replay
# (2026-08-20, peak-decay interp): read every snapshot's signed critical-
# event margins on the SAME states — by default the argmax-gate epoch's
# own live replay — so epochs are compared on identical inputs.
#
# Context: teacher-forced fixture agreement is FLAT (~99.8%) across the
# whole g19 trajectory while live gates swing 437<->82, and the cliff is
# an ordinary-sized weight step. Hypothesis: the swing is a SIGN/thinness
# change of critical-event margins on the bot's own closed-loop state
# distribution. Events use PREV-frame family (sync-runner replays land
# the resulting action on the edge frame itself).
#
#   mix run scripts/margin_trajectory.exs \
#     --policies "checkpoints/ms_g19_ep*.bin" \
#     --replay eval_runs/0820_g19_gatesweep/sweep/ep4/r1.slp \
#     --delay-id 3 --offset -4 --out eval_runs/0820_g19_gatesweep/margins.jsonl

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, replay: :string, delay_id: :integer, offset: :integer, out: :string]
  )

policies =
  (opts[:policies] || raise("--policies required"))
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.sort_by(fn p ->
    case Regex.run(~r/ep(\d+)/, p) do
      [_, n] -> String.to_integer(n)
      _ -> 0
    end
  end)

replay_path = opts[:replay] || raise("--replay required")
offset = opts[:offset] || -4
out_path = opts[:out] || raise("--out required")

{:ok, parsed} = Peppi.parse(replay_path)

frames =
  parsed
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

# Critical events by PREV-frame family (see header). {t, event}
events =
  frames
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {[a, b], t} ->
    prev_fam = ShineChain.family(a.game_state.players[1].action)
    b_edge = b.controller.button_b and not a.controller.button_b
    x_edge = b.controller.button_x and not a.controller.button_x

    cond do
      x_edge and prev_fam == :ground_reflect -> [{t, :jc_event}]
      b_edge and prev_fam in [:jumpsquat, :aerial_jump] -> [{t, :aerial_shine_event}]
      true -> []
    end
  end)

Output.banner("Margin trajectory (common yardstick)")
Output.puts("replay: #{replay_path} (#{length(frames)} frames)")
Output.puts("events: #{Enum.count(events, &(elem(&1, 1) == :jc_event))} jc, " <>
  "#{Enum.count(events, &(elem(&1, 1) == :aerial_shine_event))} aerial_shine; offset #{offset}")

# Embed ONCE with the first policy's config (series shares one config).
first = Activations.load_heads(hd(policies))
ds = Activations.embed_frames(frames, first.config, delay_id: opts[:delay_id])
emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
{total, _} = Nx.shape(emb)
window = first.window

# Windows at event+offset (the frame whose logits produced the event input).
kept =
  events
  |> Enum.map(fn {t, ev} -> {t + offset, ev} end)
  |> Enum.filter(fn {t, _} -> t >= window - 1 and t < total end)

wins = Nx.stack(Enum.map(kept, fn {t, _} -> Nx.slice_along_axis(emb, t - window + 1, window, axis: 0) end))
Output.puts("windows: #{length(kept)}")

pct = fn sorted, p -> Enum.at(sorted, min(trunc(p * length(sorted)), length(sorted) - 1)) end
File.rm(out_path)

for path <- policies do
  name = Path.basename(path, ".bin")
  loaded = Activations.load_heads(path)

  {b_logits, x_logits} =
    kept
    |> Enum.with_index()
    |> Enum.chunk_every(512)
    |> Enum.map(fn chunk ->
      idxs = Enum.map(chunk, fn {_, i} -> i end)
      batch = Nx.take(wins, Nx.tensor(idxs))
      out = loaded.predict_fn.(loaded.params, batch)
      buttons = elem(out, 0)
      {Nx.to_flat_list(buttons[[.., 1]]), Nx.to_flat_list(buttons[[.., 2]])}
    end)
    |> Enum.reduce({[], []}, fn {b, x}, {ab, ax} -> {ab ++ b, ax ++ x} end)

  margins =
    kept
    |> Enum.with_index()
    |> Enum.map(fn {{_t, ev}, i} ->
      {ev, if(ev == :jc_event, do: Enum.at(x_logits, i), else: Enum.at(b_logits, i))}
    end)

  stats =
    margins
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Map.new(fn {ev, ms} ->
      s = Enum.sort(ms)

      {ev,
       %{n: length(s), mean: Float.round(Enum.sum(s) / length(s), 3),
         p10: Float.round(pct.(s, 0.10), 3), min: Float.round(hd(s), 3),
         flip_frac: Float.round(Enum.count(s, &(&1 < 0)) / length(s), 4)}}
    end)

  jc = stats[:jc_event]
  as = stats[:aerial_shine_event]

  Output.puts(
    "#{String.pad_trailing(name, 14)} jc: p10=#{jc && jc.p10} flip=#{jc && jc.flip_frac}  " <>
      "aerial: p10=#{as && as.p10} flip=#{as && as.flip_frac}"
  )

  File.write!(out_path, Jason.encode!(%{policy: name, stats: stats}) <> "\n", [:append])
end

Output.success("Written: #{out_path}")