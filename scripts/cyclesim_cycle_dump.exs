# One-off (#5): live logits frame-by-frame through the champion's first
# few REAL multishine cycles — where does the aerial-B decision fire?
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain

loaded = Activations.load_heads("checkpoints/ms_open_z.bin")
{:ok, parsed} = Peppi.parse("eval_runs/0728_open_z_idle/r1.slp")

frames =
  parsed
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))

ds = frames |> Data.from_frames() |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)
emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
window = loaded.window
frames_arr = List.to_tuple(frames)

# First grounded-reflect frame after frame 200 anchors a real cycle
start =
  Enum.find(200..(tuple_size(frames_arr) - 60), fn t ->
    ShineChain.family(elem(frames_arr, t).game_state.players[1].action) == :ground_reflect
  end)

Output.puts("cycle anchor at index #{start}")

for t <- start..(start + 40) do
  f = elem(frames_arr, t)
  p = f.game_state.players[1]
  win = Nx.slice_along_axis(emb, t - window + 1, window, axis: 0)
  out = loaded.predict_fn.(loaded.params, Nx.new_axis(win, 0))
  buttons = out |> elem(0) |> Nx.squeeze() |> Nx.to_flat_list()
  b = Enum.at(buttons, 1)
  x = Enum.at(buttons, 2)
  fam = ShineChain.family(p.action)

  Output.puts(
    "t#{t} a=#{p.action}/#{fam} af#{p.action_frame} " <>
      "ctrl(B=#{p && f.controller.button_b},X=#{f.controller.button_x}) " <>
      "logit B=#{Float.round(b, 2)} X=#{Float.round(x, 2)}"
  )
end
