# INIT_FORENSICS options #7 + #3 (task #15) over the 12-seed crouch zoo.
#
# #7 subset-fit: teacher-forced B agreement + mean B-BCE on the
#    crouch-synth block. Discriminates "optimization never fit the
#    escape labels" (low agreement) from "fit, failed closed-loop"
#    (high agreement, live-absorbed).
# #3 trunk probes: can a linear readout decode (a) the B label and
#    (b) af PARITY (the alternation clock) from each seed's trunk?
#    Random-init trunk = the control every probe must beat.
#
# Known live tiers (INIT_FORENSICS_OPTIONS.md):
#   universal escapers a,c,i · self-consistent b,d,k ·
#   absorbed e,f,g,h,j,l (hold-B: g,h,j; silent: e,f; 2-cycle: l)
#
#   mix run scripts/interp_init_forensics.exs [--seeds "a,b,..."]

alias ExPhil.Data.{Peppi, RecoverySynth}
alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.{Data, Output}

{opts, _, _} = OptionParser.parse(System.argv(), strict: [seeds: :string])
seeds = String.split(opts[:seeds] || "a,b,c,d,e,f,g,h,i,j,k,l", ",", trim: true)

tier = fn s ->
  cond do
    s in ~w(a c i) -> "escape-universal"
    s in ~w(b d k) -> "escape-self"
    s in ~w(g h j) -> "absorbed-holdB"
    s in ~w(e f) -> "absorbed-silent"
    s == "l" -> "absorbed-2cycle"
    true -> "?"
  end
end

Output.banner("Init forensics: subset-fit (#7) + trunk probes (#3)")

{:ok, replay} = Peppi.parse("test/fixtures/replays/fox_multishine_closed.slp")

fixture_frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)

block = RecoverySynth.build_crouch(fixture_frames, port: 1, max_af: 40, lead_in: 16, ratio: 1.0)
Output.puts("crouch-synth block: #{length(block)} frames")

ds = block |> Data.from_frames() |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)
emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
{total, _} = Nx.shape(emb)
window = 16
n = total - window + 1
arr = List.to_tuple(block)

# Labels at window-end frames
b_labels = for t <- (window - 1)..(total - 1), do: (if elem(arr, t).controller.button_b, do: 1, else: 0)
par_labels = for t <- (window - 1)..(total - 1), do: rem(elem(arr, t).game_state.players[1].action_frame, 2)
y_b = Nx.tensor(b_labels, type: :s64)
y_par = Nx.tensor(par_labels, type: :s64)

windows =
  (window - 1)..(total - 1)
  |> Enum.map(&Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
  |> Nx.stack()

probe = fn acts, y ->
  half = div(n, 2)
  slice = fn t, a, b -> Nx.slice_along_axis(t, a, b, axis: 0) end
  r = Probe.fit_eval(slice.(acts, 0, half), slice.(y, 0, half), slice.(acts, half, n - half), slice.(y, half, n - half), 2)
  r.balanced_accuracy
end

# Random-init trunk control (architecture from any zoo checkpoint)
ctrl = Activations.load_trunk("checkpoints/ms_crouch_a.bin", init: :random)
ctrl_acts = ctrl.predict_fn.(ctrl.params, windows)
Output.puts("control(random trunk): B=#{Float.round(probe.(ctrl_acts, y_b), 3)} parity=#{Float.round(probe.(ctrl_acts, y_par), 3)}")

for s <- seeds do
  path = "checkpoints/ms_crouch_#{s}.bin"
  heads = Activations.load_heads(path)
  out = heads.predict_fn.(heads.params, windows)
  b_logits = out |> elem(0) |> then(& &1[[.., 1]])

  # #7: teacher-forced fit on the synth slice. Scored at label offsets
  # {-1, 0, +1} and reported as best — the training pairing lags state by
  # one frame (CycleMargins lesson), and a perfect alternator scored ~0.25
  # at offset 0 (anti-phase = worse than chance) in the v1 run.
  pred = Nx.greater(b_logits, 0.0) |> Nx.as_type(:s64)

  {acc, off} =
    Enum.map(-1..1, fn o ->
      {lo, hi} = {max(o, 0), n - abs(o)}
      p = Nx.slice_along_axis(pred, lo, hi, axis: 0)
      y = Nx.slice_along_axis(y_b, max(-o, 0), hi, axis: 0)
      {Nx.to_number(Nx.mean(Nx.equal(p, y))), o}
    end)
    |> Enum.max_by(&elem(&1, 0))
  p = Nx.sigmoid(b_logits)
  eps = 1.0e-7
  yf = Nx.as_type(y_b, :f32)
  bce =
    Nx.mean(Nx.negate(Nx.add(Nx.multiply(yf, Nx.log(Nx.add(p, eps))), Nx.multiply(Nx.subtract(1.0, yf), Nx.log(Nx.add(Nx.subtract(1.0, p), eps))))))
    |> Nx.to_number()

  # #3: trunk probes
  trunk = Activations.load_trunk(path)
  acts = trunk.predict_fn.(trunk.params, windows)
  pb = probe.(acts, y_b)
  ppar = probe.(acts, y_par)

  Output.puts(
    "#{s} #{String.pad_trailing(tier.(s), 18)} fitB=#{Float.round(acc, 3)}@#{off} bce=#{Float.round(bce, 3)} " <>
      "probeB=#{Float.round(pb, 3)} probePar=#{Float.round(ppar, 3)}"
  )
end
