Application.ensure_all_started(:nx)
alias ExPhil.Training.SilentFallWeighting, as: SFW
files = Jason.decode!(File.read!("checkpoints/coh_evt2ctx_ck8_off3_dur8e_pd15_dng/split.json"))["train"] |> Enum.take(20)
{tot, off, on} =
  Enum.reduce(files, {0, 0, 0}, fn path, {t, o, n} ->
    {:ok, replay} = ExPhil.Data.Peppi.parse(path, player_port: 1)
    frames = ExPhil.Data.Peppi.to_training_frames(replay, player_port: 1, frame_delay: 1)
    ws = SFW.frame_weights(frames, offstage_weight: 3, onset_weight: 10)
    {t + length(ws), o + Enum.count(ws, &(&1 >= 3.0)), n + Enum.count(ws, &(&1 == 10.0))}
  end)
IO.puts("frames #{tot}  offstage(>=3) #{off} (#{Float.round(100 * off / tot, 2)} %)  onsets(10) #{on} (#{Float.round(100 * on / max(off, 1), 2)} % of offstage; #{Float.round(on / length(files), 1)} per game)")
