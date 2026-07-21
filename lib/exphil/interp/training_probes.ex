defmodule ExPhil.Interp.TrainingProbes do
  @moduledoc """
  Feature-formation curves during training (the last open P2 item, and
  the epoch-budget instrument): every K epochs, quick-probe a small set
  of non-saturated ground-truth features on the CURRENT trunk and log
  the balanced accuracies. Where the curves plateau is where the epoch
  budget should stop; whether representation growth PRECEDES card growth
  is the P2/P5 question the per-round deficit reports can't see inside a
  run.

  TREND instrument, not a calibrated measurement: rows are sampled
  windows split into time blocks (frames within a game are correlated),
  and probes run few steps. Absolute numbers belong to
  `ExPhil.Interp.DeficitReport`; only the SHAPE over epochs is read.
  """

  alias ExPhil.Interp.Probe
  alias ExPhil.Training.{Data, Utils}

  # Non-saturated features (per the 2026-07-21 cross-arch report) with
  # class counts — the ones whose formation is worth watching
  @features [
    opp_knockdown: 2,
    opp_shielding: 2,
    own_offstage: 2,
    time_since_kd_bucket: 5,
    frames_until_kd_bucket: 5,
    tech_choice: 4
  ]

  @doc "The watched feature set (name => classes)."
  def features, do: @features

  @doc """
  Probe the current trunk. `trunk_fn` is a trunk predict fn sharing the
  policy's param names (`ProbeRegularizer.build_trunk_fn/1` or
  `trainer.trunk_predict_fn`); `per_frame_labels` is the tuple-ized
  output of `GroundTruth.frame_labels/2` over `dataset.frames`.

  Returns `%{feature => balanced_accuracy | nil}` (nil = a class was
  absent from the sample).
  """
  def eval(trainer, trunk_fn, dataset, per_frame_labels, opts \\ []) do
    window = Keyword.get(opts, :window_size) || trainer.config.window_size
    stride = Keyword.get(opts, :stride, 1)
    batch_size = Keyword.get(opts, :batch_size, 256)
    target_rows = Keyword.get(opts, :target_rows, 4096)
    steps = Keyword.get(opts, :steps, 150)

    n_labels = tuple_size(per_frame_labels)
    num_sequences = div(dataset.size - window, stride) + 1
    every = max(div(num_sequences, target_rows), 1)

    {h_parts, label_maps} =
      dataset
      |> Data.batched_sequences(
        batch_size: batch_size,
        window_size: window,
        stride: stride,
        lazy: true,
        shuffle: false,
        drop_last: false
      )
      |> Stream.with_index()
      |> Stream.filter(fn {_b, j} -> rem(j, every) == 0 end)
      |> Enum.reduce({[], []}, fn {batch, j}, {hs, ls} ->
        h = trunk_fn.(Utils.ensure_model_state(trainer.policy_params), batch.states)
        rows = Nx.axis_size(batch.states, 0)
        base = j * batch_size

        labels =
          for k <- 0..(rows - 1) do
            idx = (base + k) * stride + window - 1
            if idx < n_labels, do: elem(per_frame_labels, idx), else: nil
          end

        {[Nx.backend_copy(Nx.as_type(h, :f32), Nx.BinaryBackend) | hs], [labels | ls]}
      end)

    if h_parts == [] do
      %{}
    else
      x = h_parts |> Enum.reverse() |> Nx.concatenate()
      labels = label_maps |> Enum.reverse() |> Enum.concat()
      n = Nx.axis_size(x, 0)
      split_at = trunc(n * 0.75)

      x_train = Nx.slice_along_axis(x, 0, split_at, axis: 0)
      x_eval = Nx.slice_along_axis(x, split_at, n - split_at, axis: 0)

      Map.new(@features, fn {feature, k} ->
        y =
          Enum.map(labels, fn
            nil -> -1
            m -> Map.get(m, feature, -1) || -1
          end)

        {y_train, y_eval} = Enum.split(y, split_at)

        result =
          Probe.fit_eval(
            x_train,
            Nx.tensor(y_train, type: :s64),
            x_eval,
            Nx.tensor(y_eval, type: :s64),
            k,
            steps: steps
          )

        {feature, result.balanced_accuracy}
      end)
    end
  end

  @doc "One-line log string for an eval result."
  def format(bas) do
    @features
    |> Enum.map(fn {f, _} ->
      case bas[f] do
        nil -> "#{f}=-"
        ba -> "#{f}=#{:erlang.float_to_binary(ba, decimals: 3)}"
      end
    end)
    |> Enum.join(" ")
  end
end
