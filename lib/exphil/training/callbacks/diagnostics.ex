defmodule ExPhil.Training.Callbacks.Diagnostics do
  @moduledoc """
  Per-epoch diagnostics: button press rates, stick positions, per-head losses,
  action diversity. Displayed with colored comparison bars and tables.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.Output

  @impl true
  def init(opts) do
    %{
      verbose: Keyword.get(opts, :verbose, false),
      collapse_warned: false,
      # Per-head loss history for sparklines across epochs
      head_loss_history: %{buttons: [], main_x: [], main_y: [], c_x: [], c_y: [], shoulder: []}
    }
  end

  @impl true
  def on_batch_end(state, cb) do
    # Early collapse detection — check at batch 500 of epoch 1
    if state.epoch == 1 and state.batch_idx == 500 and not cb.collapse_warned do
      val_batches = state.pipeline.val_batches
      if val_batches && length(val_batches) >= 1 do
        batch = hd(val_batches)
        try do
          {btn_logits, mx_logits, _, _, _, _} =
            state.trainer.predict_fn.(state.trainer.policy_params, batch.states)

          # Check diversity — no per-index loops to avoid JIT recompilation
          pred_buttons = Nx.greater(Nx.sigmoid(btn_logits), 0.5) |> Nx.as_type(:u8)
          pred_mx = Nx.argmax(mx_logits, axis: -1)

          # Convert entire tensors to lists at once (one JIT, not N)
          all_btn_rows = Nx.to_list(pred_buttons)
          all_mx_vals = Nx.to_flat_list(pred_mx)
          combos = Enum.zip(Enum.take(all_btn_rows, 16), Enum.take(all_mx_vals, 16))
          unique = MapSet.new(combos) |> MapSet.size()

          # Check button confidence
          max_prob = Nx.reduce_max(Nx.sigmoid(btn_logits)) |> Nx.to_number()

          if unique <= 1 and max_prob < 0.1 do
            Output.puts_raw("\n  \e[31m⚠ COLLAPSE WARNING (batch 500): diversity=#{unique}, max button prob=#{Float.round(max_prob, 3)}\e[0m")
            Output.puts_raw("  \e[31m  Model is predicting neutral for everything. Consider:\e[0m")
            Output.puts_raw("  \e[31m  - Curriculum learning (--resume from 100-file checkpoint)\e[0m")
            Output.puts_raw("  \e[31m  - Higher entropy (--entropy-weight 0.1)\e[0m")
            Output.puts_raw("  \e[31m  - Bigger model (--hidden-sizes 1024,1024,512)\e[0m")
            {:cont, state, %{cb | collapse_warned: true}}
          else
            {:cont, state, cb}
          end
        rescue
          _ -> {:cont, state, cb}
        end
      else
        {:cont, state, cb}
      end
    else
      {:cont, state, cb}
    end
  end

  @impl true
  def on_epoch_end(state, cb) do
    val_batches = state.pipeline.val_batches

    new_cb =
      if val_batches && val_batches != [] do
        try do
          {head_losses, diag} = run_diagnostics(state.trainer, val_batches)

          # Update head loss history for sparklines
          updated_history = Map.new(cb.head_loss_history, fn {head, history} ->
            {head, history ++ [Map.get(head_losses, head, 0.0)]}
          end)

          # Display head loss sparklines (after 2+ epochs)
          if length(updated_history.buttons) >= 2 do
            sparkline_parts = Enum.map([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder], fn h ->
              short = %{buttons: "btn", main_x: "mx", main_y: "my", c_x: "cx", c_y: "cy", shoulder: "sh"}[h]
              vals = Map.get(updated_history, h)
              spark = Output.sparkline(vals)
              first = Float.round(hd(vals), 2)
              last = Float.round(List.last(vals), 2)
              "#{short}:#{spark}(#{first}→#{last})"
            end)
            Output.puts_raw("  Loss trend:   #{Enum.join(sparkline_parts, " ")}")
          end

          # Grouped accuracy display
          display_grouped_accuracy(diag)

          # Prediction confidence histogram
          display_confidence_histogram(state.trainer, val_batches)

          # Gradient norms (compact, always-on)
          run_gradient_diagnostics_compact(state.trainer, hd(val_batches))

          # Detailed gradient monitoring (verbose only — expensive)
          if cb.verbose or state.opts[:verbose] do
            run_gradient_diagnostics(state.trainer, hd(val_batches))
          end

          %{cb | head_loss_history: updated_history}
        rescue
          e ->
            Output.puts("  [diagnostics failed: #{Exception.message(e)}]")
            cb
        end
      else
        cb
      end

    :erlang.garbage_collect()
    {:cont, state, new_cb}
  end

  # Returns {normalized_head_losses, diag_accumulator} for use by caller
  defp run_diagnostics(trainer, val_batches) do
    sample = Enum.take(val_batches, 10)
    button_names = ~w(A B X Y Z L R D-Up)

    # Accumulate stats across sample batches
    init_acc = %{
      pred_counts: Nx.tensor(List.duplicate(0.0, 8)),
      actual_counts: Nx.tensor(List.duplicate(0.0, 8)),
      sticks: %{
        main_x: {0.0, 0.0, 0}, main_y: {0.0, 0.0, 0},
        c_x: {0.0, 0.0, 0}, c_y: {0.0, 0.0, 0},
        shoulder: {0.0, 0.0, 0}
      },
      head_losses: %{buttons: 0.0, main_x: 0.0, main_y: 0.0, c_x: 0.0, c_y: 0.0, shoulder: 0.0},
      pred_combos: MapSet.new(),
      actual_combos: MapSet.new(),
      # Accuracy metrics
      button_correct: Nx.tensor(List.duplicate(0.0, 8)),
      button_total: 0,
      stick_correct: %{main_x: 0, main_y: 0, c_x: 0, c_y: 0, shoulder: 0},
      rare_correct: %{z: 0, l: 0, r: 0},
      rare_total: %{z: 0, l: 0, r: 0},
      total: 0
    }

    diag = Enum.reduce(sample, init_acc, fn batch, acc ->
      {buttons_logits, mx_logits, my_logits, cx_logits, cy_logits, sh_logits} =
        trainer.predict_fn.(trainer.policy_params, batch.states)

      batch_size = elem(Nx.shape(mx_logits), 0)

      # Button rates
      preds = Nx.greater(Nx.sigmoid(buttons_logits), 0.5) |> Nx.as_type(:f32)
      actuals = Nx.greater(batch.actions.buttons, 0.5) |> Nx.as_type(:f32)
      pred_counts = Nx.add(acc.pred_counts, Nx.sum(preds, axes: [0]))
      actual_counts = Nx.add(acc.actual_counts, Nx.sum(actuals, axes: [0]))

      # Stick edge rates
      stick_heads = [
        {:main_x, mx_logits, batch.actions.main_x},
        {:main_y, my_logits, batch.actions.main_y},
        {:c_x, cx_logits, batch.actions.c_x},
        {:c_y, cy_logits, batch.actions.c_y},
        {:shoulder, sh_logits, batch.actions.shoulder}
      ]

      new_sticks = Enum.reduce(stick_heads, acc.sticks, fn {name, logits, targets}, stick_acc ->
        pred_bucket = Nx.argmax(logits, axis: -1)
        center = div(elem(Nx.shape(logits), 1), 2)
        pred_edge = Nx.not_equal(pred_bucket, center) |> Nx.sum() |> Nx.to_number()
        actual_edge = Nx.not_equal(targets, center) |> Nx.sum() |> Nx.to_number()
        {pe, ae, n} = Map.get(stick_acc, name)
        Map.put(stick_acc, name, {pe + pred_edge, ae + actual_edge, n + batch_size})
      end)

      # Per-head losses
      button_targets = batch.actions.buttons |> Nx.as_type(:f32)
      pos = Nx.multiply(button_targets, Axon.Activations.log_sigmoid(buttons_logits))
      neg = Nx.multiply(Nx.subtract(1.0, button_targets), Axon.Activations.log_sigmoid(Nx.negate(buttons_logits)))
      button_l = Nx.negate(Nx.add(pos, neg)) |> Nx.mean() |> Nx.to_number()

      ce = fn logits, targets ->
        n_classes = elem(Nx.shape(logits), 1)
        one_hot = Nx.equal(Nx.iota({1, n_classes}), Nx.reshape(targets, {:auto, 1})) |> Nx.as_type(:f32)
        log_probs = Axon.Activations.log_softmax(logits)
        Nx.negate(Nx.sum(Nx.multiply(one_hot, log_probs), axes: [-1])) |> Nx.mean() |> Nx.to_number()
      end

      head_losses = %{
        buttons: acc.head_losses.buttons + button_l,
        main_x: acc.head_losses.main_x + ce.(mx_logits, batch.actions.main_x),
        main_y: acc.head_losses.main_y + ce.(my_logits, batch.actions.main_y),
        c_x: acc.head_losses.c_x + ce.(cx_logits, batch.actions.c_x),
        c_y: acc.head_losses.c_y + ce.(cy_logits, batch.actions.c_y),
        shoulder: acc.head_losses.shoulder + ce.(sh_logits, batch.actions.shoulder)
      }

      # Action diversity (sample up to 64 per batch)
      pred_buttons = Nx.greater(Nx.sigmoid(buttons_logits), 0.5) |> Nx.as_type(:u8)
      pred_mx = Nx.argmax(mx_logits, axis: -1)
      pred_my = Nx.argmax(my_logits, axis: -1)
      actual_buttons = Nx.greater(batch.actions.buttons, 0.5) |> Nx.as_type(:u8)

      sample_n = min(batch_size, 64)
      pred_combos = for i <- 0..(sample_n - 1), reduce: acc.pred_combos do
        set ->
          btn = Nx.to_flat_list(pred_buttons[i])
          MapSet.put(set, {btn, Nx.to_number(pred_mx[i]), Nx.to_number(pred_my[i])})
      end
      actual_combos = for i <- 0..(sample_n - 1), reduce: acc.actual_combos do
        set ->
          btn = Nx.to_flat_list(actual_buttons[i])
          mx = Nx.to_number(batch.actions.main_x[i])
          my = Nx.to_number(batch.actions.main_y[i])
          MapSet.put(set, {btn, mx, my})
      end

      # Button accuracy: per-button correct predictions
      pred_binary = Nx.greater(Nx.sigmoid(buttons_logits), 0.5)
      target_binary = Nx.greater(batch.actions.buttons, 0.5)
      button_correct = Nx.equal(pred_binary, target_binary) |> Nx.as_type(:f32) |> Nx.sum(axes: [0])
      new_button_correct = Nx.add(acc.button_correct, button_correct)

      # Stick accuracy: top-1 bucket match
      new_stick_correct = Enum.reduce(stick_heads, acc.stick_correct, fn {name, logits, targets}, sc ->
        pred = Nx.argmax(logits, axis: -1)
        correct = Nx.equal(pred, targets) |> Nx.sum() |> Nx.to_number()
        Map.update!(sc, name, &(&1 + correct))
      end)

      # Rare action recall (Z=4, L=5, R=6)
      # Use Nx.take with fixed indices tensor — compiles once, no per-index JIT
      rare_indices = Nx.tensor([4, 5, 6])
      rare_targets = Nx.take(batch.actions.buttons, rare_indices, axis: 1)
      rare_preds = Nx.take(buttons_logits, rare_indices, axis: 1)

      rare_pressed = Nx.greater(rare_targets, 0.5)
      rare_pred_pressed = Nx.greater(rare_preds, 0)
      rare_hits = Nx.logical_and(rare_pred_pressed, rare_pressed)

      # Sum per column and convert to numbers all at once (one JIT, not three)
      pressed_counts = Nx.sum(rare_pressed, axes: [0]) |> Nx.to_flat_list() |> Enum.map(&trunc/1)
      hit_counts = Nx.sum(rare_hits, axes: [0]) |> Nx.to_flat_list() |> Enum.map(&trunc/1)

      {new_rare_correct, new_rare_total} =
        Enum.zip([:z, :l, :r], Enum.zip(pressed_counts, hit_counts))
        |> Enum.reduce({acc.rare_correct, acc.rare_total}, fn {name, {n_pressed, hits}}, {rc, rt} ->
          if n_pressed > 0 do
            {Map.update!(rc, name, &(&1 + hits)), Map.update!(rt, name, &(&1 + n_pressed))}
          else
            {rc, rt}
          end
        end)

      %{acc |
        pred_counts: pred_counts,
        actual_counts: actual_counts,
        sticks: new_sticks,
        head_losses: head_losses,
        pred_combos: pred_combos,
        actual_combos: actual_combos,
        button_correct: new_button_correct,
        button_total: acc.button_total + batch_size,
        stick_correct: new_stick_correct,
        rare_correct: new_rare_correct,
        rare_total: new_rare_total,
        total: acc.total + batch_size
      }
    end)

    n_batches = length(sample)

    # Display button press rates
    pred_rates = Nx.divide(diag.pred_counts, diag.total) |> Nx.multiply(100) |> Nx.to_flat_list()
    actual_rates = Nx.divide(diag.actual_counts, diag.total) |> Nx.multiply(100) |> Nx.to_flat_list()

    Output.puts("  Button press rates (pred vs actual):")
    Enum.zip([button_names, pred_rates, actual_rates])
    |> Enum.each(fn {name, pred, actual} ->
      Output.puts_raw(Output.comparison_bar(name, pred, actual))
    end)

    # Display stick diagnostics
    Output.puts("  Stick edge % (pred vs actual):")
    for {key, label} <- [main_x: "Main X", main_y: "Main Y", c_x: "C-X", c_y: "C-Y", shoulder: "Shldr"] do
      {pred_edge, actual_edge, n} = Map.get(diag.sticks, key)
      if n > 0 do
        pe = Float.round(pred_edge / n * 100, 1)
        ae = Float.round(actual_edge / n * 100, 1)
        Output.puts_raw(Output.comparison_bar(label, pe, ae))
      end
    end

    # Compact diagnostics display with colors
    # Head losses on one line
    head_parts = Enum.map([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder], fn h ->
      v = Float.round(Map.get(diag.head_losses, h) / n_batches, 2)
      short = %{buttons: "btn", main_x: "mx", main_y: "my", c_x: "cx", c_y: "cy", shoulder: "sh"}[h]
      color = cond do
        v < 0.5 -> "\e[32m"   # green — good
        v < 1.0 -> "\e[33m"   # yellow — ok
        true -> "\e[31m"      # red — high
      end
      "#{color}#{short}=#{v}\e[0m"
    end)
    Output.puts_raw("  Head losses:  #{Enum.join(head_parts, "  ")}")

    # Button accuracy on one line with colors
    if diag.button_total > 0 do
      btn_acc = Nx.divide(diag.button_correct, diag.button_total) |> Nx.multiply(100) |> Nx.to_flat_list()
      acc_parts = Enum.zip(button_names, btn_acc) |> Enum.map(fn {name, acc} ->
        v = Float.round(acc, 0) |> trunc()
        color = cond do
          v >= 95 -> "\e[32m"
          v >= 80 -> "\e[33m"
          true -> "\e[31m"
        end
        "#{color}#{name}=#{v}%\e[0m"
      end)
      overall = Float.round(Enum.sum(btn_acc) / length(btn_acc), 1)
      Output.puts_raw("  Button acc:   #{Enum.join(acc_parts, " ")}  \e[1m(#{overall}%)\e[0m")
    end

    # Stick accuracy on one line
    if diag.total > 0 do
      stick_parts = Enum.map([:main_x, :main_y, :c_x, :c_y, :shoulder], fn key ->
        acc = Float.round(Map.get(diag.stick_correct, key) / diag.total * 100, 0) |> trunc()
        short = %{main_x: "mx", main_y: "my", c_x: "cx", c_y: "cy", shoulder: "sh"}[key]
        color = cond do
          acc >= 80 -> "\e[32m"
          acc >= 40 -> "\e[33m"
          true -> "\e[31m"
        end
        "#{color}#{short}=#{acc}%\e[0m"
      end)
      Output.puts_raw("  Stick acc:    #{Enum.join(stick_parts, "  ")}")
    end

    # Diversity with colored bar
    pred_unique = MapSet.size(diag.pred_combos)
    actual_unique = MapSet.size(diag.actual_combos)
    coverage = if actual_unique > 0, do: Float.round(pred_unique / actual_unique * 100, 1), else: 0.0
    div_bar_len = min(round(coverage / 5), 20)
    div_color = cond do
      coverage >= 50 -> "\e[32m"
      coverage >= 20 -> "\e[33m"
      true -> "\e[31m"
    end
    div_bar = div_color <> String.duplicate("█", div_bar_len) <> String.duplicate("░", 20 - div_bar_len) <> "\e[0m"
    Output.puts_raw("  Diversity:    #{div_bar} #{pred_unique}/#{actual_unique} (#{coverage}%)")

    # Rare recall on one line
    rare_parts = Enum.map([:z, :l, :r], fn btn ->
      total = Map.get(diag.rare_total, btn, 0)
      if total > 0 do
        recall = Float.round(Map.get(diag.rare_correct, btn) / total * 100, 1)
        color = cond do
          recall >= 30 -> "\e[32m"
          recall >= 10 -> "\e[33m"
          true -> "\e[31m"
        end
        "#{color}#{btn}=#{recall}%\e[0m"
      else
        "\e[2m#{btn}=n/a\e[0m"
      end
    end)
    Output.puts_raw("  Rare recall:  #{Enum.join(rare_parts, "  ")}")

    # Temporal consistency (only for temporal/sequence models)
    # Measures prediction stability: for each pair of adjacent sequences in a batch
    # (which overlap by window_size - stride frames), how often are predictions the same?
    if length(sample) >= 2 do
      try do
        # Compare predictions on batch N vs batch N+1
        consistencies =
          sample
          |> Enum.chunk_every(2, 1, :discard)
          |> Enum.map(fn [b1, b2] ->
            {btn1, mx1, my1, _, _, _} = trainer.predict_fn.(trainer.policy_params, b1.states)
            {btn2, mx2, my2, _, _, _} = trainer.predict_fn.(trainer.policy_params, b2.states)

            # Button agreement: same predicted buttons
            p1 = Nx.greater(Nx.sigmoid(btn1), 0.5)
            p2 = Nx.greater(Nx.sigmoid(btn2), 0.5)
            btn_agree = Nx.equal(p1, p2) |> Nx.mean() |> Nx.to_number()

            # Stick agreement: same argmax bucket
            mx_agree = Nx.equal(Nx.argmax(mx1, axis: -1), Nx.argmax(mx2, axis: -1)) |> Nx.mean() |> Nx.to_number()
            my_agree = Nx.equal(Nx.argmax(my1, axis: -1), Nx.argmax(my2, axis: -1)) |> Nx.mean() |> Nx.to_number()

            {btn_agree, mx_agree, my_agree}
          end)

        if consistencies != [] do
          avg_btn = Enum.map(consistencies, &elem(&1, 0)) |> then(&(Enum.sum(&1) / length(&1)))
          avg_mx = Enum.map(consistencies, &elem(&1, 1)) |> then(&(Enum.sum(&1) / length(&1)))
          avg_my = Enum.map(consistencies, &elem(&1, 2)) |> then(&(Enum.sum(&1) / length(&1)))

          # Color: 100% = degenerate (red), 70-95% = healthy (green), <70% = jittery (yellow)
          color_consistency = fn v ->
            pct = Float.round(v * 100, 0) |> trunc()
            color = cond do
              pct >= 99 -> "\e[31m"   # red — degenerate (same prediction every frame)
              pct >= 70 -> "\e[32m"   # green — healthy variation
              true -> "\e[33m"        # yellow — jittery
            end
            "#{color}#{pct}%\e[0m"
          end

          warning = if avg_btn > 0.99 and avg_mx > 0.99, do: "  \e[31m(degenerate)\e[0m", else: ""
          Output.puts_raw("  Consistency:  btn=#{color_consistency.(avg_btn)}  mx=#{color_consistency.(avg_mx)}  my=#{color_consistency.(avg_my)}#{warning}")
        end
      rescue
        _ -> :ok
      end
    end

    # Return normalized head losses and full diag for caller
    normalized_losses = Map.new(diag.head_losses, fn {h, v} -> {h, v / n_batches} end)
    {normalized_losses, diag}
  end

  # Grouped accuracy: L|R (shield) and X|Y (jump) — functionally equivalent in Melee
  defp display_grouped_accuracy(diag) do
    if diag.button_total > 0 do
      btn_acc = Nx.divide(diag.button_correct, diag.button_total) |> Nx.to_flat_list()
      # Button order: A=0, B=1, X=2, Y=3, Z=4, L=5, R=6, D-Up=7
      jump_acc = (Enum.at(btn_acc, 2) + Enum.at(btn_acc, 3)) / 2
      shield_acc = (Enum.at(btn_acc, 5) + Enum.at(btn_acc, 6)) / 2

      # Grouped rates: were any of the group pressed?
      pred_rates = Nx.divide(diag.pred_counts, diag.total) |> Nx.to_flat_list()
      actual_rates = Nx.divide(diag.actual_counts, diag.total) |> Nx.to_flat_list()

      jump_pred = Enum.at(pred_rates, 2) + Enum.at(pred_rates, 3)
      jump_actual = Enum.at(actual_rates, 2) + Enum.at(actual_rates, 3)
      shield_pred = Enum.at(pred_rates, 5) + Enum.at(pred_rates, 6)
      shield_actual = Enum.at(actual_rates, 5) + Enum.at(actual_rates, 6)

      color = fn v ->
        pct = Float.round(v * 100, 0) |> trunc()
        c = cond do
          pct >= 95 -> "\e[32m"
          pct >= 80 -> "\e[33m"
          true -> "\e[31m"
        end
        "#{c}#{pct}%\e[0m"
      end

      rate_str = fn pred, actual ->
        p = Float.round(pred * 100, 1)
        a = Float.round(actual * 100, 1)
        delta = p - a
        arrow = cond do
          abs(delta) < 3 -> ""
          delta > 0 -> " \e[33m↑#{Float.round(delta, 1)}\e[0m"
          true -> " \e[33m↓#{Float.round(abs(delta), 1)}\e[0m"
        end
        "#{Float.round(p, 1)}%vs#{Float.round(a, 1)}%#{arrow}"
      end

      Output.puts_raw("  Grouped:      jump(X|Y)=#{color.(jump_acc)} rate=#{rate_str.(jump_pred, jump_actual)}  shield(L|R)=#{color.(shield_acc)} rate=#{rate_str.(shield_pred, shield_actual)}")
    end
  end

  # Prediction confidence histogram: distribution of max button probabilities
  defp display_confidence_histogram(trainer, val_batches) do
    batch = hd(val_batches)

    try do
      {btn_logits, mx_logits, _, _, _, _} =
        trainer.predict_fn.(trainer.policy_params, batch.states)

      # Button confidence: max sigmoid probability per sample
      btn_probs = Nx.sigmoid(btn_logits)
      max_btn_conf = Nx.reduce_max(btn_probs, axes: [1]) |> Nx.to_flat_list()

      # Stick confidence: max softmax probability per sample (main_x as representative)
      mx_probs = Nx.exp(Axon.Activations.log_softmax(mx_logits))
      max_mx_conf = Nx.reduce_max(mx_probs, axes: [1]) |> Nx.to_flat_list()

      # Bucket into ranges: [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
      bucket = fn vals ->
        counts = List.duplicate(0, 5)
        Enum.reduce(vals, counts, fn v, acc ->
          idx = min(trunc(v * 5), 4)
          List.update_at(acc, idx, &(&1 + 1))
        end)
      end

      btn_buckets = bucket.(max_btn_conf)
      mx_buckets = bucket.(max_mx_conf)
      n = length(max_btn_conf)

      # Render as mini bars
      render_hist = fn buckets ->
        max_count = Enum.max(buckets)
        scale = if max_count > 0, do: 8.0 / max_count, else: 1.0
        bars = ~w(▁ ▂ ▃ ▄ ▅ ▆ ▇ █)
        Enum.map(buckets, fn c ->
          idx = min(round(c * scale), 7) |> max(0)
          Enum.at(bars, idx)
        end) |> Enum.join("")
      end

      btn_hist = render_hist.(btn_buckets)
      mx_hist = render_hist.(mx_buckets)

      # Median confidence
      btn_median = Enum.sort(max_btn_conf) |> Enum.at(div(n, 2)) |> Float.round(2)
      mx_median = Enum.sort(max_mx_conf) |> Enum.at(div(n, 2)) |> Float.round(2)

      btn_color = cond do
        btn_median >= 0.7 -> "\e[32m"
        btn_median >= 0.4 -> "\e[33m"
        true -> "\e[31m"
      end

      mx_color = cond do
        mx_median >= 0.5 -> "\e[32m"
        mx_median >= 0.2 -> "\e[33m"
        true -> "\e[31m"
      end

      Output.puts_raw("  Confidence:   btn=#{btn_color}#{btn_hist}\e[0m med=#{btn_color}#{btn_median}\e[0m  stick=#{mx_color}#{mx_hist}\e[0m med=#{mx_color}#{mx_median}\e[0m")
    rescue
      _ -> :ok
    end
  end

  # Compact gradient norms — always shown, single line
  defp run_gradient_diagnostics_compact(trainer, batch) do
    alias ExPhil.Training.Imitation.TrainLoop

    try do
      {grads, _loss} = TrainLoop.compute_gradients(trainer, batch)
      norms = flatten_norms(grads, "")

      if norms != [] do
        all_norms = Enum.map(norms, &elem(&1, 1))
        max_norm = Enum.max(all_norms)
        min_norm = Enum.min(all_norms)
        mean_norm = Enum.sum(all_norms) / length(all_norms)

        max_color = if max_norm > 100, do: "\e[31m", else: if(max_norm > 10, do: "\e[33m", else: "\e[32m")
        min_color = if min_norm < 1.0e-7, do: "\e[31m", else: "\e[32m"

        Output.puts_raw("  Grad norms:   #{max_color}max=#{Float.round(max_norm, 4)}\e[0m  mean=#{Float.round(mean_norm, 4)}  #{min_color}min=#{Float.round(min_norm, 6)}\e[0m")
      end
    rescue
      _ -> :ok
    end
  end

  defp run_gradient_diagnostics(trainer, batch) do
    alias ExPhil.Training.Imitation.TrainLoop

    {grads, _loss} = TrainLoop.compute_gradients(trainer, batch)
    norms = flatten_norms(grads, "")

    if norms != [] do
      Output.puts("  Gradient norms:")
      {max_name, max_norm} = Enum.max_by(norms, &elem(&1, 1))
      {min_name, min_norm} = Enum.min_by(norms, &elem(&1, 1))

      rows = [
        ["max", max_name, Float.round(max_norm, 6)],
        ["min", min_name, Float.round(min_norm, 6)]
      ]
      Output.puts_raw(Output.table(["", "Layer", "Norm"], Enum.map(rows, fn r -> Enum.map(r, &to_string/1) end)))

      if max_norm > 100, do: Output.warning("  Exploding gradients detected (norm=#{Float.round(max_norm, 1)})")
      if min_norm < 1.0e-7, do: Output.warning("  Vanishing gradients detected (norm=#{Float.round(min_norm, 10)})")
    end
  end

  defp flatten_norms(%Nx.Tensor{} = t, path) do
    norm = t |> Nx.flatten() |> Nx.LinAlg.norm() |> Nx.to_number()
    [{path, norm}]
  end

  defp flatten_norms(map, path) when is_map(map) and not is_struct(map) do
    Enum.flat_map(map, fn {k, v} ->
      flatten_norms(v, if(path == "", do: to_string(k), else: "#{path}.#{k}"))
    end)
  end

  defp flatten_norms(_, _), do: []
end
