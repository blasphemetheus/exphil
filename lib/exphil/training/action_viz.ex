defmodule ExPhil.Training.ActionViz do
  @moduledoc """
  Action distribution visualization for policy analysis.

  Tracks and displays what buttons and stick positions the model uses,
  helping identify if the model is learning diverse actions or collapsing
  to a small subset of behaviors.

  ## Usage

      # During evaluation
      viz = ActionViz.new()

      for action <- actions do
        viz = ActionViz.record(viz, action)
      end

      ActionViz.print_summary(viz)

  ## Output Example

      ═══ Action Distribution (n=1000) ═══

      Buttons:
        A     ████████████████████░░░░░ 80.5%
        B     ██████████░░░░░░░░░░░░░░░ 42.3%
        X     ██░░░░░░░░░░░░░░░░░░░░░░░  8.1%
        Y     █░░░░░░░░░░░░░░░░░░░░░░░░  3.2%
        Z     ░░░░░░░░░░░░░░░░░░░░░░░░░  0.1%
        ...

      Main Stick (heatmap):
        ░░░░░░░░░░░░░░░░░
        ░░░░░░▓▓▓░░░░░░░░
        ░░░░▓▓████▓▓░░░░░
        ░░░▓███████▓░░░░░  <- Y axis
        ░░░░▓▓████▓▓░░░░░
        ░░░░░░▓▓▓░░░░░░░░
        ░░░░░░░░░░░░░░░░░
               X axis ->
  """

  alias ExPhil.Training.Output

  @button_names [:a, :b, :x, :y, :z, :l, :r, :d_up]
  @button_display %{
    a: "A",
    b: "B",
    x: "X",
    y: "Y",
    z: "Z",
    l: "L",
    r: "R",
    d_up: "D-Up"
  }

  defstruct [
    # Button press counts
    button_counts: %{},
    # Stick position counts (discretized to grid)
    main_stick_counts: %{},
    c_stick_counts: %{},
    # Shoulder counts
    shoulder_counts: %{},
    # Total samples
    total: 0,
    # Grid resolution for stick visualization
    grid_size: 17
  ]

  @type t :: %__MODULE__{
          button_counts: %{atom() => non_neg_integer()},
          main_stick_counts: %{{integer(), integer()} => non_neg_integer()},
          c_stick_counts: %{{integer(), integer()} => non_neg_integer()},
          shoulder_counts: %{integer() => non_neg_integer()},
          total: non_neg_integer(),
          grid_size: pos_integer()
        }

  @doc """
  Create a new action visualizer.

  ## Options

    * `:grid_size` - Resolution for stick heatmap (default: 17, matching 16 buckets + 1)

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    grid_size = Keyword.get(opts, :grid_size, 17)
    button_counts = Map.new(@button_names, fn b -> {b, 0} end)

    %__MODULE__{
      button_counts: button_counts,
      main_stick_counts: %{},
      c_stick_counts: %{},
      shoulder_counts: %{},
      total: 0,
      grid_size: grid_size
    }
  end

  @doc """
  Record an action from the policy.

  Accepts either:
  - A map with :buttons, :main_x, :main_y, :c_x, :c_y, :shoulder keys (integers)
  - An Nx.Tensor of the flattened controller output

  ## Examples

      viz = ActionViz.record(viz, %{
        buttons: 0b00000001,  # A button pressed
        main_x: 8,           # Center
        main_y: 8,
        c_x: 8,
        c_y: 8,
        shoulder: 0
      })

  """
  @spec record(t(), map() | Nx.Tensor.t()) :: t()
  def record(%__MODULE__{} = viz, action) when is_map(action) do
    # Extract components
    buttons = Map.get(action, :buttons, 0)
    main_x = Map.get(action, :main_x, 8)
    main_y = Map.get(action, :main_y, 8)
    c_x = Map.get(action, :c_x, 8)
    c_y = Map.get(action, :c_y, 8)
    shoulder = Map.get(action, :shoulder, 0)

    # Update button counts (buttons is a bitmask or list)
    button_counts =
      if is_integer(buttons) do
        update_button_counts_bitmask(viz.button_counts, buttons)
      else
        # Assume list of atoms
        update_button_counts_list(viz.button_counts, buttons)
      end

    # Update stick counts
    main_stick_counts = Map.update(viz.main_stick_counts, {main_x, main_y}, 1, &(&1 + 1))
    c_stick_counts = Map.update(viz.c_stick_counts, {c_x, c_y}, 1, &(&1 + 1))

    # Update shoulder counts
    shoulder_counts = Map.update(viz.shoulder_counts, shoulder, 1, &(&1 + 1))

    %{
      viz
      | button_counts: button_counts,
        main_stick_counts: main_stick_counts,
        c_stick_counts: c_stick_counts,
        shoulder_counts: shoulder_counts,
        total: viz.total + 1
    }
  end

  def record(%__MODULE__{} = viz, tensor) when is_struct(tensor, Nx.Tensor) do
    # Convert tensor to action map
    # Assumes tensor is [buttons(8), main_x, main_y, c_x, c_y, shoulder]
    flat = Nx.to_flat_list(tensor)

    {button_vals, rest} = Enum.split(flat, 8)
    [main_x, main_y, c_x, c_y, shoulder | _] = rest

    buttons =
      button_vals
      |> Enum.with_index()
      |> Enum.filter(fn {v, _} -> v > 0.5 end)
      |> Enum.map(fn {_, i} -> Enum.at(@button_names, i) end)

    record(viz, %{
      buttons: buttons,
      main_x: round(main_x),
      main_y: round(main_y),
      c_x: round(c_x),
      c_y: round(c_y),
      shoulder: round(shoulder)
    })
  end

  @doc """
  Record multiple actions from a batch.
  """
  @spec record_batch(t(), [map()] | Nx.Tensor.t()) :: t()
  def record_batch(%__MODULE__{} = viz, actions) when is_list(actions) do
    Enum.reduce(actions, viz, &record(&2, &1))
  end

  def record_batch(%__MODULE__{} = viz, tensor) when is_struct(tensor, Nx.Tensor) do
    # Batch tensor: {batch_size, action_dim}
    tensor
    |> Nx.to_batched(1)
    |> Enum.reduce(viz, fn action, acc ->
      record(acc, Nx.squeeze(action))
    end)
  end

  @doc """
  Record actions from component tensors (buttons, sticks, shoulder).

  This is useful for recording predictions from policy networks that output
  separate tensors for each component.

  ## Parameters

    * `viz` - The ActionViz struct
    * `components` - Map with keys: :buttons, :main_x, :main_y, :c_x, :c_y, :shoulder
      - :buttons is a {batch, 8} boolean tensor
      - :main_x, :main_y, :c_x, :c_y are {batch} integer tensors (bucket indices)
      - :shoulder is a {batch} integer tensor
    * `axis_buckets` - Number of axis buckets (for converting indices to grid)

  """
  @spec record_batch(t(), map(), non_neg_integer()) :: t()
  def record_batch(%__MODULE__{} = viz, components, axis_buckets) when is_map(components) do
    buttons = components[:buttons]
    main_x = components[:main_x]
    main_y = components[:main_y]
    c_x = components[:c_x]
    c_y = components[:c_y]
    shoulder = components[:shoulder]

    batch_size = Nx.axis_size(main_x, 0)

    # Convert to lists for iteration
    buttons_list = Nx.to_list(buttons)
    main_x_list = Nx.to_list(main_x)
    main_y_list = Nx.to_list(main_y)
    c_x_list = Nx.to_list(c_x)
    c_y_list = Nx.to_list(c_y)
    shoulder_list = Nx.to_list(shoulder)

    # Record each sample
    Enum.reduce(0..(batch_size - 1), viz, fn i, acc ->
      button_values = Enum.at(buttons_list, i)

      action = %{
        buttons: button_values,
        main_x: bucket_to_grid(Enum.at(main_x_list, i), axis_buckets, acc.grid_size),
        main_y: bucket_to_grid(Enum.at(main_y_list, i), axis_buckets, acc.grid_size),
        c_x: bucket_to_grid(Enum.at(c_x_list, i), axis_buckets, acc.grid_size),
        c_y: bucket_to_grid(Enum.at(c_y_list, i), axis_buckets, acc.grid_size),
        shoulder: Enum.at(shoulder_list, i)
      }

      record(acc, action)
    end)
  end

  # Convert axis bucket index to grid position
  defp bucket_to_grid(bucket_idx, axis_buckets, grid_size) do
    # Map bucket index (0..axis_buckets) to grid position (0..grid_size-1)
    # Bucket 0 = center/neutral, buckets spread around
    round(bucket_idx / axis_buckets * (grid_size - 1))
  end

  @doc """
  Print a summary of the action distribution.
  """
  @spec print_summary(t(), keyword()) :: :ok
  def print_summary(viz, opts \\ [])

  def print_summary(%__MODULE__{total: 0}, _opts) do
    Output.puts_raw("  No actions recorded")
    :ok
  end

  def print_summary(%__MODULE__{} = viz, opts) do
    show_sticks = Keyword.get(opts, :show_sticks, true)
    show_shoulder = Keyword.get(opts, :show_shoulder, true)

    Output.section("Action Distribution (n=#{viz.total})")
    Output.puts_raw("")

    # Button distribution
    Output.puts_raw("  " <> Output.colorize("Buttons:", :bold))
    print_button_distribution(viz)

    if show_sticks do
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Main Stick:", :bold))
      print_stick_heatmap(viz.main_stick_counts, viz.grid_size, viz.total)

      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("C-Stick:", :bold))
      print_stick_heatmap(viz.c_stick_counts, viz.grid_size, viz.total)
    end

    if show_shoulder && map_size(viz.shoulder_counts) > 0 do
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Shoulder:", :bold))
      print_shoulder_distribution(viz)
    end

    Output.divider()
    :ok
  end

  @doc """
  Get statistics as a map (for logging/export).
  """
  @spec to_stats(t()) :: map()
  def to_stats(%__MODULE__{} = viz) do
    %{
      total_samples: viz.total,
      button_rates: button_rates(viz),
      stick_entropy: %{
        main: stick_entropy(viz.main_stick_counts, viz.total),
        c: stick_entropy(viz.c_stick_counts, viz.total)
      }
    }
  end

  @doc """
  Calculate button press rates.
  """
  @spec button_rates(t()) :: map()
  def button_rates(%__MODULE__{total: 0}), do: Map.new(@button_names, &{&1, 0.0})

  def button_rates(%__MODULE__{} = viz) do
    Map.new(viz.button_counts, fn {button, count} ->
      {button, count / viz.total}
    end)
  end

  # Private helpers

  defp update_button_counts_bitmask(counts, bitmask) do
    @button_names
    |> Enum.with_index()
    |> Enum.reduce(counts, fn {button, idx}, acc ->
      if Bitwise.band(bitmask, Bitwise.bsl(1, idx)) != 0 do
        Map.update!(acc, button, &(&1 + 1))
      else
        acc
      end
    end)
  end

  defp update_button_counts_list(counts, buttons) when is_list(buttons) do
    Enum.reduce(buttons, counts, fn button, acc ->
      if Map.has_key?(acc, button) do
        Map.update!(acc, button, &(&1 + 1))
      else
        acc
      end
    end)
  end

  defp print_button_distribution(%__MODULE__{total: total, button_counts: counts}) do
    bar_width = 25

    counts
    |> Enum.sort_by(fn {_k, v} -> -v end)
    |> Enum.each(fn {button, count} ->
      pct = if total > 0, do: count / total * 100, else: 0.0
      filled = round(pct / 100 * bar_width)
      bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)
      name = @button_display[button] || to_string(button)
      name_padded = String.pad_trailing(name, 5)
      pct_str = :io_lib.format("~5.1f%", [pct]) |> IO.iodata_to_binary()

      color =
        cond do
          pct >= 50 -> :green
          pct >= 20 -> :yellow
          pct >= 5 -> :cyan
          true -> :dim
        end

      Output.puts_raw("    #{name_padded} #{Output.colorize(bar, color)} #{pct_str}")
    end)
  end

  defp print_stick_heatmap(counts, grid_size, total) when total > 0 do
    # Find max count for scaling
    max_count = counts |> Map.values() |> Enum.max(fn -> 1 end)

    # Heatmap characters (8 intensity levels)
    chars = [" ", "░", "▒", "▓", "█", "█", "█", "█"]

    # Print grid (Y axis is inverted for display)
    for y <- (grid_size - 1)..0//-1 do
      row =
        for x <- 0..(grid_size - 1) do
          count = Map.get(counts, {x, y}, 0)
          intensity = if max_count > 0, do: round(count / max_count * 7), else: 0
          intensity = min(intensity, 7)
          Enum.at(chars, intensity)
        end

      y_label = if y == div(grid_size, 2), do: "<-Y", else: "   "
      Output.puts_raw("    #{Enum.join(row)} #{y_label}")
    end

    Output.puts_raw("    " <> String.duplicate(" ", div(grid_size, 2)) <> "X->")

    # Show center concentration
    center = div(grid_size, 2)
    center_count = Map.get(counts, {center, center}, 0)
    center_pct = Float.round(center_count / total * 100, 1)
    Output.puts_raw("    (center: #{center_pct}%)")
  end

  defp print_stick_heatmap(_counts, _grid_size, 0) do
    Output.puts_raw("    (no data)")
  end

  defp print_shoulder_distribution(%__MODULE__{total: total, shoulder_counts: counts}) do
    max_val = counts |> Map.keys() |> Enum.max(fn -> 0 end)

    for i <- 0..max_val do
      count = Map.get(counts, i, 0)
      pct = if total > 0, do: count / total * 100, else: 0.0
      bar_width = round(pct / 100 * 20)
      bar = String.duplicate("█", bar_width)
      Output.puts_raw("    #{i}: #{String.pad_trailing(bar, 20)} #{Float.round(pct, 1)}%")
    end
  end

  # Calculate entropy of stick positions (higher = more diverse)
  defp stick_entropy(counts, total) when total > 0 do
    counts
    |> Map.values()
    |> Enum.map(fn count ->
      p = count / total
      if p > 0, do: -p * :math.log2(p), else: 0.0
    end)
    |> Enum.sum()
    |> Float.round(2)
  end

  defp stick_entropy(_, _), do: 0.0
end
