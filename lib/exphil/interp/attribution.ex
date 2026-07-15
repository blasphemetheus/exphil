defmodule ExPhil.Interp.Attribution do
  @moduledoc """
  Input attribution (gradient × input) for policy heads — interp P3 case #3.

  Answers "which input dimensions drove this decision?" by differentiating
  a head's chosen logit w.r.t. the input window and aggregating |grad × x|
  into named dimension groups (own position, opponent position, ...).

  Dimension groups are discovered EMPIRICALLY: embed a reference game
  state, perturb one field (e.g. opponent x), re-embed, and diff — the
  changed dims belong to that field. This can't drift out of sync with the
  embedding layout the way a hand-written offset table would (the layout
  has already scrambled silently once — see embed_path_parity_test).
  """

  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Embeddings

  @doc """
  Discover embedding dim indices for a set of named perturbations.

  Returns `%{group_name => [dim indices]}` for the given embed config
  (prev-action variant chosen via `:use_prev_action`).
  """
  def discover_dims(_embed_config, opts \\ []) do
    use_prev_action = Keyword.get(opts, :use_prev_action, true)

    base = reference_state()
    prev = if use_prev_action, do: neutral_controller(), else: nil

    embed = fn gs -> Embeddings.Game.embed(gs, prev, 1) |> Nx.backend_transfer(Nx.BinaryBackend) end
    e0 = embed.(base)

    perturbations = [
      own_position: update_player(base, 1, %{x: 31.4, y: 7.7}),
      own_facing: update_player(base, 1, %{facing: -1.0}),
      own_action: update_player(base, 1, %{action: 66, action_frame: 3}),
      own_speeds: update_player(base, 1, %{speed_air_x_self: 1.3, speed_y_self: -2.1}),
      opp_position: update_player(base, 2, %{x: -27.2, y: 11.3}),
      opp_facing: update_player(base, 2, %{facing: -1.0}),
      opp_action: update_player(base, 2, %{action: 90, action_frame: 4}),
      opp_percent: update_player(base, 2, %{percent: 87.0})
    ]

    groups =
      Map.new(perturbations, fn {name, gs} ->
        diff = Nx.not_equal(embed.(gs), e0)
        idx = diff |> Nx.to_flat_list() |> Enum.with_index() |> Enum.filter(fn {d, _} -> d == 1 end) |> Enum.map(&elem(&1, 1))
        {name, idx}
      end)

    prev_dims =
      if use_prev_action do
        e_prev = Embeddings.Game.embed(base, pressed_controller(), 1) |> Nx.backend_transfer(Nx.BinaryBackend)
        Nx.not_equal(e_prev, e0)
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.filter(fn {d, _} -> d == 1 end)
        |> Enum.map(&elem(&1, 1))
      else
        []
      end

    Map.put(groups, :prev_action, prev_dims)
  end

  @doc """
  Gradient × input saliency for one head over a batch of input windows.

  `predict_fn`/`params` from `Activations.load_heads/1`; `states` is
  `{n, window, embed}`. `head` picks which output; the scalar objective is
  the summed logit at each row's (fixed) argmax bucket.

  Returns `{n, embed}` — |grad × input| summed over the window axis.
  """
  def saliency(predict_fn, params, states, head \\ :main_x) do
    head_idx = %{buttons: 0, main_x: 1, main_y: 2, c_x: 3, c_y: 4, shoulder: 5}[head]

    # params passed as a jit argument (closure-captured EXLA tensors crash
    # the trace) and the argmax one-hot computed INSIDE the trace under
    # stop_grad — the objective is each row's chosen-bucket logit, with the
    # choice itself treated as a constant.
    grads =
      Nx.Defn.jit_apply(
        fn p, s ->
          Nx.Defn.grad(s, fn s2 ->
            out = predict_fn.(p, s2)
            lg = elem(out, head_idx)

            onehot =
              lg
              |> Nx.Defn.Kernel.stop_grad()
              |> Nx.argmax(axis: 1)
              |> Nx.new_axis(1)
              |> then(&Nx.equal(Nx.iota(Nx.shape(lg), axis: 1), &1))
              |> Nx.as_type(Nx.type(lg))

            Nx.sum(Nx.multiply(lg, onehot))
          end)
        end,
        [params, states],
        compiler: EXLA
      )

    grads
    |> Nx.multiply(states)
    |> Nx.abs()
    |> Nx.sum(axes: [1])
  end

  @doc """
  Aggregate per-dim saliency `{n, embed}` into `%{group => {n} share}` —
  each group's fraction of total saliency per row.
  """
  def group_shares(sal, dim_groups) do
    total = Nx.sum(sal, axes: [1]) |> Nx.max(1.0e-9)

    Map.new(dim_groups, fn {name, idx} ->
      share =
        case idx do
          [] ->
            Nx.broadcast(0.0, Nx.shape(total))

          _ ->
            sal
            |> Nx.take(Nx.tensor(idx), axis: 1)
            |> Nx.sum(axes: [1])
            |> Nx.divide(total)
        end

      {name, share}
    end)
  end

  # ============================================================================
  # Reference states for dim discovery
  # ============================================================================

  defp reference_state do
    %GameState{
      frame: 1000,
      stage: 32,
      players: %{1 => reference_player(10.0, 1.0), 2 => reference_player(-15.0, 1.0)},
      projectiles: [],
      menu_state: nil
    }
  end

  defp reference_player(x, facing) do
    %Player{
      character: 16,
      x: x,
      y: 0.0,
      percent: 42.0,
      stock: 3,
      facing: facing,
      action: 14,
      action_frame: 1,
      invulnerable: false,
      jumps_left: 1,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  defp update_player(gs, port, changes) do
    %{gs | players: Map.update!(gs.players, port, &struct(&1, changes))}
  end

  defp neutral_controller do
    %ExPhil.Bridge.ControllerState{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0,
      button_a: false,
      button_b: false,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false
    }
  end

  defp pressed_controller do
    %ExPhil.Bridge.ControllerState{
      main_stick: %{x: 0.9, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0,
      button_a: true,
      button_b: false,
      button_x: true,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false
    }
  end
end
