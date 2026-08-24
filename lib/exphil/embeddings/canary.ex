defmodule ExPhil.Embeddings.Canary do
  @moduledoc """
  Train-vs-live embedding fingerprint (GUARDS_BACKLOG #1).

  One canonical synthetic gamestate, embedded through the BATCHED path
  (training truth) at checkpoint save and through the LIVE path at
  Agent load. Any divergence — id spaces, block order, scaling,
  gating, config drift — raises at load instead of silently skewing
  play. Kills the class that produced the base-block reversal, the af
  convention saga, the 288-vs-336 layout mismatch, and the 08-24
  stage-as-"other" bug (months of every live game embedding a wrong
  stage flag, invisible to val_loss).

  Runtime values that legitimately differ per session (delay_id, queue
  contents, name_id) are PINNED to fixed canary values on both sides —
  the fingerprint measures configuration and code, not session state.

  Values are rounded to 1e-4 before comparison to absorb backend float
  wobble; the full rounded vector is stored in checkpoint metadata
  (a few hundred floats) so mismatches can name the divergent dims.
  """

  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Embeddings.Game, as: GameEmbed

  @canary_delay_id 0

  @doc """
  The canonical gamestate: FoD (external 2) with stage internals set,
  distinct player positions/actions/characters, a projectile-free
  simple state that still exercises every config-gated family.
  """
  def gamestate do
    %GameState{
      frame: 1234,
      stage: 2,
      menu_state: 2,
      players: %{1 => player(1), 2 => player(2)},
      projectiles: [],
      distance: 42.5,
      fod_platform_left: 21.5,
      fod_platform_right: 27.0,
      stadium_type: nil
    }
  end

  defp player(port) do
    %Player{
      character: port * 2,
      x: 10.0 * port,
      y: 5.0 * port,
      percent: 12.0 * port,
      stock: 4 - port,
      facing: if(port == 1, do: 1, else: -1),
      action: 14 + port,
      action_frame: 3.0,
      invulnerable: false,
      jumps_left: port,
      on_ground: true,
      shield_strength: 55.0,
      speed_air_x_self: 0.1,
      speed_ground_x_self: 0.2,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_y_self: -0.3
    }
  end

  @doc "Fingerprint via the BATCHED (training) path."
  def fingerprint_batched(config) do
    [gamestate()]
    |> GameEmbed.embed_states_fast(1, config: config, delay_id: @canary_delay_id)
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> Nx.squeeze(axes: [0])
    |> rounded()
  end

  @doc "Fingerprint via the LIVE (agent) path."
  def fingerprint_live(config) do
    gamestate()
    |> GameEmbed.embed(nil, 1,
      config: config,
      delay_id: @canary_delay_id,
      queue_controllers: []
    )
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> rounded()
  end

  @doc """
  Compare a stored fingerprint against a live one. `:ok` or
  `{:error, report}` naming size mismatch or the first divergent dims.
  """
  def compare(stored, live, tolerance \\ 1.0e-3)

  def compare(stored, live, _tolerance) when length(stored) != length(live) do
    {:error, {:size_mismatch, length(stored), length(live)}}
  end

  def compare(stored, live, tolerance) do
    divergent =
      stored
      |> Enum.zip(live)
      |> Enum.with_index()
      |> Enum.filter(fn {{a, b}, _i} -> abs(a - b) > tolerance end)

    case divergent do
      [] ->
        :ok

      list ->
        {:error,
         {:divergent_dims, length(list),
          Enum.take(list, 5) |> Enum.map(fn {{a, b}, i} -> {i, a, b} end)}}
    end
  end

  defp rounded(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&Float.round(&1 * 1.0, 4))
  end
end
