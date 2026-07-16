defmodule ExPhil.Agents.MewtwoFairExpert do
  @moduledoc """
  Scripted Mewtwo fair-chain expert for DAgger-style relabeling.

  Drill: short-hop fair -> L-cancel -> repeat, with full-hop fair and
  double-jump fair variants (fixture: test/fixtures/replays/mewtwo_fair_chains.slp,
  a human recording vs a stationary Fox on FD).

  Same labeling protocol as `ExPhil.Agents.MultishineExpert` (replay landing
  convention, composes with `Data.shift_actions/2`, recovery taps keyed on the
  previously-landed input), with one structural upgrade the multishine drill
  didn't need: a **physics-keyed table**. SH fair, FH fair, and DJ fair all
  occupy action 66 (ATTACK_AIR_F) with the same action_frame values — what
  distinguishes the correct input (most importantly the L-cancel frame) is
  remaining airtime. So the table is hierarchical:

  1. **Fine key** `{action, af, grounded, jumps_left, height_bucket}` —
     disambiguates the jump variants by double-jump availability and height
     above the stage (FD ground is y=0; the drill assumes FD).
  2. **Coarse key** `{action, af, grounded}` — fallback where the fine table
     is sparse.
  3. **Recovery rules** — grounded: tap jump (Y — the recorder's jump button)
     to restart the drill; airborne falling near the ground: tap L (L-cancel
     insurance); airborne rising without an attack out: c-stick fair toward
     facing; otherwise neutral.

  Death/respawn states return `:skip`.
  """

  alias ExPhil.Bridge.ControllerState

  defstruct [:fine, :coarse]

  @type t :: %__MODULE__{fine: map(), coarse: map()}

  @fixture_path "test/fixtures/replays/mewtwo_fair_chains.slp"

  # Universal Melee action-state IDs
  @jump_states 25..28
  # ATTACK_AIR_N..ATTACK_AIR_LW — the only states where an airborne L press
  # L-cancels; anywhere else it AIRDODGES (observed live: stray airdodges
  # from an unconditional falling-near-ground L tap)
  @aerial_attacks 65..70
  @first_actionable 14

  # Height (y) below which a falling aerial should be L-cancelling: an input
  # landing within the next few frames will coincide with touchdown
  @lcancel_height 8.0

  # |x| beyond which a grounded jump-restart risks drifting off the stage
  # (FD edges at ±85.57) — walk back toward center instead. Observed live:
  # the jump-tap restart rule near the edge SD'd an under-trained policy.
  @edge_margin 60.0

  @doc """
  Build the expert from a fair-chain recording (defaults to the canonical
  fixture). Options: `:player_port` (default 1), `:min_count` (default 3
  fine / 4 coarse — drops one-off noise cells).
  """
  @spec from_fixture(String.t(), keyword()) :: t()
  def from_fixture(path \\ @fixture_path, opts \\ []) do
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)

    replay
    |> ExPhil.Data.Peppi.to_training_frames(
      player_port: Keyword.get(opts, :player_port, 1),
      opponent_port: Keyword.get(opts, :opponent_port, 2)
    )
    |> from_frames(opts)
  end

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)
    opp_port = Keyword.get(opts, :opponent_port, if(port == 1, do: 2, else: 1))
    min_count = Keyword.get(opts, :min_count, 3)

    usable =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.reject(fn f -> trunc(f.game_state.players[port].action) < @first_actionable end)

    %__MODULE__{
      fine: build_table(usable, port, &fine_key(&1, opp_of(&2, opp_port)), min_count),
      coarse: build_table(usable, port, fn p, _f -> coarse_key(p) end, max(min_count, 4))
    }
  end

  defp opp_of(frame, opp_port), do: frame.game_state.players[opp_port]

  @doc """
  Label a player state with the expert's input (landing convention).

  `prev` is the input that actually landed on the previous frame — recovery
  taps alternate against it so press EDGES are learnable. `opponent` (the
  other port's player) keys the distance features: the approach-fair drill
  teaches WHEN to swing, and that decision lives in the gap between the two
  characters — without it the table collapses "in range" and "across the
  stage" into one cell (observed live: metronome fairs at nothing).
  """
  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(expert, player, prev \\ nil, opponent \\ nil) do
    case label_traced(expert, player, prev, opponent) do
      {:ok, controller, _source} -> {:ok, controller}
      :skip -> :skip
    end
  end

  @doc """
  Like `label/4` but also returns which path produced the label:
  `:edge | :fine | :coarse | :approach | :jump_restart | :lcancel |
  :cstick_fair | :airborne_neutral`. Audit/coverage instrumentation
  (teacher-quality audit, 2026-07-15).
  """
  @spec label_traced(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t(), atom()} | :skip
  def label_traced(%__MODULE__{fine: fine, coarse: coarse}, player, prev \\ nil, opponent \\ nil) do
    cond do
      trunc(player.action) < @first_actionable ->
        :skip

      # Edge safety OVERRIDES the table: table keys deliberately exclude x,
      # so past the margin the table keeps serving center-stage answers —
      # including the fixture's natural fade-back drift — straight off the
      # edge (observed live: SD by jumping and drifting backwards).
      abs(player.x || 0.0) > @edge_margin ->
        {:ok, edge_recovery(player, prev), :edge}

      controller = fine[fine_key(player, opponent)] ->
        {:ok, controller, :fine}

      controller = coarse[coarse_key(player)] ->
        {:ok, controller, :coarse}

      true ->
        {controller, branch} = recovery_traced(player, prev, opponent)
        {:ok, controller, branch}
    end
  end

  @doc """
  Fraction of fixture frames where the expert's label matches the recorded
  buttons exactly (coverage diagnostic on expert-quality data).
  """
  @spec button_agreement(t(), [map()], keyword()) :: float()
  def button_agreement(%__MODULE__{} = expert, frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)

    scored =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.flat_map(fn f ->
        case label(expert, f.game_state.players[port]) do
          {:ok, c} -> [{c, f.controller}]
          :skip -> []
        end
      end)

    case scored do
      [] -> 0.0
      pairs -> Enum.count(pairs, fn {a, b} -> buttons(a) == buttons(b) end) / length(pairs)
    end
  end

  # -- Keys -------------------------------------------------------------------

  defp fine_key(player, opponent) do
    {trunc(player.action), trunc(player.action_frame), player.on_ground,
     player.jumps_left || 0, height_bucket(player.y), distance_bucket(player, opponent),
     side_bucket(player, opponent)}
  end

  # Facing-relative side of the opponent: 1 = in front, -1 = behind, :na
  # when unknown. The key was previously UNSIGNED (abs distance only), so
  # the table could not distinguish opponent-in-front from opponent-behind
  # — the policy faithfully learned direction-blind fairs (the
  # fair-in-place-while-Fox-stands-behind loop; interp case #3,
  # 2026-07-15). Signing the key also unlocks the turnaround exemplars
  # already present in the recordings, which unsigned buckets averaged
  # away. Sparsity cost: keys double; coarse-key fallback covers gaps.
  defp side_bucket(_player, nil), do: :na

  defp side_bucket(player, opponent) do
    dx = (opponent.x || 0.0) - (player.x || 0.0)
    facing = if (player.facing || 1) >= 0, do: 1, else: -1
    if dx * facing >= 0, do: 1, else: -1
  end

  # Horizontal gap to the opponent, in 15-unit buckets capped at 4 (60+
  # units = "across the stage"). :na when the opponent is unknown (some
  # tests) — a distinct key, never colliding with measured buckets.
  defp distance_bucket(_player, nil), do: :na

  defp distance_bucket(player, opponent) do
    dist = abs((player.x || 0.0) - (opponent.x || 0.0))
    min(div(trunc(dist), 15), 4)
  end

  defp coarse_key(player) do
    {trunc(player.action), trunc(player.action_frame), player.on_ground}
  end

  # 6-unit buckets, capped: 0 = at/below stage level, 9 = 54+ units up
  defp height_bucket(y) do
    y
    |> max(0.0)
    |> then(&trunc(&1 / 6))
    |> min(9)
  end

  # -- Recovery rules -----------------------------------------------------------

  # Past the margin nothing matters but getting back over stage — label()
  # short-circuits here BEFORE the table (whose keys exclude x). Steer
  # toward center; keep the L-cancel tap when landing mid-aerial.
  defp edge_recovery(player, prev) do
    action = trunc(player.action)
    falling? = (player.speed_y_self || 0.0) < 0.0
    toward_center = if (player.x || 0.0) > 0, do: 0.0, else: 1.0
    steer = %{neutral() | main_stick: %{x: toward_center, y: 0.5}}

    if not player.on_ground and falling? and (player.y || 0.0) < @lcancel_height and
         action in @aerial_attacks and not held?(prev, :button_l) do
      %{steer | button_l: true}
    else
      steer
    end
  end

  # Center-stage table misses only — edge states never reach here
  defp recovery_traced(player, prev, opponent) do
    action = trunc(player.action)
    falling? = (player.speed_y_self || 0.0) < 0.0

    cond do
      # Opponent BEHIND: turn/walk toward them before anything else. The old
      # jump-restart here taught direction-blind metronome jumping (pathology
      # #4: the grounded jump default was ~31% of ALL labels, 2026-07-15
      # coverage audit) and left (behind, adjacent) states with no exit
      # (observed live: 75s mutual-idle deadlock). Also gives case #3's
      # turnaround exemplars a default-path ally.
      player.on_ground and opponent != nil and side_bucket(player, opponent) == -1 ->
        toward = if (opponent.x || 0.0) > (player.x || 0.0), do: 1.0, else: 0.0
        {%{neutral() | main_stick: %{x: toward, y: 0.5}}, :turn_toward}

      # Not at fair spacing (in front): walk there — approach replaces the
      # old 35-unit boundary; jump-restarts only happen AT spacing now.
      player.on_ground and opponent != nil and
          abs((player.x || 0.0) - (opponent.x || 0.0)) > 30.0 ->
        toward = if (opponent.x || 0.0) > (player.x || 0.0), do: 1.0, else: 0.0
        {%{neutral() | main_stick: %{x: toward, y: 0.5}}, :approach}

      # Grounded at spacing with the opponent in front (or opponent unknown —
      # some tests): restart the drill with a jump (recorder uses Y).
      player.on_ground ->
        if held?(prev, :button_y),
          do: {neutral(), :jump_restart},
          else: {tap(:button_y), :jump_restart}

      # Falling close to the stage IN AN AERIAL ATTACK: L-cancel insurance —
      # an L landing in the next couple of frames covers the touchdown
      # window. Outside attack states an airborne L is an airdodge, not a
      # cancel — never tap it there.
      falling? and (player.y || 0.0) < @lcancel_height and action in @aerial_attacks ->
        if held?(prev, :button_l),
          do: {neutral(), :lcancel},
          else: {tap(:button_l), :lcancel}

      # Rising in a jump without an attack out: c-stick fair toward facing
      # (c-stick avoids main-stick drift side effects).
      action in @jump_states ->
        {cstick_fair(player, prev), :cstick_fair}

      # Anything else airborne (riding out the fair, tumble, specials): let
      # the table/state evolve; neutral is safe.
      true ->
        {neutral(), :airborne_neutral}
    end
  end

  defp cstick_fair(player, prev) do
    if cstick_deflected?(prev) do
      neutral()
    else
      cx = if (player.facing || 1) > 0, do: 1.0, else: 0.0
      %{neutral() | c_stick: %{x: cx, y: 0.5}}
    end
  end

  defp cstick_deflected?(nil), do: false

  defp cstick_deflected?(prev) do
    case prev.c_stick do
      %{x: x} -> abs(x - 0.5) > 0.2
      _ -> false
    end
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp tap(button), do: neutral() |> Map.put(button, true)
  defp neutral, do: ControllerState.neutral()

  # -- Table construction -------------------------------------------------------

  # key_fn receives (player, frame) so fine keys can reach the opponent
  defp build_table(frames, port, key_fn, min_count) do
    frames
    |> Enum.group_by(fn f -> key_fn.(f.game_state.players[port], f) end)
    |> Enum.filter(fn {_key, group} -> length(group) >= min_count end)
    |> Map.new(fn {key, group} -> {key, modal_controller(group)} end)
  end

  defp modal_controller(group) do
    modal_sig =
      group
      |> Enum.frequencies_by(&signature(&1.controller))
      |> Enum.max_by(fn {_sig, count} -> count end)
      |> elem(0)

    Enum.find(group, &(signature(&1.controller) == modal_sig)).controller
  end

  defp signature(c) do
    {buttons(c), grid(c.main_stick.x), grid(c.main_stick.y), grid(c.c_stick.x), grid(c.c_stick.y)}
  end

  defp buttons(c) do
    {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r}
  end

  defp grid(v), do: round(v * 20)
end
