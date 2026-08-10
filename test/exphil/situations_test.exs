defmodule ExPhil.SituationsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.GameState
  alias ExPhil.Bridge.Player
  alias ExPhil.Situations

  @fd 32
  @ys 8

  defp player(overrides) do
    struct(
      %Player{
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        action: 14,
        action_frame: 1,
        facing: 1,
        on_ground: true,
        jumps_left: 2,
        shield_strength: 60.0,
        hitstun_frames_left: 0
      },
      overrides
    )
  end

  defp gs(own, opp, opts \\ []) do
    %GameState{
      frame: Keyword.get(opts, :frame, 500),
      stage: Keyword.get(opts, :stage, @fd),
      players: %{1 => own, 2 => opp}
    }
  end

  defp labels(own, opp, opts \\ []) do
    {_ctx, set} = Situations.fold(Situations.new_context(), gs(own, opp, opts), 1)
    set
  end

  test "registry fits a u64 and bits round-trip" do
    assert length(Situations.labels()) <= 64

    set = MapSet.new([:neutral, :onstage_center, :edge_danger, :warmup_frames])
    assert set |> Situations.to_mask() |> Situations.from_mask() == set
  end

  test "two standing players at center: neutral umbrella + center geometry" do
    set = labels(player(x: 10.0), player(x: -20.0))
    assert :neutral in set
    assert :onstage_center in set
    refute :advantage in set
    refute :disadvantage in set
    refute :offstage in set
  end

  test "umbrella exclusivity: a trade (both in hitstun) gets no parent" do
    set = labels(player(hitstun_frames_left: 10), player(hitstun_frames_left: 10))
    refute :neutral in set
    refute :advantage in set
    refute :disadvantage in set
    assert :in_hitstun in set
  end

  test "edge_danger: dashing toward the FD edge inside the margin" do
    set = labels(player(x: 75.0, action: 0x14, speed_ground_x_self: 2.0), player(x: -20.0))
    assert :edge_danger in set
    assert :onstage_corner in set

    # dashing AWAY is not danger
    set = labels(player(x: 75.0, action: 0x14, speed_ground_x_self: -2.0), player(x: -20.0))
    refute :edge_danger in set
  end

  test "per-stage geometry: x=40 is safe on FD, danger zone on YS" do
    # x=40 discriminates the external/internal id collision: YS edge 56
    # -> danger at |x| > 36 (real YS), while the collision's FoD edge
    # 63.35 -> danger only past 43.35
    fox = player(x: 40.0, action: 0x14, speed_ground_x_self: 2.0)
    refute :edge_danger in labels(fox, player(x: -20.0), stage: @fd)
    assert :edge_danger in labels(fox, player(x: -20.0), stage: @ys)
  end

  test "offstage disadvantage cluster: being_edgeguarded, recovery, resources" do
    own = player(x: 100.0, y: -30.0, on_ground: false, jumps_left: 0)
    set = labels(own, player(x: 40.0))

    assert :offstage in set
    assert :being_edgeguarded in set
    assert :recovery_low in set
    assert :below_ledge in set
    assert :resource_exhausted in set
    refute :edgeguard in set
  end

  test "edgeguard is the mirror and excludes ledge (that's ledge_trap)" do
    opp_off = player(x: -100.0, y: -20.0, on_ground: false)
    set = labels(player(x: -60.0), opp_off)
    assert :edgeguard in set

    opp_ledge = player(x: -90.0, y: -8.0, on_ground: false, action: 253)
    set = labels(player(x: -60.0), opp_ledge)
    refute :edgeguard in set
    assert :ledge_trap in set
    assert :ledge_occupied_by_opp in set
  end

  test "juggle and its mirror" do
    below = player(x: 0.0)
    above_in_hitstun = player(x: 5.0, y: 40.0, on_ground: false, hitstun_frames_left: 12)

    assert :juggle in labels(below, above_in_hitstun)
    assert :being_juggled in labels(above_in_hitstun, below)
  end

  test "tech_chase on opponent knockdown nearby" do
    set = labels(player(x: 10.0), player(x: 25.0, action: 184))
    assert :tech_chase in set
    assert :advantage in set
  end

  test "ledge_option_pending on CliffWait" do
    own = player(x: 88.0, y: -8.0, on_ground: false, action: 253)
    set = labels(own, player(x: 0.0))
    assert :ledge_option_pending in set
    assert :ledge_hang in set
  end

  test "shield pressure both directions and shield_low" do
    shielding = player(x: 0.0, action: 179, shield_strength: 12.0)
    attacker = player(x: 15.0)

    set = labels(attacker, shielding)
    assert :shield_pressure_ours in set

    set = labels(shielding, attacker)
    assert :shield_pressure_theirs in set
    assert :shield_low in set
  end

  test "combo_active and conversion_open open on a hit from neutral and persist" do
    ctx = Situations.new_context()
    opp_free = player(x: 20.0)
    opp_hit = player(x: 20.0, hitstun_frames_left: 15)
    own = player(x: 0.0)

    {ctx, set0} = Situations.fold(ctx, gs(own, opp_free, frame: 500), 1)
    refute :combo_active in set0

    {ctx, set1} = Situations.fold(ctx, gs(own, opp_hit, frame: 501), 1)
    assert :combo_active in set1
    assert :conversion_open in set1

    # gap frames keep the combo window alive (< 20 frames)
    {_ctx, set2} = Situations.fold(ctx, gs(own, opp_free, frame: 502), 1)
    assert :combo_active in set2
    assert :conversion_open in set2
  end

  test "approach and retreat from the distance window" do
    ctx = Situations.new_context()
    opp = player(x: 60.0)

    # 12 frames closing 3 units/frame
    {ctx, _} =
      Enum.reduce(0..11, {ctx, nil}, fn i, {c, _} ->
        Situations.fold(c, gs(player(x: i * 3.0), opp, frame: 500 + i), 1)
      end)

    {_ctx, set} = Situations.fold(ctx, gs(player(x: 36.0), opp, frame: 512), 1)
    assert :approach in set
    refute :retreat in set
  end

  test "post_kill_neutral after opponent stock drop" do
    ctx = Situations.new_context()
    own = player(x: 0.0)

    {ctx, _} = Situations.fold(ctx, gs(own, player(stock: 4), frame: 500), 1)
    {_ctx, set} = Situations.fold(ctx, gs(own, player(stock: 3), frame: 510), 1)
    assert :post_kill_neutral in set
  end

  test "game-flow flags: last stocks, percent lead/deficit, warmup" do
    set = labels(player(stock: 1, percent: 10.0), player(stock: 1, percent: 80.0), frame: 50)
    assert :last_stock_ours in set
    assert :last_stock_theirs in set
    assert :percent_lead in set
    refute :percent_deficit in set
    assert :warmup_frames in set
  end

  test "platform_underneath over a BF side platform, on_platform when standing on it" do
    # BF side platforms ~ y 27.2, x 20-57
    airborne_over = player(x: 40.0, y: 45.0, on_ground: false)
    set = labels(airborne_over, player(x: -20.0), stage: 31)
    assert :platform_underneath in set

    standing_on = player(x: 40.0, y: 27.2, on_ground: true)
    set = labels(standing_on, player(x: -20.0), stage: 31)
    assert :on_platform in set
  end

  test "execution windows" do
    assert :jc_window in labels(player(action: 24), player(x: 30.0))
    assert :shine_cancellable in labels(player(action: 360), player(x: 30.0))
  end

  test "label_states batch frontend returns one mask per state" do
    states = [gs(player(x: 0.0), player(x: 30.0)), gs(player(x: 5.0), player(x: 30.0))]
    masks = Situations.label_states(states, 1)
    assert length(masks) == 2
    assert Enum.all?(masks, &is_integer/1)
    assert :neutral in Situations.from_mask(hd(masks))
  end
end
