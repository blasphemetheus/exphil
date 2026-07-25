defmodule ExPhil.ConstantsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Constants

  doctest ExPhil.Constants

  describe "frame timing" do
    test "fps returns 60" do
      assert Constants.fps() == 60
    end

    test "max_game_frames is 8 minutes at 60fps" do
      assert Constants.max_game_frames() == 28_800
      assert Constants.max_game_frames() == 8 * 60 * 60
    end

    test "max_game_seconds is 8 minutes" do
      assert Constants.max_game_seconds() == 480
      assert Constants.max_game_seconds() == 8 * 60
    end

    test "online_frame_delay is 18" do
      assert Constants.online_frame_delay() == 18
    end
  end

  describe "action states" do
    test "num_actions is 399" do
      assert Constants.num_actions() == 399
    end

    test "default_action_embed_dim is 64" do
      assert Constants.default_action_embed_dim() == 64
    end
  end

  describe "action state IDs" do
    test "first_actionable is Wait (14)" do
      assert Constants.first_actionable() == 14
    end

    test "jumpsquat is KneeBend (24)" do
      assert Constants.jumpsquat() == 24
    end

    test "aerial_jump is 25, immediately after jumpsquat" do
      assert Constants.aerial_jump() == 25
      assert Constants.aerial_jump() == Constants.jumpsquat() + 1
    end

    test "reflector_ground covers 360..363" do
      assert Constants.reflector_ground() == 360..363
      assert Enum.to_list(Constants.reflector_ground()) == [360, 361, 362, 363]
    end

    test "reflector_air covers 365..368" do
      assert Constants.reflector_air() == 365..368
      assert Enum.to_list(Constants.reflector_air()) == [365, 366, 367, 368]
    end

    test "ground and air reflector ranges are disjoint" do
      ground = MapSet.new(Constants.reflector_ground())
      air = MapSet.new(Constants.reflector_air())

      assert MapSet.disjoint?(ground, air),
             "ground/air shine must stay distinguishable — multishine chain detection depends on it"
    end

    test "all action state IDs are within num_actions" do
      ids =
        [
          Constants.first_actionable(),
          Constants.jumpsquat(),
          Constants.aerial_jump()
        ] ++ Enum.to_list(Constants.reflector_ground()) ++ Enum.to_list(Constants.reflector_air())

      for id <- ids do
        assert id >= 0 and id < Constants.num_actions(),
               "action state #{id} outside 0..#{Constants.num_actions() - 1}"
      end
    end

    test "action state IDs are usable in guards and patterns via module attributes" do
      # Regression guard: Constants exposes plain functions, which are illegal
      # in guards/patterns. Consumers rely on compile-time attribute
      # assignment still yielding a literal. If that ever breaks, the modules
      # in eval/shine_chain.ex stop compiling — pin the mechanism here.
      defmodule GuardProbe do
        @jumpsquat ExPhil.Constants.jumpsquat()
        @reflector_ground ExPhil.Constants.reflector_ground()

        def family(a) when a in @reflector_ground, do: :ground_reflect
        def family(@jumpsquat), do: :jumpsquat
        def family(_), do: :other
      end

      assert GuardProbe.family(361) == :ground_reflect
      assert GuardProbe.family(24) == :jumpsquat
      assert GuardProbe.family(99) == :other
    end
  end

  describe "characters" do
    test "num_characters is 33" do
      assert Constants.num_characters() == 33
    end

    test "default_char_embed_dim is 64" do
      assert Constants.default_char_embed_dim() == 64
    end
  end

  describe "stages" do
    test "num_stages is 64" do
      assert Constants.num_stages() == 64
    end

    test "num_competitive_stages is 6" do
      assert Constants.num_competitive_stages() == 6
    end

    test "competitive_stage_ids has correct stage IDs" do
      ids = Constants.competitive_stage_ids()

      assert ids[:fountain_of_dreams] == 2
      assert ids[:pokemon_stadium] == 3
      assert ids[:yoshis_story] == 8
      assert ids[:dream_land] == 28
      assert ids[:battlefield] == 31
      assert ids[:final_destination] == 32
    end
  end

  describe "combat mechanics" do
    test "max_hitstun_frames is 120" do
      assert Constants.max_hitstun_frames() == 120
    end

    test "max_shieldstun_frames is 25" do
      assert Constants.max_shieldstun_frames() == 25
    end

    test "standard_action_frames is 60" do
      assert Constants.standard_action_frames() == 60
    end

    test "max_jumps is 6" do
      assert Constants.max_jumps() == 6
    end
  end

  describe "items" do
    test "link_bomb_timer is 180 (3 seconds)" do
      assert Constants.link_bomb_timer() == 180
      assert Constants.link_bomb_timer() == 3 * 60
    end
  end

  describe "controller" do
    test "default_stick_buckets is 17" do
      assert Constants.default_stick_buckets() == 17
    end

    test "shoulder_positions is 4" do
      assert Constants.shoulder_positions() == 4
    end

    test "num_buttons is 8" do
      assert Constants.num_buttons() == 8
    end
  end

  describe "neural network defaults" do
    test "default_hidden_sizes returns expected values" do
      sizes = Constants.default_hidden_sizes()
      assert sizes == [512, 256]
      assert is_list(sizes)
      assert Enum.all?(sizes, &is_integer/1)
    end

    test "tensor_alignment is 8" do
      assert Constants.tensor_alignment() == 8
    end
  end

  describe "normalize_frame/1" do
    test "normalizes to [0, 1]" do
      assert Constants.normalize_frame(0) == 0.0
      assert Constants.normalize_frame(14400) == 0.5
      assert Constants.normalize_frame(28800) == 1.0
    end

    test "caps at 1.0 for values over max" do
      assert Constants.normalize_frame(30000) == 1.0
      assert Constants.normalize_frame(100_000) == 1.0
    end

    test "handles negative values" do
      assert Constants.normalize_frame(-100) < 0.0
    end
  end

  describe "normalize_hitstun/1" do
    test "normalizes to [0, 1]" do
      assert Constants.normalize_hitstun(0) == 0.0
      assert Constants.normalize_hitstun(60) == 0.5
      assert Constants.normalize_hitstun(120) == 1.0
    end

    test "caps at 1.0 for values over max" do
      assert Constants.normalize_hitstun(200) == 1.0
    end
  end

  describe "normalize_action_progress/1" do
    test "normalizes to [0, 1]" do
      assert Constants.normalize_action_progress(0) == 0.0
      assert Constants.normalize_action_progress(30) == 0.5
      assert Constants.normalize_action_progress(60) == 1.0
    end

    test "caps at 1.0 for values over max" do
      assert Constants.normalize_action_progress(100) == 1.0
    end
  end

  describe "consistency" do
    test "max_game_frames equals fps * max_game_seconds" do
      assert Constants.max_game_frames() == Constants.fps() * Constants.max_game_seconds()
    end
  end
end
