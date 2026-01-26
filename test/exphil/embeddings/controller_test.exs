defmodule ExPhil.Embeddings.ControllerTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Controller
  alias ExPhil.Bridge.ControllerState

  describe "default_config/0" do
    test "returns config with sensible defaults" do
      config = Controller.default_config()

      assert config.axis_buckets == 16
      assert config.shoulder_buckets == 4
    end
  end

  describe "embedding_size/1" do
    test "computes correct size for default config" do
      config = Controller.default_config()
      size = Controller.embedding_size(config)

      # 8 buttons + 2*(16+1) main + 2*(16+1) c + (4+1) shoulder
      # = 8 + 34 + 34 + 5 = 81
      expected = 8 + 2 * 17 + 2 * 17 + 5
      assert size == expected
    end

    test "scales with bucket counts" do
      small_config = %Controller{axis_buckets: 8, shoulder_buckets: 2}
      large_config = %Controller{axis_buckets: 32, shoulder_buckets: 8}

      small_size = Controller.embedding_size(small_config)
      large_size = Controller.embedding_size(large_config)

      assert large_size > small_size
    end
  end

  describe "continuous_embedding_size/0" do
    test "returns expected size" do
      size = Controller.continuous_embedding_size()

      # 8 buttons + 2 main + 2 c + 1 shoulder = 13
      assert size == 13
    end
  end

  describe "embed_continuous/1" do
    test "handles nil controller" do
      result = Controller.embed_continuous(nil)

      assert Nx.shape(result) == {Controller.continuous_embedding_size()}
      assert Nx.to_flat_list(result) |> Enum.all?(&(&1 == 0.0))
    end

    test "embeds controller state" do
      cs = %ControllerState{
        # Full right, neutral Y
        main_stick: %{x: 1.0, y: 0.0},
        # Neutral
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.5,
        r_shoulder: 0.0,
        button_a: true,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      result = Controller.embed_continuous(cs)

      assert Nx.shape(result) == {13}

      values = Nx.to_flat_list(result)

      # Button A should be 1.0
      assert Enum.at(values, 0) == 1.0

      # Main stick X = 1.0, shifted to [-1, 1] = 1.0
      assert_in_delta Enum.at(values, 8), 1.0, 0.001

      # Main stick Y = 0.0, shifted = -1.0
      assert_in_delta Enum.at(values, 9), -1.0, 0.001
    end

    test "encodes all button states" do
      cs = %ControllerState{
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: true,
        button_b: true,
        button_x: true,
        button_y: true,
        button_z: true,
        button_l: true,
        button_r: true,
        button_d_up: true
      }

      result = Controller.embed_continuous(cs)
      buttons = Nx.to_flat_list(result) |> Enum.take(8)

      # All buttons should be 1.0
      assert buttons == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    end
  end

  describe "discretize_axis/2" do
    test "discretizes 0.0 to bucket 0" do
      assert Controller.discretize_axis(0.0, 16) == 0
    end

    test "discretizes 1.0 to max bucket" do
      assert Controller.discretize_axis(1.0, 16) == 16
    end

    test "discretizes 0.5 to middle bucket" do
      # 0.5 * 16 + 0.5 = 8.5, truncated to 8
      assert Controller.discretize_axis(0.5, 16) == 8
    end

    test "clamps out-of-range values" do
      assert Controller.discretize_axis(-0.5, 16) == 0
      assert Controller.discretize_axis(1.5, 16) == 16
    end

    test "handles nil gracefully" do
      assert Controller.discretize_axis(nil, 16) == 0
    end
  end

  describe "undiscretize_axis/2" do
    test "converts bucket back to continuous value" do
      assert Controller.undiscretize_axis(0, 16) == 0.0
      assert Controller.undiscretize_axis(16, 16) == 1.0
      assert Controller.undiscretize_axis(8, 16) == 0.5
    end

    test "is inverse of discretize for bucket centers" do
      for bucket <- 0..16 do
        continuous = Controller.undiscretize_axis(bucket, 16)
        # Discretizing should give back same bucket (or close)
        back = Controller.discretize_axis(continuous, 16)
        assert abs(back - bucket) <= 1
      end
    end
  end

  describe "embed_discrete/2" do
    test "handles nil controller" do
      config = Controller.default_config()
      result = Controller.embed_discrete(nil, config)

      assert Nx.shape(result) == {Controller.embedding_size(config)}
    end

    test "creates one-hot encoded sticks" do
      cs = %ControllerState{
        # Left, Up
        main_stick: %{x: 0.0, y: 1.0},
        # Neutral
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      config = Controller.default_config()
      result = Controller.embed_discrete(cs, config)

      assert is_struct(result, Nx.Tensor)
      assert Nx.shape(result) == {Controller.embedding_size(config)}
    end
  end

  describe "component_indices/1" do
    test "returns correct index ranges" do
      config = Controller.default_config()
      indices = Controller.component_indices(config)

      assert indices.buttons == {0, 8}
      assert indices.main_x == {8, 8 + 17}
      assert indices.main_y == {8 + 17, 8 + 34}
    end

    test "indices are contiguous" do
      config = Controller.default_config()
      indices = Controller.component_indices(config)

      # Check that each component starts where the previous ends
      {_, buttons_end} = indices.buttons
      {main_x_start, main_x_end} = indices.main_x
      {main_y_start, main_y_end} = indices.main_y
      {c_x_start, c_x_end} = indices.c_x
      {c_y_start, c_y_end} = indices.c_y
      {shoulder_start, _shoulder_end} = indices.shoulder

      assert main_x_start == buttons_end
      assert main_y_start == main_x_end
      assert c_x_start == main_y_end
      assert c_y_start == c_x_end
      assert shoulder_start == c_y_end
    end
  end

  describe "decode/2" do
    test "converts samples back to ControllerState" do
      samples = %{
        # 0.5
        main_x: 8,
        # 0.5
        main_y: 8,
        # 0.0
        c_x: 0,
        # 1.0
        c_y: 16,
        # 0.5
        shoulder: 2,
        buttons: %{
          a: true,
          b: false,
          x: false,
          y: true,
          z: false,
          l: false,
          r: false,
          d_up: false
        }
      }

      cs = Controller.decode(samples)

      assert %ControllerState{} = cs
      assert cs.button_a == true
      assert cs.button_y == true
      assert cs.button_b == false
      assert_in_delta cs.main_stick.x, 0.5, 0.01
      assert_in_delta cs.c_stick.y, 1.0, 0.01
    end

    test "handles custom config" do
      samples = %{
        # 0.5 with 8 buckets
        main_x: 4,
        main_y: 4,
        c_x: 4,
        c_y: 4,
        shoulder: 1,
        buttons: %{
          a: false,
          b: false,
          x: false,
          y: false,
          z: false,
          l: false,
          r: false,
          d_up: false
        }
      }

      config = %Controller{axis_buckets: 8, shoulder_buckets: 2}
      cs = Controller.decode(samples, config)

      assert_in_delta cs.main_stick.x, 0.5, 0.01
      assert_in_delta cs.l_shoulder, 0.5, 0.01
    end
  end
end
