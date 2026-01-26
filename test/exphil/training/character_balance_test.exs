defmodule ExPhil.Training.CharacterBalanceTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.CharacterBalance

  describe "compute_weights/2" do
    test "rare characters get higher weights" do
      counts = %{mewtwo: 50000, ganondorf: 30000, link: 10000}
      weights = CharacterBalance.compute_weights(counts)

      # Link (rarest) should have highest weight
      assert weights.link >= weights.ganondorf
      assert weights.ganondorf >= weights.mewtwo
    end

    test "rarest character gets weight 1.0" do
      counts = %{fox: 100_000, link: 1000}
      weights = CharacterBalance.compute_weights(counts)

      assert weights.link == 1.0
      assert weights.fox < 1.0
    end

    test "handles single character" do
      counts = %{mewtwo: 10000}
      weights = CharacterBalance.compute_weights(counts)

      assert weights.mewtwo == 1.0
    end

    test "respects min_weight option" do
      counts = %{fox: 1_000_000, link: 100}
      weights = CharacterBalance.compute_weights(counts, min_weight: 0.01)

      # Fox would be 0.0001 without min_weight
      assert weights.fox >= 0.01
    end

    test "respects max_weight option" do
      counts = %{fox: 100, link: 10000}
      weights = CharacterBalance.compute_weights(counts, max_weight: 5.0)

      # Fox would be 100x without max_weight
      assert weights.fox <= 5.0
    end
  end

  describe "extract_character/1" do
    test "extracts from regular frame" do
      frame = %{
        game_state: %{
          players: %{
            1 => %{character: :mewtwo}
          }
        }
      }

      assert CharacterBalance.extract_character(frame) == :mewtwo
    end

    test "extracts from sequence frame" do
      frame = %{
        sequence: [
          %{game_state: %{players: %{1 => %{character: :ganondorf}}}}
        ]
      }

      assert CharacterBalance.extract_character(frame) == :ganondorf
    end

    test "returns nil for missing character" do
      assert CharacterBalance.extract_character(%{}) == nil
    end

    test "handles integer character IDs" do
      frame = %{
        game_state: %{
          players: %{
            # Mewtwo ID
            0 => %{character: 10}
          }
        }
      }

      assert CharacterBalance.extract_character(frame) == :mewtwo
    end
  end

  describe "count_characters/1" do
    test "counts frames by character" do
      frames = [
        %{game_state: %{players: %{1 => %{character: :mewtwo}}}},
        %{game_state: %{players: %{1 => %{character: :mewtwo}}}},
        %{game_state: %{players: %{1 => %{character: :link}}}}
      ]

      counts = CharacterBalance.count_characters(frames)

      assert counts.mewtwo == 2
      assert counts.link == 1
    end

    test "ignores frames with no character" do
      frames = [
        %{game_state: %{players: %{1 => %{character: :mewtwo}}}},
        # No character
        %{},
        # Empty players
        %{game_state: %{players: %{}}}
      ]

      counts = CharacterBalance.count_characters(frames)

      assert map_size(counts) == 1
      assert counts.mewtwo == 1
    end
  end

  describe "frame_weights/2" do
    test "assigns weights based on character" do
      frames = [
        %{game_state: %{players: %{1 => %{character: :mewtwo}}}},
        %{game_state: %{players: %{1 => %{character: :link}}}}
      ]

      weights = %{mewtwo: 0.5, link: 2.0}
      frame_weights = CharacterBalance.frame_weights(frames, weights)

      assert frame_weights == [0.5, 2.0]
    end

    test "uses default weight for unknown characters" do
      frames = [
        %{game_state: %{players: %{1 => %{character: :unknown_char}}}}
      ]

      weights = %{mewtwo: 0.5}
      frame_weights = CharacterBalance.frame_weights(frames, weights)

      # Default weight is 1.0
      assert frame_weights == [1.0]
    end
  end

  describe "weighted_sample/2" do
    test "returns correct number of samples" do
      weights = [1.0, 1.0, 1.0, 1.0]
      samples = CharacterBalance.weighted_sample(weights, 3)

      assert length(samples) == 3
    end

    test "samples are valid indices" do
      weights = [1.0, 2.0, 3.0]
      samples = CharacterBalance.weighted_sample(weights, 10)

      assert Enum.all?(samples, &(&1 >= 0 and &1 < 3))
    end

    test "higher weights get sampled more often" do
      # Weight index 2 much higher than others
      weights = [0.1, 0.1, 10.0]
      samples = CharacterBalance.weighted_sample(weights, 100)

      # Index 2 should appear most frequently
      freq = Enum.frequencies(samples)
      assert Map.get(freq, 2, 0) > Map.get(freq, 0, 0)
      assert Map.get(freq, 2, 0) > Map.get(freq, 1, 0)
    end

    test "handles all-zero weights gracefully" do
      weights = [0.0, 0.0, 0.0]
      samples = CharacterBalance.weighted_sample(weights, 3)

      # Should fallback to uniform random
      assert length(samples) == 3
      assert Enum.all?(samples, &(&1 >= 0 and &1 < 3))
    end
  end

  describe "format_distribution/2" do
    test "formats character distribution" do
      counts = %{mewtwo: 500, link: 500}
      weights = %{mewtwo: 1.0, link: 1.0}

      lines = CharacterBalance.format_distribution(counts, weights)

      assert length(lines) == 2
      assert Enum.all?(lines, &String.contains?(&1, "50.0%"))
    end
  end
end
