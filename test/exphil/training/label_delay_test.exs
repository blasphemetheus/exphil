defmodule ExPhil.Training.LabelDelayTest do
  use ExUnit.Case, async: false

  alias ExPhil.Data.LabelConvention
  alias ExPhil.Training.{Config, LabelDelay, Pipeline, Streaming}

  test "all aliases resolve identically and explicit zero cannot hide a conflict" do
    for flag <- ["--label-delay", "--frame-delay", "--action-delay"] do
      config = Config.parse_args([flag, "2", "--backbone", "gru"])
      assert Keyword.take(config, LabelDelay.keys()) |> Keyword.values() |> Enum.uniq() == [2]
      assert LabelConvention.reaction_delay(Map.new(config)) == 2
    end

    assert Config.parse_args([])[:label_delay] == 0

    assert Config.defaults()
           |> Keyword.put(:action_delay, 2)
           |> LabelDelay.resolve!()
           |> Keyword.fetch!(:label_delay) == 2

    assert_raise ArgumentError, ~r/Conflicting/, fn ->
      Config.parse_args(["--frame-delay", "0", "--action-delay", "2"])
    end

    assert_raise ArgumentError, ~r/Conflicting/, fn ->
      LabelDelay.resolve!(label_delay: 2, frame_delay: 0)
    end

    for value <- [-1, "2", 1.5] do
      assert_raise ArgumentError, ~r/nonnegative integer/, fn ->
        LabelDelay.resolve!(action_delay: value)
      end
    end
  end

  @tag :tmp_dir
  test "CLI overrides YAML as one delay value, including explicit zero", %{tmp_dir: directory} do
    path = Path.join(directory, "config.yaml")
    File.write!(path, "action_delay: 3\n")
    assert Config.parse_args(["--config", path])[:label_delay] == 3
    opts = Config.parse_args(["--config", path, "--preset", "quick", "--frame-delay", "0"])
    assert opts[:label_delay] == 0
    assert opts[:action_delay] == 0
    {:ok, loaded} = Config.load_with_yaml(path, ["--frame-delay", "2"])
    assert loaded[:label_delay] == 2

    File.write!(path, "frame_delay: 0\naction_delay: 3\n")
    assert_raise ArgumentError, ~r/Conflicting/, fn -> Config.parse_args(["--config", path]) end
  end

  @tag :tmp_dir
  test "resume converts legacy numbering once, while explicit CLI can override it", %{
    tmp_dir: directory
  } do
    path = Path.join(directory, "old.axon")
    File.write!(path, :erlang.term_to_binary(%{config: %{frame_delay: 3, embed_size: 32}}))
    opts = Config.parse_args(["--resume", path])
    assert opts[:label_delay] == 2
    assert opts[:action_delay] == 2
    assert Config.parse_args(["--resume", path, "--label-delay", "0"])[:label_delay] == 0

    File.write!(path, :erlang.term_to_binary(%{config: %{frame_delay: 0, embed_size: 32}}))
    assert_raise ArgumentError, ~r/leaked labels/, fn -> Config.parse_args(["--resume", path]) end
    assert Config.parse_args(["--resume", path, "--label-delay", "0"])[:label_delay] == 0
  end

  test "legacy checkpoint and live delay numbering remain unchanged" do
    assert LabelConvention.reaction_delay(%{frame_delay: 3}) == 2

    assert LabelConvention.reaction_delay(%{"label_delay" => 2, "label_convention" => "causal"}) ==
             2

    assert LabelConvention.delay_id(3, %{label_convention: :causal}) == 2
    assert LabelConvention.delay_id(3, %{}) == 3

    assert LabelConvention.train_reaction_delays(%{
             label_convention: :causal,
             train_delays: [1, 2]
           }) == [1, 2]
  end

  @tag :tmp_dir
  test "exports and full resume keep the executed reaction delay", %{tmp_dir: directory} do
    alias ExPhil.Training.{Checkpoint, Imitation}
    trainer = Imitation.new(embed_size: 32, hidden_sizes: [8], action_delay: 2)
    path = Path.join(directory, "delay.axon")
    :ok = Imitation.save_checkpoint(trainer, path)
    fresh = Imitation.new(embed_size: 32, hidden_sizes: [8], label_delay: 0)
    {:ok, resumed} = Imitation.load_checkpoint(fresh, path)
    assert resumed.config[:label_delay] == 0
    assert resumed.config[:action_delay] == 0
    assert resumed.config[:label_convention] == :causal

    export = Path.join(directory, "delay_policy.bin")
    :ok = Imitation.export_policy(trainer, export)
    {:ok, saved} = Checkpoint.load_policy(export)
    assert saved.config[:label_delay] == 2
    assert saved.config[:train_delays] == [2]
    assert ExPhil.Training.Comparability.key(saved.config).label_delay == 2

    augmented = %{
      trainer
      | config:
          Map.merge(
            trainer.config,
            %{frame_delay_augment: true, frame_delay_min: 1, frame_delay_max: 3}
          )
    }

    :ok = Imitation.export_policy(augmented, export)
    {:ok, saved} = Checkpoint.load_policy(export)
    assert saved.config[:train_delays] == [3, 4, 5]
  end

  test "unsupported augmentation and precomputed corpus delay fail before file loading" do
    assert_raise ArgumentError, ~r/augmentation/, fn ->
      Pipeline.setup(temporal: true, frame_delay_augment: true)
    end

    assert_raise ArgumentError, ~r/corpus labels/, fn ->
      Pipeline.setup(corpus: "unused", label_delay: 2)
    end

    assert_raise ArgumentError, ~r/Conflicting/, fn ->
      Pipeline.setup(frame_delay: 0, action_delay: 2)
    end
  end

  @tag :nif
  test "streaming aliases produce identical successor targets" do
    fixture = "test/fixtures/replays/fox_multishine.slp"

    results =
      for key <- LabelDelay.keys() do
        {:ok, frames, _errors} =
          Streaming.parse_chunk([fixture], [{key, 2}, {:show_progress, false}])

        assert length(frames) > 10
        Enum.map(frames, &{&1.game_state.frame, &1.controller})
      end

    assert Enum.uniq(results) |> length() == 1
  end

  @tag :nif
  @tag :tmp_dir
  test "standard batches shift once and match streaming targets for every alias", %{
    tmp_dir: directory
  } do
    fixture = "test/fixtures/replays/fox_multishine.slp"
    File.cp!(fixture, Path.join(directory, "first.slp"))
    File.cp!(fixture, Path.join(directory, "second.slp"))

    {:ok, expected, _errors} =
      Streaming.parse_chunk([fixture, fixture], label_delay: 2, show_progress: false)

    for key <- LabelDelay.keys() do
      opts = [
        {key, 2},
        {:replays, directory},
        {:cache_embeddings, false},
        {:val_split, 0},
        {:batch_size, 32},
        {:temporal, false}
      ]

      {:ok, pipeline} = Pipeline.setup(opts)

      assert Enum.map(pipeline.train_dataset.frames, &{&1.game_state.frame, &1.controller}) ==
               Enum.map(expected, &{&1.game_state.frame, &1.controller})

      {stream, _count} = Pipeline.batch_stream(pipeline, shuffle: false, drop_last: false)
      actual = Enum.map(stream, & &1.actions.main_x) |> Nx.concatenate() |> Nx.to_flat_list()
      assert actual == Enum.map(expected, &ExPhil.Training.Data.frame_action(&1).main_x)
      assert pipeline.resolved_opts[:label_delay] == 2

      {:ok, streaming} =
        Pipeline.setup(
          Keyword.merge(Keyword.merge(Config.defaults(), opts),
            temporal: true,
            bptt: true,
            unroll: 16,
            batch_size: 1,
            stream_chunk_size: 1,
            bptt_val_files: 1,
            cache_streaming: false
          )
        )

      assert streaming.streaming_chunk_opts[:label_delay] == 2
      {stream, _count} = Pipeline.batch_stream(streaming, [])
      first = Enum.at(stream, 0)

      expected_actions =
        Enum.take(expected, 16) |> Enum.map(&ExPhil.Training.Data.frame_action(&1).main_x)

      assert Nx.to_flat_list(first.actions.main_x) == expected_actions
      assert Nx.to_flat_list(hd(streaming.val_batches).actions.main_x) == expected_actions
    end
  end
end
