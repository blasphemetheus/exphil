defmodule ExPhil.Training.BpttStylePipelineTest do
  use ExUnit.Case, async: false
  @moduletag :nif
  alias ExPhil.Training.{Config, Pipeline, PlayerRegistry}

  @tag :tmp_dir
  test "held-out filename tags cannot enter the training registry", %{tmp_dir: dir} do
    fixture = "test/fixtures/replays/fox_multishine.slp"
    train = Path.join(dir, "00 [TRAIN] Fox + Game & Watch (FD).slp")
    val = Path.join(dir, "99 [VALIDATION] Fox + Game & Watch (FD).slp")
    File.cp!(fixture, train)
    File.cp!(fixture, val)

    opts =
      Keyword.merge(Config.defaults(),
        replays: dir,
        temporal: true,
        backbone: :gru,
        bptt: true,
        unroll: 16,
        batch_size: 2,
        stream_chunk_size: 1,
        bptt_val_files: 1,
        learn_player_styles: true,
        train_character: :fox,
        select_character_port: true,
        label_delay: 0,
        cache_streaming: false,
        checkpoint: Path.join(dir, "model.axon")
      )

    {:ok, pipeline} = Pipeline.setup(opts)
    registry = pipeline.streaming_dataset_opts[:player_registry]
    assert PlayerRegistry.list_tags(registry) == ["TRAIN"]
    assert PlayerRegistry.get_id(registry, "TRAIN") == 1
    assert PlayerRegistry.get_id(registry, "VALIDATION") == nil
    assert PlayerRegistry.get_tag(registry, 0) == nil
    assert {:ok, ^registry} = PlayerRegistry.from_json(Path.join(dir, "model_players.json"))
    assert length(pipeline.replay_files) == 1
    assert pipeline.val_batches != []
  end
end
