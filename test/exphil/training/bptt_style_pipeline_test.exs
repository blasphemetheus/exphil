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

  @tag :tmp_dir
  test "registry ids follow game count, not file order (the 111-slot cap keeps frequent players)",
       %{tmp_dir: dir} do
    fixture = "test/fixtures/replays/fox_multishine.slp"

    for name <- [
          "00 [ZZZ] Fox + Marth (FD).slp",
          "01 [MID] Fox + Marth (FD).slp",
          "02 [TOP] Fox + Marth (FD).slp",
          "03 [TOP] Fox + Marth (FD).slp",
          "04 [TOP] Fox + Marth (FD).slp",
          "05 [MID] Fox + Marth (FD).slp",
          "99 [VAL] Fox + Marth (FD).slp"
        ],
        do: File.cp!(fixture, Path.join(dir, name))

    opts =
      Keyword.merge(Config.defaults(),
        replays: dir,
        temporal: true,
        backbone: :gru,
        bptt: true,
        unroll: 16,
        batch_size: 2,
        stream_chunk_size: 2,
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
    assert Enum.map(1..3, &PlayerRegistry.get_tag(registry, &1)) == ["TOP", "MID", "ZZZ"]
    assert PlayerRegistry.get_id(registry, "VAL") == nil
  end
end

defmodule ExPhil.Training.PlayerTagMapPipelineTest do
  use ExUnit.Case, async: false
  @moduletag :nif
  alias ExPhil.Training.{Config, Pipeline, PlayerRegistry, PlayerTagMap, Streaming}

  @tag :tmp_dir
  test "--player-tag-map overrides filename tags for frames and registry; wrong port is ignored", %{tmp_dir: dir} do
    fixture = "test/fixtures/replays/fox_multishine.slp"
    a = Path.join(dir, "00 [FILE] Fox + Marth (FD).slp")
    b = Path.join(dir, "01 [KEEP] Fox + Marth (FD).slp")
    c = Path.join(dir, "02 Fox + Marth (FD).slp")
    v = Path.join(dir, "99 [VAL] Fox + Marth (FD).slp")
    for p <- [a, b, c, v], do: File.cp!(fixture, p)

    {:ok, meta} = ExPhil.Data.Peppi.metadata(a)
    fox_port = Enum.find(meta.players, &(&1.character_name == "Fox")).port
    other_port = 5 - fox_port

    map_path = Path.join(dir, "player_tag_map.json")

    File.write!(map_path, Jason.encode!(%{
      "protocol" => "style_identify_v1",
      "entries" => %{
        a => %{"tag" => "MATCHED", "port" => fox_port, "p" => 0.9},
        # entry for the wrong port must not apply
        b => %{"tag" => "WRONG", "port" => other_port, "p" => 0.9},
        c => %{"tag" => "~c07", "port" => fox_port, "p" => 1.2}
      }
    }))

    map = PlayerTagMap.load!(map_path)
    assert Streaming.subject_tag(a, fox_port, "Fox", tag_map: map) == "MATCHED"
    assert Streaming.subject_tag(b, fox_port, "Fox", tag_map: map) == "KEEP"
    assert Streaming.subject_tag(c, fox_port, "Fox", tag_map: map) == "~c07"
    assert PlayerTagMap.pseudo?("~c07") and not PlayerTagMap.pseudo?("MATCHED")

    opts =
      Keyword.merge(Config.defaults(),
        replays: dir, temporal: true, backbone: :gru, bptt: true, unroll: 16, batch_size: 2,
        stream_chunk_size: 4, bptt_val_files: 1, learn_player_styles: true, train_character: :fox,
        select_character_port: true, label_delay: 0, cache_streaming: false,
        player_tag_map: map_path, checkpoint: Path.join(dir, "model.axon")
      )

    {:ok, pipeline} = Pipeline.setup(opts)
    registry = pipeline.streaming_dataset_opts[:player_registry]
    assert Enum.sort(PlayerRegistry.list_tags(registry)) == ["KEEP", "MATCHED", "~c07"]

    # frames of the overridden file carry the map's tag
    {:ok, frames, []} = Streaming.parse_chunk([{a, fox_port}], subject_character: "Fox", label_delay: 0, show_progress: false, tag_map: map)
    assert hd(frames)[:player_tag] == "MATCHED"
    {:ok, frames_c, []} = Streaming.parse_chunk([{c, fox_port}], subject_character: "Fox", label_delay: 0, show_progress: false, tag_map: map)
    assert hd(frames_c)[:player_tag] == "~c07"
  end
end
