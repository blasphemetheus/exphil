defmodule ExPhil.Training.StreamingSeedTest do
  use ExUnit.Case, async: false
  @moduletag :gpu
  alias ExPhil.Training.{Data, Pipeline}

  test "prepared streaming batches honor the requested shuffle seed" do
    frames =
      for i <- 0..63 do
        %{
          game_state: %{frame: i},
          controller: %{
            button_a: false,
            button_b: false,
            button_x: false,
            button_y: false,
            button_z: false,
            button_l: false,
            button_r: false,
            button_d_up: false,
            main_stick: %{x: 0.5, y: 0.5},
            c_stick: %{x: 0.5, y: 0.5},
            l_shoulder: 0.0,
            r_shoulder: 0.0
          }
        }
      end

    dataset = Data.from_frame_lists([frames])
    dataset = %{dataset | embedded_frames: Nx.iota({64, 1}, type: :f32)}

    pipeline = %Pipeline{
      streaming: true,
      file_chunks: [[]],
      prepared_chunks: [{dataset, 1, []}],
      streaming_chunk_opts: [],
      streaming_dataset_opts: [],
      estimated_batches: 16,
      resolved_opts: [
        awbc: false,
        pipeline_chunks: true,
        temporal: true,
        batch_size: 4,
        window_size: 4,
        stride: 1,
        seed: 905,
        neutral_weight: 1.0
      ]
    }

    batches = fn p ->
      {stream, _} = Pipeline.batch_stream(p, [])
      Enum.map(stream, &Nx.to_binary(&1.states))
    end

    first = batches.(pipeline)
    assert length(first) == 16
    assert first == batches.(pipeline)
    other = %{pipeline | resolved_opts: Keyword.put(pipeline.resolved_opts, :seed, 906)}
    refute first == batches.(other)
  end
end
