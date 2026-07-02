defmodule ExPhil.Inference.PolicyServing do
  @moduledoc """
  Nx.Serving-based policy inference server for real-time Dolphin gameplay.

  Wraps a trained policy network in an Nx.Serving process that auto-batches
  requests from multiple game instances, enabling efficient GPU utilization
  for parallel self-play.

  ## Usage

  ### Inline (no server process)

      serving = PolicyServing.create("checkpoints/model_best_policy.bin")
      action = Nx.Serving.run(serving, game_state)

  ### As a supervised process (auto-batching)

      # In your supervision tree
      children = [
        PolicyServing.child_spec(
          checkpoint: "checkpoints/model_best_policy.bin",
          batch_size: 8,
          batch_timeout: 10
        )
      ]

      # From any game instance process
      action = PolicyServing.predict(game_state)

  ## Auto-batching

  When running as a process, multiple game instances can call `predict/1`
  concurrently. The serving collects requests and batches them together
  (up to `batch_size` or `batch_timeout` ms), runs a single GPU inference,
  then distributes results back to each caller.

  This is critical for self-play: 8 games × batch 1 = 8 GPU calls at low
  utilization. With auto-batching: 1 GPU call at batch 8 = same latency,
  8x throughput.

  ## Multi-GPU

  Pass `partitions: true` to distribute across available GPUs:

      PolicyServing.child_spec(
        checkpoint: path,
        partitions: true  # auto-detects GPUs
      )

  ## Temporal Models (Mamba, LSTM)

  For sequence models, pass a frame buffer as part of the input:

      action = PolicyServing.predict({game_state, frame_buffer})

  The client_preprocessing builds the temporal embedding from the buffer.
  """

  require Logger

  alias ExPhil.Embeddings
  alias ExPhil.Networks.Policy.Sampling

  @default_batch_size 8
  @default_batch_timeout 10  # ms — must be <16.67ms for 60fps

  @doc """
  Quantize a trained model to INT8 for faster inference.

  Uses Axon.Quantization for weight-only 8-bit integer quantization.
  Reduces memory footprint and may improve inference speed.

      {q_model, q_state} = PolicyServing.quantize(model, model_state)
  """
  def quantize(model, model_state) do
    Axon.Quantization.quantize(model, model_state)
  end

  @doc """
  Create an Nx.Serving for policy inference.

  ## Options
  - `:embed_config` — Embedding config (default: ExPhil.Embeddings.config())
  - `:temperature` — Sampling temperature (default: 1.0)
  - `:deterministic` — Use argmax instead of sampling (default: false)
  - `:temporal` — Enable temporal mode with frame buffer (default: false)
  - `:compiler` — Nx compiler (default: EXLA)
  - `:quantize` — Apply INT8 quantization (default: false)
  """
  @spec create(Path.t(), keyword()) :: Nx.Serving.t()
  def create(checkpoint_path, opts \\ []) do
    embed_config = Keyword.get(opts, :embed_config, Embeddings.config())
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)
    temporal = Keyword.get(opts, :temporal, false)
    compiler = Keyword.get(opts, :compiler, EXLA)
    quantize_model = Keyword.get(opts, :quantize, false)

    # Load policy
    {:ok, policy} = ExPhil.Training.load_policy(checkpoint_path)
    params = policy.params || policy[:params]
    config = policy.config || policy[:config] || %{}

    # Build model and optionally quantize
    model = policy.model || build_model_from_config(config)

    {model, params} =
      if quantize_model do
        Logger.info("Quantizing model to INT8...")
        {q_model, q_state} = Axon.Quantization.quantize(model, params)
        {q_model, q_state}
      else
        {model, params}
      end

    # Build predict function
    {_init_fn, predict_fn} = ExPhil.Training.Utils.build_compiled(model, mode: :inference)

    Nx.Serving.new(fn serving_opts ->
      Nx.Defn.jit(
        fn input -> predict_fn.(params, input) end,
        Keyword.merge(serving_opts, compiler: compiler, on_conflict: :reuse)
      )
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      embedding = preprocess_input(input, embed_config, temporal)
      {Nx.Batch.stack([embedding]), %{temperature: temperature, deterministic: deterministic}}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _server_info}, client_info ->
      postprocess_output(result, client_info)
    end)
  end

  @doc """
  Create a child spec for use in a supervision tree.

  ## Options
  - `:checkpoint` — Path to policy checkpoint (required)
  - `:name` — Process name (default: ExPhil.Inference.PolicyServing)
  - `:batch_size` — Max batch size (default: 8)
  - `:batch_timeout` — Max wait time in ms (default: 10)
  - `:partitions` — Enable multi-GPU partitioning (default: false)
  - Plus all options from `create/2`
  """
  def child_spec(opts) do
    checkpoint = Keyword.fetch!(opts, :checkpoint)
    name = Keyword.get(opts, :name, __MODULE__)
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    batch_timeout = Keyword.get(opts, :batch_timeout, @default_batch_timeout)
    partitions = Keyword.get(opts, :partitions, false)

    serving = create(checkpoint, opts)

    %{
      id: name,
      start: {Nx.Serving, :start_link, [[
        serving: serving,
        name: name,
        batch_size: batch_size,
        batch_timeout: batch_timeout,
        partitions: partitions
      ]]}
    }
  end

  @doc """
  Run inference on a game state using the named serving process.

  Returns a sampled action map with `:buttons`, `:main_x`, `:main_y`,
  `:c_x`, `:c_y`, `:shoulder` keys.

  For temporal models, pass `{game_state, frame_buffer}` where frame_buffer
  is a list of recent game states.
  """
  @spec predict(term(), atom()) :: map()
  def predict(input, name \\ __MODULE__) do
    Nx.Serving.batched_run(name, input)
  end

  # ============================================================================
  # Private — Preprocessing
  # ============================================================================

  defp preprocess_input(game_state, embed_config, false = _temporal) do
    # Single-frame: embed one game state
    Embeddings.Game.embed_state(game_state, 1, config: embed_config)
  end

  defp preprocess_input({game_state, frame_buffer}, embed_config, true = _temporal) do
    # Temporal: embed frame buffer as a sequence
    frames = frame_buffer ++ [game_state]
    embeddings = Enum.map(frames, fn gs ->
      Embeddings.Game.embed_state(gs, 1, config: embed_config)
    end)
    Nx.stack(embeddings)  # {seq_len, embed_dim}
  end

  defp preprocess_input(game_state, embed_config, true = _temporal) do
    # Temporal without explicit buffer — single frame, model handles state internally
    Embeddings.Game.embed_state(game_state, 1, config: embed_config)
  end

  # ============================================================================
  # Private — Postprocessing
  # ============================================================================

  defp postprocess_output(result, client_info) do
    temp = Map.get(client_info, :temperature, 1.0)
    det = Map.get(client_info, :deterministic, false)

    # result is a tuple of 6 logit tensors from predict_fn
    # Squeeze batch dim (Serving adds it), sample action
    {buttons_logits, mx_logits, my_logits, cx_logits, cy_logits, sh_logits} = result

    %{
      buttons: Sampling.sample_buttons(Nx.squeeze(buttons_logits), det),
      main_x: Sampling.sample_categorical(Nx.squeeze(mx_logits), temp, det),
      main_y: Sampling.sample_categorical(Nx.squeeze(my_logits), temp, det),
      c_x: Sampling.sample_categorical(Nx.squeeze(cx_logits), temp, det),
      c_y: Sampling.sample_categorical(Nx.squeeze(cy_logits), temp, det),
      shoulder: Sampling.sample_categorical(Nx.squeeze(sh_logits), temp, det)
    }
  end

  # ============================================================================
  # Private — Model Building
  # ============================================================================

  defp build_model_from_config(config) do
    # Rebuild model architecture from saved config
    ExPhil.Networks.Policy.build(
      embed_size: config[:embed_size] || config["embed_size"] || 288,
      hidden_sizes: config[:hidden_sizes] || config["hidden_sizes"] || [512, 512, 256],
      temporal: config[:temporal] || config["temporal"] || false,
      backbone: config[:backbone] || config["backbone"] || :mlp,
      window_size: config[:window_size] || config["window_size"] || 60,
      num_layers: config[:num_layers] || config["num_layers"] || 2,
      dropout: 0.0  # No dropout during inference
    )
  end
end
