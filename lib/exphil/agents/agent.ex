defmodule ExPhil.Agents.Agent do
  @moduledoc """
  GenServer that holds a trained policy and performs inference.

  An Agent encapsulates a neural network policy and provides a simple
  interface for getting actions from game states. Agents can be hot-swapped
  with new policies during runtime.

  ## Usage

      # Start an agent with a trained policy
      {:ok, agent} = ExPhil.Agents.Agent.start_link(
        name: :mewtwo_agent,
        policy_path: "checkpoints/mewtwo_policy.axon"
      )

      # Get action for a game state
      {:ok, action} = ExPhil.Agents.Agent.get_action(agent, game_state)

      # Get controller state directly
      {:ok, controller} = ExPhil.Agents.Agent.get_controller(agent, game_state)

      # Hot-swap policy
      :ok = ExPhil.Agents.Agent.load_policy(agent, "checkpoints/new_policy.axon")

  ## Frame Delay Handling

  For online play with frame delay, the agent can maintain a frame buffer:

      {:ok, agent} = ExPhil.Agents.Agent.start_link(
        name: :online_agent,
        policy_path: "checkpoints/policy.axon",
        frame_delay: 4
      )

  """

  use GenServer

  alias ExPhil.{Embeddings, Networks, Training}
  alias ExPhil.Bridge.{GameState, ControllerState}
  alias ExPhil.Training.Utils

  require Logger

  @registry ExPhil.Registry

  defstruct [
    :name,
    :policy_params,
    :predict_fn,
    :embed_config,
    :frame_delay,
    :frame_buffer,
    :deterministic,
    :temperature,
    # Temporal inference config
    :temporal,
    :backbone,
    :window_size,
    # JIT warmup tracking
    :warmed_up,
    # Action repeat (skip inference N-1 frames)
    :action_repeat,
    :frames_since_inference,
    :last_action,
    # Mamba incremental inference cache
    :mamba_cache,
    :use_incremental
  ]

  @type t :: %__MODULE__{}

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start an Agent GenServer.

  ## Options
    - `:name` - Atom name for registry lookup
    - `:policy_path` - Path to exported policy file
    - `:policy` - Pre-loaded policy map (alternative to path)
    - `:frame_delay` - Frame delay for online play (default: 0)
    - `:deterministic` - Use deterministic action selection (default: false)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:embed_config` - Embedding configuration (auto-detected from policy)
  """
  def start_link(opts) do
    name = Keyword.get(opts, :name)

    gen_opts = if name do
      [name: {:via, Registry, {@registry, {:agent, name}}}]
    else
      []
    end

    GenServer.start_link(__MODULE__, opts, gen_opts)
  end

  @doc """
  Get the action for a game state.

  Returns a map with button presses and stick positions.
  """
  @spec get_action(GenServer.server(), GameState.t(), keyword()) ::
    {:ok, map()} | {:error, term()}
  def get_action(agent, game_state, opts \\ []) do
    GenServer.call(agent, {:get_action, game_state, opts})
  end

  @doc """
  Get action with confidence scores.

  Returns `{:ok, action, confidence}` where confidence is a map with:
  - `:overall` - Overall confidence (0-1)
  - `:buttons` - Button prediction confidence
  - `:main` - Main stick confidence
  - `:c` - C-stick confidence
  - `:shoulder` - Shoulder confidence
  """
  @spec get_action_with_confidence(GenServer.server(), GameState.t(), keyword()) ::
    {:ok, map(), map()} | {:error, term()}
  def get_action_with_confidence(agent, game_state, opts \\ []) do
    GenServer.call(agent, {:get_action_with_confidence, game_state, opts})
  end

  @doc """
  Get controller state for a game state.

  Returns a ControllerState struct ready to send to the bridge.
  """
  @spec get_controller(GenServer.server(), GameState.t(), keyword()) ::
    {:ok, ControllerState.t()} | {:error, term()}
  def get_controller(agent, game_state, opts \\ []) do
    GenServer.call(agent, {:get_controller, game_state, opts})
  end

  @doc """
  Load a new policy, hot-swapping the current one.
  """
  @spec load_policy(GenServer.server(), Path.t() | map()) :: :ok | {:error, term()}
  def load_policy(agent, policy_or_path) do
    GenServer.call(agent, {:load_policy, policy_or_path})
  end

  @doc """
  Get current agent configuration.
  """
  @spec get_config(GenServer.server()) :: map()
  def get_config(agent) do
    GenServer.call(agent, :get_config)
  end

  @doc """
  Update agent settings.
  """
  @spec configure(GenServer.server(), keyword()) :: :ok
  def configure(agent, opts) do
    GenServer.call(agent, {:configure, opts})
  end

  @doc """
  Reset the frame buffer. Call this when starting a new game.
  """
  @spec reset_buffer(GenServer.server()) :: :ok
  def reset_buffer(agent) do
    GenServer.call(agent, :reset_buffer)
  end

  @doc """
  Warmup JIT compilation by running a dummy inference.

  Call this during menu navigation so the first real game frame
  gets fast inference. Returns the warmup time in milliseconds.
  """
  @spec warmup(GenServer.server()) :: {:ok, non_neg_integer()} | {:error, term()}
  def warmup(agent) do
    GenServer.call(agent, :warmup, 120_000)  # 2 minute timeout for JIT
  end

  @doc """
  Check if the agent has been warmed up (JIT compiled).
  """
  @spec warmed_up?(GenServer.server()) :: boolean()
  def warmed_up?(agent) do
    GenServer.call(agent, :warmed_up?)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    frame_delay = Keyword.get(opts, :frame_delay, 0)
    deterministic = Keyword.get(opts, :deterministic, false)
    temperature = Keyword.get(opts, :temperature, 1.0)
    action_repeat = Keyword.get(opts, :action_repeat, 1)

    # Allow disabling incremental inference for testing/comparison
    use_incremental = Keyword.get(opts, :use_incremental, true)

    state = %__MODULE__{
      name: Keyword.get(opts, :name),
      frame_delay: frame_delay,
      frame_buffer: :queue.new(),
      deterministic: deterministic,
      temperature: temperature,
      # Temporal config - will be set when policy is loaded
      temporal: false,
      backbone: :mlp,
      window_size: 60,
      # JIT warmup
      warmed_up: false,
      # Action repeat
      action_repeat: action_repeat,
      frames_since_inference: 0,
      last_action: nil,
      # Mamba incremental inference (will be initialized when Mamba policy loads)
      mamba_cache: nil,
      use_incremental: use_incremental
    }

    # Load policy if provided
    state = cond do
      Keyword.has_key?(opts, :policy_path) ->
        case load_policy_internal(state, Keyword.fetch!(opts, :policy_path)) do
          {:ok, new_state} -> new_state
          {:error, reason} ->
            Logger.warning("[Agent] Failed to load policy: #{inspect(reason)}")
            state
        end

      Keyword.has_key?(opts, :policy) ->
        case load_policy_internal(state, Keyword.fetch!(opts, :policy)) do
          {:ok, new_state} -> new_state
          {:error, reason} ->
            Logger.warning("[Agent] Failed to load policy: #{inspect(reason)}")
            state
        end

      true ->
        state
    end

    {:ok, state}
  end

  @impl true
  def handle_call({:get_action, game_state, opts}, _from, state) do
    case compute_action(state, game_state, opts) do
      {:ok, action, _confidence, new_state} ->
        {:reply, {:ok, action}, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:get_action_with_confidence, game_state, opts}, _from, state) do
    case compute_action(state, game_state, opts) do
      {:ok, action, confidence, new_state} ->
        {:reply, {:ok, action, confidence}, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:get_controller, game_state, opts}, _from, state) do
    case compute_action(state, game_state, opts) do
      {:ok, action, _confidence, new_state} ->
        controller = action_to_controller(action, state)
        {:reply, {:ok, controller}, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:load_policy, policy_or_path}, _from, state) do
    case load_policy_internal(state, policy_or_path) do
      {:ok, new_state} ->
        Logger.info("[Agent] Policy loaded successfully")
        {:reply, :ok, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:get_config, _from, state) do
    config = %{
      name: state.name,
      frame_delay: state.frame_delay,
      deterministic: state.deterministic,
      temperature: state.temperature,
      has_policy: state.policy_params != nil,
      # Temporal config
      temporal: state.temporal,
      backbone: state.backbone,
      window_size: state.window_size,
      buffer_size: :queue.len(state.frame_buffer),
      # Incremental inference status
      use_incremental: state.use_incremental,
      incremental_active: state.mamba_cache != nil,
      incremental_step: if(state.mamba_cache, do: state.mamba_cache.step, else: 0)
    }
    {:reply, config, state}
  end

  @impl true
  def handle_call(:reset_buffer, _from, state) do
    # Reset mamba cache if using incremental inference
    new_mamba_cache = if state.backbone == :mamba and state.use_incremental do
      init_mamba_cache(state)
    else
      state.mamba_cache
    end

    new_state = %{state |
      frame_buffer: :queue.new(),
      last_action: nil,
      frames_since_inference: 0,
      mamba_cache: new_mamba_cache
    }
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:warmup, _from, state) do
    if state.policy_params == nil do
      {:reply, {:error, :no_policy_loaded}, state}
    else
      Logger.info("[Agent] Starting JIT warmup...")
      start_time = System.monotonic_time(:millisecond)

      # Create a dummy game state for warmup
      dummy_state = create_dummy_game_state()
      dummy_embedded = embed_game_state(dummy_state, 1, state.embed_config)

      # Run inference to trigger JIT compilation
      if state.temporal do
        # For temporal models, we need a full window of dummy embeddings
        dummy_sequence = List.duplicate(dummy_embedded, state.window_size)
        |> Nx.stack()
        |> Nx.new_axis(0)  # Add batch dimension

        _output = state.predict_fn.(Utils.ensure_model_state(state.policy_params), dummy_sequence)
      else
        # For MLP, just single frame
        input = Nx.reshape(dummy_embedded, {1, :auto})
        _output = state.predict_fn.(Utils.ensure_model_state(state.policy_params), input)
      end

      elapsed = System.monotonic_time(:millisecond) - start_time
      Logger.info("[Agent] JIT warmup complete (#{elapsed}ms)")

      {:reply, {:ok, elapsed}, %{state | warmed_up: true}}
    end
  end

  @impl true
  def handle_call(:warmed_up?, _from, state) do
    {:reply, state.warmed_up, state}
  end

  @impl true
  def handle_call({:configure, opts}, _from, state) do
    state = state
    |> maybe_update(:deterministic, opts)
    |> maybe_update(:temperature, opts)
    |> maybe_update(:frame_delay, opts)

    {:reply, :ok, state}
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp compute_action(state, _game_state, _opts) when state.policy_params == nil do
    {:error, :no_policy_loaded}
  end

  defp compute_action(state, game_state, opts) do
    # Action repeat: return cached action if we haven't hit the repeat interval
    if state.action_repeat > 1 and state.last_action != nil and
       state.frames_since_inference < state.action_repeat do
      # Return cached action with default confidence, increment counter
      new_state = %{state | frames_since_inference: state.frames_since_inference + 1}
      default_conf = %{overall: 0.0, buttons: 0.0, main: 0.0, c: 0.0, shoulder: 0.0}
      {:ok, state.last_action, default_conf, new_state}
    else
      # Time to run actual inference
      do_compute_action(state, game_state, opts)
    end
  end

  defp do_compute_action(state, game_state, opts) do
    try do
      # Embed game state
      player_port = Keyword.get(opts, :player_port, 1)
      embedded = embed_game_state(game_state, player_port, state.embed_config)

      # Route to appropriate inference mode
      {action, confidence, new_state} = cond do
        # Mamba with incremental inference (O(1) per frame)
        state.temporal and state.backbone == :mamba and state.use_incremental and state.mamba_cache != nil ->
          compute_incremental_mamba_action(state, embedded, opts)

        # Standard temporal inference (O(window_size) per frame)
        state.temporal ->
          compute_temporal_action(state, embedded, opts)

        # Single-frame MLP inference
        true ->
          compute_single_frame_action(state, embedded, opts)
      end

      # Cache action for action repeat
      new_state = %{new_state |
        last_action: action,
        frames_since_inference: 1,
        warmed_up: true  # Mark as warmed up after first successful inference
      }

      {:ok, action, confidence, new_state}
    rescue
      e ->
        Logger.error("[Agent] Error computing action: #{inspect(e)}")
        {:error, e}
    end
  end

  # Single-frame inference (original behavior)
  defp compute_single_frame_action(state, embedded, opts) do
    # Add batch dimension [1, embed_size]
    embedded_batch = Nx.reshape(embedded, {1, Nx.size(embedded)})

    deterministic = Keyword.get(opts, :deterministic, state.deterministic)
    temperature = Keyword.get(opts, :temperature, state.temperature)

    action = Networks.Policy.sample(
      state.policy_params,
      state.predict_fn,
      embedded_batch,
      deterministic: deterministic,
      temperature: temperature
    )

    # Compute confidence from logits
    confidence = Networks.Policy.compute_confidence(action)

    {action, confidence, state}
  end

  # Temporal inference with frame buffering
  defp compute_temporal_action(state, embedded, opts) do
    # Add new frame to buffer
    buffer = :queue.in(embedded, state.frame_buffer)

    # Trim buffer if it exceeds window size
    {buffer, _dropped} = trim_buffer(buffer, state.window_size)

    # Get number of frames we have
    buffer_len = :queue.len(buffer)

    # Build sequence tensor
    sequence = if buffer_len < state.window_size do
      # Pad with first frame until we have enough
      # This allows inference to start immediately (warmup period)
      pad_sequence(buffer, state.window_size)
    else
      # Full window available
      buffer_to_tensor(buffer)
    end

    # Reshape to [1, window_size, embed_size]
    embed_size = Nx.size(embedded)
    sequence_batch = Nx.reshape(sequence, {1, state.window_size, embed_size})

    deterministic = Keyword.get(opts, :deterministic, state.deterministic)
    temperature = Keyword.get(opts, :temperature, state.temperature)

    action = Networks.Policy.sample(
      state.policy_params,
      state.predict_fn,
      sequence_batch,
      deterministic: deterministic,
      temperature: temperature
    )

    # Compute confidence from logits
    confidence = Networks.Policy.compute_confidence(action)

    # Update state with new buffer
    new_state = %{state | frame_buffer: buffer}

    {action, confidence, new_state}
  end

  # Incremental Mamba inference with state caching (O(1) per frame)
  # This is 60x faster than compute_temporal_action for window_size=60
  defp compute_incremental_mamba_action(state, embedded, opts) do
    alias ExPhil.Networks.Mamba

    # Project embedding to hidden size (matching what the model does)
    # embedded: [embed_size] -> need [1, embed_size] for batch
    hidden_size = state.embed_config[:hidden_size] || 256
    embed_size = Nx.size(embedded)

    # Reshape to batch: [1, embed_size]
    x = Nx.reshape(embedded, {1, embed_size})

    # Project to hidden_size if needed (input projection layer)
    x = if embed_size != hidden_size do
      input_proj = state.policy_params["input_projection"] ||
                   state.policy_params[:input_projection]

      if input_proj do
        kernel = input_proj["kernel"] || input_proj[:kernel]
        bias = input_proj["bias"] || input_proj[:bias]
        out = Nx.dot(x, kernel)
        if bias, do: Nx.add(out, bias), else: out
      else
        # No projection params found, pad or truncate
        if embed_size < hidden_size do
          Nx.pad(x, 0.0, [{0, 0, 0}, {0, hidden_size - embed_size, 0}])
        else
          Nx.slice_along_axis(x, 0, hidden_size, axis: 1)
        end
      end
    else
      x
    end

    # Single step through Mamba with cached state
    {backbone_output, new_cache} = Mamba.step(
      x,
      state.policy_params,
      state.mamba_cache
    )

    # backbone_output: [1, hidden_size]
    # Now run through policy heads (dense layers after backbone)
    deterministic = Keyword.get(opts, :deterministic, state.deterministic)
    temperature = Keyword.get(opts, :temperature, state.temperature)

    action = sample_from_backbone_output(
      backbone_output,
      state.policy_params,
      deterministic: deterministic,
      temperature: temperature,
      axis_buckets: state.embed_config[:axis_buckets] || 16,
      shoulder_buckets: state.embed_config[:shoulder_buckets] || 4
    )

    confidence = Networks.Policy.compute_confidence(action)

    new_state = %{state | mamba_cache: new_cache}

    {action, confidence, new_state}
  end

  # Sample action from backbone output (run through policy heads)
  defp sample_from_backbone_output(backbone_output, params, opts) do
    deterministic = Keyword.get(opts, :deterministic, false)
    temperature = Keyword.get(opts, :temperature, 1.0)
    axis_buckets = Keyword.get(opts, :axis_buckets, 16)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, 4)

    # Get policy head parameters
    # These are the final dense layers that produce logits
    buttons_logits = dense_head(backbone_output, params, "buttons_head")
    main_x_logits = dense_head(backbone_output, params, "main_x_head")
    main_y_logits = dense_head(backbone_output, params, "main_y_head")
    c_x_logits = dense_head(backbone_output, params, "c_x_head")
    c_y_logits = dense_head(backbone_output, params, "c_y_head")
    shoulder_logits = dense_head(backbone_output, params, "shoulder_head")

    # Sample from each head
    buttons = sample_buttons(buttons_logits, deterministic, temperature)
    main_x = sample_categorical(main_x_logits, axis_buckets, deterministic, temperature)
    main_y = sample_categorical(main_y_logits, axis_buckets, deterministic, temperature)
    c_x = sample_categorical(c_x_logits, axis_buckets, deterministic, temperature)
    c_y = sample_categorical(c_y_logits, axis_buckets, deterministic, temperature)
    shoulder = sample_categorical(shoulder_logits, shoulder_buckets, deterministic, temperature)

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder,
      # Include logits for confidence computation
      buttons_logits: buttons_logits,
      main_x_logits: main_x_logits,
      main_y_logits: main_y_logits,
      c_x_logits: c_x_logits,
      c_y_logits: c_y_logits,
      shoulder_logits: shoulder_logits
    }
  end

  defp dense_head(x, params, head_name) do
    head_params = params[head_name] || params[String.to_atom(head_name)] || %{}
    kernel = head_params["kernel"] || head_params[:kernel]

    if kernel do
      bias = head_params["bias"] || head_params[:bias]
      out = Nx.dot(x, kernel)
      if bias, do: Nx.add(out, bias), else: out
    else
      # Fallback: head not found, return zeros (will produce uniform sampling)
      Logger.warning("[Agent] Policy head #{head_name} not found in params")
      Nx.broadcast(0.0, {1, 8})  # Default 8 outputs
    end
  end

  defp sample_buttons(logits, deterministic, temperature) do
    # Buttons are independent Bernoulli (8 buttons)
    probs = Nx.sigmoid(Nx.divide(logits, temperature))

    if deterministic do
      Nx.greater(probs, 0.5) |> Nx.squeeze()
    else
      key = Nx.Random.key(System.system_time(:nanosecond))
      {rand, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      Nx.greater(probs, rand) |> Nx.squeeze()
    end
  end

  defp sample_categorical(logits, _num_categories, deterministic, temperature) do
    # Apply temperature and sample
    scaled_logits = Nx.divide(logits, temperature)

    if deterministic do
      Nx.argmax(scaled_logits, axis: -1) |> Nx.squeeze()
    else
      # Gumbel-max trick for sampling
      key = Nx.Random.key(System.system_time(:nanosecond))
      {rand, _} = Nx.Random.uniform(key, shape: Nx.shape(scaled_logits))
      gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(rand, 1.0e-10)))))
      Nx.argmax(Nx.add(scaled_logits, gumbel), axis: -1) |> Nx.squeeze()
    end
  end

  # Initialize Mamba cache from state config
  defp init_mamba_cache(state) do
    alias ExPhil.Networks.Mamba

    hidden_size = state.embed_config[:hidden_size] || 256
    num_layers = state.embed_config[:num_layers] || 2

    Mamba.init_cache(
      batch_size: 1,
      hidden_size: hidden_size,
      num_layers: num_layers
    )
  end

  # Trim buffer to max size, dropping oldest frames
  defp trim_buffer(buffer, max_size) do
    len = :queue.len(buffer)
    if len > max_size do
      # Drop oldest frames
      to_drop = len - max_size
      {_dropped, trimmed} = Enum.reduce(1..to_drop, {[], buffer}, fn _, {dropped, buf} ->
        {{:value, frame}, new_buf} = :queue.out(buf)
        {[frame | dropped], new_buf}
      end)
      {trimmed, to_drop}
    else
      {buffer, 0}
    end
  end

  # Convert buffer queue to stacked tensor [seq_len, embed_size]
  defp buffer_to_tensor(buffer) do
    buffer
    |> :queue.to_list()
    |> Nx.stack()
  end

  # Pad sequence by repeating first frame
  defp pad_sequence(buffer, target_size) do
    frames = :queue.to_list(buffer)
    current_len = length(frames)

    if current_len == 0 do
      # No frames yet - this shouldn't happen but handle gracefully
      raise "Cannot pad empty buffer"
    end

    # Repeat first frame to pad the beginning
    first_frame = hd(frames)
    padding_count = target_size - current_len
    padded_frames = List.duplicate(first_frame, padding_count) ++ frames

    Nx.stack(padded_frames)
  end

  defp embed_game_state(game_state, player_port, _embed_config) do
    # Use default Game config for embedding
    # The embed_config we store is just for axis_buckets/shoulder_buckets, not the full Game struct
    Embeddings.Game.embed(game_state, nil, player_port)
  end

  defp action_to_controller(action, state) do
    Networks.to_controller_state(action, axis_buckets: state.embed_config[:axis_buckets] || 16)
  end

  defp load_policy_internal(state, path) when is_binary(path) do
    case Training.load_policy(path) do
      {:ok, policy} -> load_policy_internal(state, policy)
      error -> error
    end
  end

  defp load_policy_internal(state, %{params: params, config: config} = _policy) do
    # Extract config
    embed_config = Map.get(config, :embed_config, %{})
    embed_size = Map.get(config, :embed_size) || Map.get(embed_config, :embed_size, 1991)
    axis_buckets = Map.get(config, :axis_buckets, 16)
    shoulder_buckets = Map.get(config, :shoulder_buckets, 4)

    # Temporal config
    temporal = Map.get(config, :temporal, false)
    backbone = Map.get(config, :backbone, :mlp)
    window_size = Map.get(config, :window_size, 60)

    # MLP backbone config
    hidden_sizes = Map.get(config, :hidden_sizes, [512, 512])
    dropout = Map.get(config, :dropout, 0.1)

    # Build appropriate model based on temporal flag
    # Note: dropout is included to match training architecture, but Axon
    # disables dropout during inference mode by default
    model = if temporal do
      Logger.info("[Agent] Loading temporal policy (backbone: #{backbone}, window: #{window_size})")
      Networks.Policy.build_temporal(
        embed_size: embed_size,
        backbone: backbone,
        window_size: window_size,
        num_heads: Map.get(config, :num_heads, 4),
        head_dim: Map.get(config, :head_dim, 64),
        hidden_size: Map.get(config, :hidden_size, 256),
        num_layers: Map.get(config, :num_layers, 2),
        dropout: dropout,
        axis_buckets: axis_buckets,
        shoulder_buckets: shoulder_buckets
      )
    else
      Networks.Policy.build(
        embed_size: embed_size,
        hidden_sizes: hidden_sizes,
        dropout: dropout,
        axis_buckets: axis_buckets,
        shoulder_buckets: shoulder_buckets
      )
    end

    {_init_fn, predict_fn} = Axon.build(model)

    # Build embed_config with all needed fields
    full_embed_config = Map.merge(%{
      embed_size: embed_size,
      axis_buckets: axis_buckets,
      shoulder_buckets: shoulder_buckets,
      hidden_size: Map.get(config, :hidden_size, 256),
      num_layers: Map.get(config, :num_layers, 2)
    }, embed_config)

    # Initialize Mamba cache if using Mamba backbone with incremental inference
    mamba_cache = if temporal and backbone == :mamba and state.use_incremental do
      Logger.info("[Agent] Initializing Mamba incremental inference cache")
      alias ExPhil.Networks.Mamba
      Mamba.init_cache(
        batch_size: 1,
        hidden_size: full_embed_config.hidden_size,
        num_layers: full_embed_config.num_layers
      )
    else
      nil
    end

    new_state = %{state |
      policy_params: params,
      predict_fn: predict_fn,
      embed_config: full_embed_config,
      # Set temporal config
      temporal: temporal,
      backbone: backbone,
      window_size: window_size,
      # Reset frame buffer when loading new policy
      frame_buffer: :queue.new(),
      # Mamba cache for incremental inference
      mamba_cache: mamba_cache
    }

    {:ok, new_state}
  end

  defp load_policy_internal(_state, invalid) do
    {:error, {:invalid_policy, invalid}}
  end

  defp maybe_update(state, key, opts) do
    if Keyword.has_key?(opts, key) do
      Map.put(state, key, Keyword.fetch!(opts, key))
    else
      state
    end
  end

  # Create a dummy game state for JIT warmup
  defp create_dummy_game_state do
    alias ExPhil.Bridge.{GameState, Player}

    dummy_player = %Player{
      character: 25,  # Mewtwo
      x: 0.0,
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: 1,
      action: 0,
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }

    %GameState{
      frame: 0,
      stage: 32,  # Final Destination
      menu_state: 2,  # In-game
      players: %{1 => dummy_player, 2 => dummy_player},
      projectiles: [],
      distance: 50.0
    }
  end
end
