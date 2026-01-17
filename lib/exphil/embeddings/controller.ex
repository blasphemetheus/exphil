defmodule ExPhil.Embeddings.Controller do
  @moduledoc """
  Controller state embedding - converts controller inputs to Nx tensors.

  The controller has several components:
  - 8 buttons (A, B, X, Y, Z, L, R, D_UP)
  - Main stick (X, Y axes)
  - C-stick (X, Y axes)
  - Shoulder/trigger (L analog)

  ## Discretization

  For training, we discretize the continuous stick values:
  - Sticks: 16 positions per axis (native is 160, so 16 = 10x compression)
  - Shoulder: 4 positions (native is 140)

  This reduces the action space while retaining meaningful distinctions.

  ## Autoregressive Sampling

  The controller head in the neural network uses autoregressive sampling:
  buttons → main_x → main_y → c_x → c_y → shoulder

  Each component is conditioned on previous samples, enabling the model
  to learn correlations (e.g., pressing A while tilting stick up = up-tilt).
  """

  # Uses Nx with module prefix
  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Bridge.ControllerState

  # Default discretization
  @default_axis_buckets 16
  @default_shoulder_buckets 4

  # Legal buttons in Melee
  @legal_buttons [:a, :b, :x, :y, :z, :l, :r, :d_up]

  @doc """
  Configuration for controller embedding.
  """
  defstruct [
    axis_buckets: @default_axis_buckets,
    shoulder_buckets: @default_shoulder_buckets
  ]

  @type config :: %__MODULE__{
    axis_buckets: non_neg_integer(),
    shoulder_buckets: non_neg_integer()
  }

  @spec default_config() :: config()
  def default_config, do: %__MODULE__{}

  @doc """
  Calculate the embedding size for a given configuration.

  Default: 8 buttons + 2*axis_buckets (main) + 2*axis_buckets (c) + shoulder_buckets
  """
  @spec embedding_size(config()) :: non_neg_integer()
  def embedding_size(config \\ default_config()) do
    # Buttons: 8 booleans
    buttons_size = length(@legal_buttons)

    # Sticks: one-hot per axis
    main_size = 2 * (config.axis_buckets + 1)  # +1 for one-hot
    c_size = 2 * (config.axis_buckets + 1)

    # Shoulder: one-hot
    shoulder_size = config.shoulder_buckets + 1

    buttons_size + main_size + c_size + shoulder_size
  end

  @doc """
  Size of the continuous (non-discretized) controller embedding.
  Used for embedding observed controller state.
  """
  @spec continuous_embedding_size() :: non_neg_integer()
  def continuous_embedding_size do
    8 +  # buttons
    2 +  # main stick
    2 +  # c stick
    1    # shoulder
  end

  @doc """
  Embed a controller state as a continuous tensor.

  This is used for the *previous* action input to the network,
  where we want the actual values rather than discretized.
  """
  @spec embed_continuous(ControllerState.t() | nil) :: Nx.Tensor.t()
  def embed_continuous(nil) do
    Nx.broadcast(0.0, {continuous_embedding_size()})
  end

  def embed_continuous(%ControllerState{} = cs) do
    Nx.concatenate([
      embed_buttons_continuous(cs),
      embed_stick_continuous(cs.main_stick),
      embed_stick_continuous(cs.c_stick),
      embed_shoulder_continuous(cs.l_shoulder)
    ])
  end

  defp embed_buttons_continuous(%ControllerState{} = cs) do
    buttons = [
      cs.button_a,
      cs.button_b,
      cs.button_x,
      cs.button_y,
      cs.button_z,
      cs.button_l,
      cs.button_r,
      cs.button_d_up
    ]

    buttons
    |> Enum.map(fn b -> if b, do: 1.0, else: 0.0 end)
    |> Nx.tensor(type: :f32)
  end

  defp embed_stick_continuous(%{x: x, y: y}) do
    # Sticks are in [0, 1] range, center at 0.5
    # Shift to [-1, 1] range for neural network
    Nx.tensor([(x - 0.5) * 2.0, (y - 0.5) * 2.0], type: :f32)
  end

  defp embed_stick_continuous(nil) do
    Nx.tensor([0.0, 0.0], type: :f32)
  end

  defp embed_shoulder_continuous(value) when is_number(value) do
    Nx.tensor([value], type: :f32)
  end

  defp embed_shoulder_continuous(_), do: Nx.tensor([0.0], type: :f32)

  @doc """
  Embed a controller state with discretization (for action output).

  This creates a one-hot encoded representation suitable for
  computing cross-entropy loss during training.
  """
  @spec embed_discrete(ControllerState.t() | nil, config()) :: Nx.Tensor.t()
  def embed_discrete(cs, config \\ default_config())

  def embed_discrete(nil, config) do
    Nx.broadcast(0.0, {embedding_size(config)})
  end

  def embed_discrete(%ControllerState{} = cs, config) do
    Nx.concatenate([
      embed_buttons_continuous(cs),  # Buttons stay as 0/1
      embed_stick_discrete(cs.main_stick, config.axis_buckets),
      embed_stick_discrete(cs.c_stick, config.axis_buckets),
      embed_shoulder_discrete(cs.l_shoulder, config.shoulder_buckets)
    ])
  end

  defp embed_stick_discrete(%{x: x, y: y}, buckets) do
    x_bucket = discretize_axis(x, buckets)
    y_bucket = discretize_axis(y, buckets)

    Nx.concatenate([
      Primitives.one_hot(x_bucket, size: buckets + 1),
      Primitives.one_hot(y_bucket, size: buckets + 1)
    ])
  end

  defp embed_stick_discrete(nil, buckets) do
    # Neutral stick (center)
    center = div(buckets, 2)
    Nx.concatenate([
      Primitives.one_hot(center, size: buckets + 1),
      Primitives.one_hot(center, size: buckets + 1)
    ])
  end

  defp embed_shoulder_discrete(value, buckets) when is_number(value) do
    bucket = discretize_axis(value, buckets)
    Primitives.one_hot(bucket, size: buckets + 1)
  end

  defp embed_shoulder_discrete(_, buckets) do
    Primitives.one_hot(0, size: buckets + 1)
  end

  @doc """
  Discretize a continuous axis value [0, 1] to a bucket index.
  """
  @spec discretize_axis(float(), non_neg_integer()) :: non_neg_integer()
  def discretize_axis(value, buckets) when is_number(value) do
    # Clamp to [0, 1] and convert to bucket
    value = Kernel.max(0.0, Kernel.min(1.0, value))
    bucket = trunc(value * buckets + 0.5)
    Kernel.min(bucket, buckets)
  end

  def discretize_axis(_, _), do: 0

  @doc """
  Convert a bucket index back to continuous value.
  """
  @spec undiscretize_axis(non_neg_integer(), non_neg_integer()) :: float()
  def undiscretize_axis(bucket, buckets) do
    bucket / buckets
  end

  @doc """
  Get action indices for each controller component.

  Returns a map with start/end indices for each component in the
  discretized embedding. Useful for extracting predictions.
  """
  @spec component_indices(config()) :: map()
  def component_indices(config \\ default_config()) do
    buttons_size = 8
    axis_size = config.axis_buckets + 1
    shoulder_size = config.shoulder_buckets + 1

    %{
      buttons: {0, buttons_size},
      main_x: {buttons_size, buttons_size + axis_size},
      main_y: {buttons_size + axis_size, buttons_size + 2 * axis_size},
      c_x: {buttons_size + 2 * axis_size, buttons_size + 3 * axis_size},
      c_y: {buttons_size + 3 * axis_size, buttons_size + 4 * axis_size},
      shoulder: {buttons_size + 4 * axis_size, buttons_size + 4 * axis_size + shoulder_size}
    }
  end

  @doc """
  Decode a sampled controller action back to a ControllerState.

  Takes the raw network outputs (logits or samples) and converts
  them back to a ControllerState that can be sent to the game.
  """
  @spec decode(map(), config()) :: ControllerState.t()
  def decode(samples, config \\ default_config()) do
    %ControllerState{
      main_stick: %{
        x: undiscretize_axis(samples.main_x, config.axis_buckets),
        y: undiscretize_axis(samples.main_y, config.axis_buckets)
      },
      c_stick: %{
        x: undiscretize_axis(samples.c_x, config.axis_buckets),
        y: undiscretize_axis(samples.c_y, config.axis_buckets)
      },
      l_shoulder: undiscretize_axis(samples.shoulder, config.shoulder_buckets),
      r_shoulder: 0.0,
      button_a: samples.buttons.a || false,
      button_b: samples.buttons.b || false,
      button_x: samples.buttons.x || false,
      button_y: samples.buttons.y || false,
      button_z: samples.buttons.z || false,
      button_l: samples.buttons.l || false,
      button_r: samples.buttons.r || false,
      button_d_up: samples.buttons.d_up || false
    }
  end
end
