defmodule ExPhil.Training.FlashTraining do
  @moduledoc """
  Custom training loop using FlashAttention NIF for both forward and backward passes.

  This module provides an alternative training path for attention-based models
  (sliding_window, jamba) that uses the FlashAttention NIF instead of Nx attention.
  This can provide 2-3x speedup on Ampere+ GPUs.

  ## Why Custom Loop?

  NIFs cannot participate in Nx autodiff because they require `Nx.to_binary()`
  which severs the computation graph. Instead, we manually chain gradients:

  ```
  Forward:  Embedding → [FlashAttn NIF] → PolicyHead → Loss
                              ↓
                        saves {O, logsumexp}

  Backward: d_loss → PolicyHead.grad → [FlashAttn NIF backward] → Embedding.grad
                                              ↓
                                      uses saved {O, logsumexp}
  ```

  ## Usage

      # Create trainer with flash attention enabled
      trainer = FlashTraining.new(
        embed_config: embed_config,
        policy_config: policy_config,
        learning_rate: 1.0e-4
      )

      # Train step
      {new_trainer, metrics} = FlashTraining.train_step(trainer, batch)

  ## Limitations

  - Cannot use with Axon.Loop (uses custom loop instead)
  - Only supports attention-based backbones (sliding_window, jamba)
  - Requires model to be structured as: embedding → attention → policy_head
  """

  alias ExPhil.Native.FlashAttention
  alias ExPhil.Training.Output

  require Logger

  defstruct [
    # Model components (split for manual gradient flow)
    :embedding_model,
    :embedding_params,
    :qkv_projection,
    :qkv_params,
    :attention_config,
    :post_attention_model,
    :post_attention_params,
    :policy_head_model,
    :policy_head_params,
    # Optimizer
    :optimizer,
    :optimizer_state,
    # Config
    :config,
    :step,
    # Compiled functions
    :embedding_fn,
    :qkv_fn,
    :post_attention_fn,
    :policy_head_fn
  ]

  @default_config %{
    learning_rate: 1.0e-4,
    max_grad_norm: 1.0,
    weight_decay: 1.0e-5,
    num_heads: 4,
    head_dim: 64,
    causal: true
  }

  @type t :: %__MODULE__{}

  @doc """
  Create a new FlashTraining trainer.

  This is a simplified trainer for demonstration. In practice, you would
  integrate this with the full Imitation trainer infrastructure.

  ## Options

  - `:num_heads` - Number of attention heads (default: 4)
  - `:head_dim` - Dimension per head (default: 64)
  - `:hidden_dim` - Hidden dimension (default: 256)
  - `:learning_rate` - Learning rate (default: 1.0e-4)
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = Map.merge(@default_config, Map.new(opts))

    num_heads = config.num_heads
    head_dim = config.head_dim
    hidden_dim = Map.get(config, :hidden_dim, num_heads * head_dim)

    # Build simple QKV projection for demonstration
    # In practice, this would be extracted from the full model
    qkv_model = build_qkv_projection(hidden_dim, num_heads, head_dim)
    {qkv_init_fn, qkv_fn} = Axon.build(qkv_model)

    # Initialize QKV params using Axon.ModelState
    dummy_input = Nx.broadcast(0.0, {1, 1, hidden_dim})
    qkv_model_state = qkv_init_fn.(dummy_input, %{})
    # Extract the underlying parameter map from ModelState
    qkv_params = qkv_model_state.data

    # Build optimizer - returns {init_fn, update_fn}
    {optimizer_init, optimizer_update} =
      Polaris.Optimizers.adamw(
        learning_rate: config.learning_rate,
        decay: config.weight_decay
      )

    optimizer_state = optimizer_init.(qkv_params)

    %__MODULE__{
      qkv_projection: qkv_model,
      qkv_params: qkv_params,
      qkv_fn: qkv_fn,
      attention_config: %{
        num_heads: num_heads,
        head_dim: head_dim,
        causal: config.causal
      },
      optimizer: optimizer_update,
      optimizer_state: optimizer_state,
      config: config,
      step: 0
    }
  end

  @doc """
  Perform one training step using FlashAttention NIF.

  ## Arguments

  - `trainer` - The trainer struct
  - `batch` - Map with `:input` (embedded states) and `:target` tensors

  ## Returns

  `{new_trainer, metrics}` where metrics contains `:loss` and `:step`
  """
  @spec train_step(t(), map()) :: {t(), map()}
  def train_step(trainer, batch) do
    %{input: input, target: target} = batch
    %{num_heads: num_heads, head_dim: head_dim, causal: causal} = trainer.attention_config

    # Get batch dimensions
    {batch_size, seq_len, hidden_dim} = Nx.shape(input)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    # 1. Project to Q, K, V using QKV projection (Nx, differentiable)
    #    This is the "pre-attention" part we can differentiate
    {q, k, v} = forward_qkv(trainer, input)

    # Reshape for attention: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
    q_reshaped = Nx.reshape(q, {batch_size, seq_len, num_heads, head_dim})
    k_reshaped = Nx.reshape(k, {batch_size, seq_len, num_heads, head_dim})
    v_reshaped = Nx.reshape(v, {batch_size, seq_len, num_heads, head_dim})

    # 2. FlashAttention forward (NIF, saves state for backward)
    {:ok, attn_out, logsumexp} =
      FlashAttention.forward_with_states(q_reshaped, k_reshaped, v_reshaped, causal: causal)

    # Reshape back: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
    attn_out_flat = Nx.reshape(attn_out, {batch_size, seq_len, hidden_dim})

    # 3. Simple output projection and loss (for demonstration)
    #    In practice, this would go through the full policy head
    output = attn_out_flat
    loss = compute_mse_loss(output, target)

    # =========================================================================
    # BACKWARD PASS
    # =========================================================================

    # 4. Gradient of loss w.r.t. output
    #    For MSE: d_loss/d_output = 2 * (output - target) / n
    n = Nx.size(output) |> Nx.to_number()
    d_output = Nx.multiply(Nx.subtract(output, target), 2.0 / n)

    # Reshape for attention backward
    d_attn_out = Nx.reshape(d_output, {batch_size, seq_len, num_heads, head_dim})

    # 5. FlashAttention backward (NIF)
    {:ok, dq, dk, dv} =
      FlashAttention.backward(
        d_attn_out,
        q_reshaped,
        k_reshaped,
        v_reshaped,
        attn_out,
        logsumexp,
        causal: causal
      )

    # Reshape gradients back: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
    dq_flat = Nx.reshape(dq, {batch_size, seq_len, hidden_dim})
    dk_flat = Nx.reshape(dk, {batch_size, seq_len, hidden_dim})
    dv_flat = Nx.reshape(dv, {batch_size, seq_len, hidden_dim})

    # 6. Backward through QKV projection (Nx.grad)
    qkv_grads = backward_qkv(trainer, input, dq_flat, dk_flat, dv_flat)

    # =========================================================================
    # OPTIMIZER STEP
    # =========================================================================

    # 7. Apply optimizer updates
    # optimizer is the update function from Polaris
    {updates, new_optimizer_state} =
      trainer.optimizer.(
        qkv_grads,
        trainer.optimizer_state,
        trainer.qkv_params
      )

    new_qkv_params = apply_updates(trainer.qkv_params, updates)

    # Update trainer
    new_trainer = %{
      trainer
      | qkv_params: new_qkv_params,
        optimizer_state: new_optimizer_state,
        step: trainer.step + 1
    }

    loss_value = Nx.to_number(loss)
    {new_trainer, %{loss: loss_value, step: new_trainer.step}}
  end

  @doc """
  Run a simple training loop for testing.
  """
  def train_loop(trainer, batches, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 1)
    log_interval = Keyword.get(opts, :log_interval, 10)

    Enum.reduce(1..epochs, trainer, fn epoch, acc_trainer ->
      Output.puts("Epoch #{epoch}")

      {final_trainer, _} =
        Enum.reduce(Enum.with_index(batches), {acc_trainer, []}, fn {batch, idx},
                                                                    {t, losses} ->
          {new_t, metrics} = train_step(t, batch)

          if rem(idx + 1, log_interval) == 0 do
            avg_loss = metrics.loss
            Output.puts("  Step #{new_t.step}: loss = #{Float.round(avg_loss, 6)}")
          end

          {new_t, [metrics.loss | losses]}
        end)

      final_trainer
    end)
  end

  # ===========================================================================
  # Private Functions
  # ===========================================================================

  # Build QKV projection model
  defp build_qkv_projection(hidden_dim, num_heads, head_dim) do
    qkv_dim = num_heads * head_dim * 3

    Axon.input("input", shape: {nil, nil, hidden_dim})
    |> Axon.dense(qkv_dim, name: "qkv_projection")
  end

  # Forward through QKV projection, returning {Q, K, V}
  defp forward_qkv(trainer, input) do
    %{num_heads: num_heads, head_dim: head_dim} = trainer.attention_config
    hidden_dim = num_heads * head_dim

    # Run QKV projection
    qkv = trainer.qkv_fn.(trainer.qkv_params, input)

    # Split into Q, K, V
    {batch, seq, _} = Nx.shape(qkv)
    qkv_reshaped = Nx.reshape(qkv, {batch, seq, 3, hidden_dim})

    q = qkv_reshaped[[.., .., 0, ..]]
    k = qkv_reshaped[[.., .., 1, ..]]
    v = qkv_reshaped[[.., .., 2, ..]]

    {q, k, v}
  end

  # Backward through QKV projection
  # Returns gradients for QKV projection parameters
  defp backward_qkv(trainer, input, dq, dk, dv) do
    %{num_heads: num_heads, head_dim: head_dim} = trainer.attention_config
    hidden_dim = num_heads * head_dim
    {batch, seq, _} = Nx.shape(input)

    # Combine dq, dk, dv back into d_qkv
    d_qkv =
      Nx.stack([dq, dk, dv], axis: 2)
      |> Nx.reshape({batch, seq, 3 * hidden_dim})

    # Compute gradient of QKV projection w.r.t. its parameters
    # For a linear layer y = xW + b:
    #   dW = x^T @ dy
    #   db = sum(dy, axis=0)
    #   dx = dy @ W^T (not needed here as input is not trainable)

    # Get weight matrix from params
    %{"qkv_projection" => %{"kernel" => _w, "bias" => _b}} = trainer.qkv_params

    # Reshape for matmul: [batch, seq, hidden] -> [batch*seq, hidden]
    input_flat = Nx.reshape(input, {batch * seq, hidden_dim})
    d_qkv_flat = Nx.reshape(d_qkv, {batch * seq, 3 * hidden_dim})

    # dW = input^T @ d_qkv
    dw = Nx.dot(Nx.transpose(input_flat), d_qkv_flat)

    # db = sum(d_qkv, axis=0)
    db = Nx.sum(d_qkv_flat, axes: [0])

    # Return gradients in same structure as params
    %{"qkv_projection" => %{"kernel" => dw, "bias" => db}}
  end

  # Simple MSE loss for demonstration
  defp compute_mse_loss(output, target) do
    diff = Nx.subtract(output, target)
    Nx.mean(Nx.multiply(diff, diff))
  end

  # Apply optimizer updates to parameters
  defp apply_updates(params, updates) do
    Map.new(params, fn {layer_name, layer_params} ->
      layer_updates = updates[layer_name]

      new_layer_params =
        Map.new(layer_params, fn {param_name, param_value} ->
          update = layer_updates[param_name]
          {param_name, Nx.add(param_value, update)}
        end)

      {layer_name, new_layer_params}
    end)
  end
end
