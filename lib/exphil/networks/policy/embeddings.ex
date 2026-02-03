defmodule ExPhil.Networks.Policy.Embeddings do
  @moduledoc """
  Embedding preprocessing for policy networks.

  Handles learned embeddings for action states and characters:
  - Extracts action/character IDs from input tensors
  - Looks up learned embeddings
  - Concatenates with continuous features

  ## Input Tensor Structure

  The input tensor can contain IDs at the end for learned embeddings:
  ```
  [continuous_features, action_ids (optional), character_ids (optional)]
  ```

  For temporal models (3D input):
  ```
  [batch, seq_len, features]
  ```

  ## Embedding Modes

  - **Action embeddings**: Convert 399-dim one-hot → 64-dim learned
  - **Character embeddings**: Convert 33-dim one-hot → 64-dim learned

  This saves ~670 dimensions while often improving model quality.

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Embeddings.Game` - Game state embedding
  """

  require Axon

  alias ExPhil.Constants

  # Embedding constants from Constants module
  @num_actions Constants.num_actions()
  @num_characters Constants.num_characters()
  @default_num_action_ids 2
  @default_num_character_ids 2

  @doc """
  Build embedding preprocessing that handles both action and character IDs.

  The input tensor has the following structure:
  [continuous_features, action_ids (if learned), character_ids (if learned)]

  This function extracts IDs, looks them up in embedding tables, and
  concatenates everything back together.

  ## Parameters
    - `input` - Axon input layer
    - `embed_size` - Total input size
    - `action_embed_size` - Size of action embedding (nil = no action embedding)
    - `num_action_ids` - Number of action IDs (default: 2)
    - `character_embed_size` - Size of character embedding (nil = no character embedding)
    - `num_character_ids` - Number of character IDs (default: 2)
  """
  @spec build_embedding_preprocessing(
          Axon.t(),
          non_neg_integer(),
          non_neg_integer() | nil,
          non_neg_integer(),
          non_neg_integer() | nil,
          non_neg_integer()
        ) :: Axon.t()
  def build_embedding_preprocessing(
        input,
        embed_size,
        action_embed_size,
        num_action_ids,
        character_embed_size,
        num_character_ids
      ) do
    # Calculate sizes based on what's present
    num_action_slots = if action_embed_size, do: num_action_ids, else: 0
    num_char_slots = if character_embed_size, do: num_character_ids, else: 0
    continuous_size = embed_size - num_action_slots - num_char_slots

    cond do
      # Both action and character embeddings
      action_embed_size && character_embed_size ->
        build_combined_embedding_layer(
          input,
          continuous_size,
          num_action_ids,
          action_embed_size,
          num_character_ids,
          character_embed_size
        )

      # Only action embeddings
      action_embed_size ->
        build_action_embedding_layer(input, embed_size, action_embed_size, num_action_ids)

      # Only character embeddings
      character_embed_size ->
        build_character_embedding_layer(
          input,
          embed_size,
          character_embed_size,
          num_character_ids
        )

      # No learned embeddings
      true ->
        input
    end
  end

  @doc """
  Build combined action and character embedding preprocessing.

  Input tensor structure: [continuous, action_ids, character_ids]
  Output: [continuous, embedded_actions, embedded_characters]
  """
  @spec build_combined_embedding_layer(
          Axon.t(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: Axon.t()
  def build_combined_embedding_layer(
        input,
        continuous_size,
        num_action_ids,
        action_embed_size,
        num_character_ids,
        character_embed_size
      ) do
    # Extract continuous features
    continuous =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, 0, continuous_size, axis: 1)
        end,
        name: "extract_continuous"
      )

    # Extract action IDs (after continuous)
    action_ids =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, continuous_size, num_action_ids, axis: 1)
          |> Nx.as_type(:s32)
        end,
        name: "extract_action_ids"
      )

    # Extract character IDs (after action IDs)
    character_ids =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, continuous_size + num_action_ids, num_character_ids, axis: 1)
          |> Nx.as_type(:s32)
        end,
        name: "extract_character_ids"
      )

    # Embed actions
    action_embeddings =
      Axon.embedding(action_ids, @num_actions, action_embed_size, name: "action_embedding")

    flat_action_embeddings = Axon.flatten(action_embeddings, name: "flatten_action_embeds")

    # Embed characters
    character_embeddings =
      Axon.embedding(character_ids, @num_characters, character_embed_size,
        name: "character_embedding"
      )

    flat_character_embeddings =
      Axon.flatten(character_embeddings, name: "flatten_character_embeds")

    # Concatenate all
    Axon.concatenate([continuous, flat_action_embeddings, flat_character_embeddings],
      name: "concat_with_embeds"
    )
  end

  @doc """
  Build character-only embedding layer.

  Input: [batch, continuous_size + num_character_ids]
  Output: [batch, continuous_size + num_character_ids * character_embed_size]
  """
  @spec build_character_embedding_layer(
          Axon.t(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: Axon.t()
  def build_character_embedding_layer(
        input,
        total_embed_size,
        character_embed_size,
        num_character_ids
      ) do
    continuous_size = total_embed_size - num_character_ids

    # Extract continuous features
    continuous =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, 0, continuous_size, axis: 1)
        end,
        name: "extract_continuous"
      )

    # Extract character IDs
    character_ids =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, continuous_size, num_character_ids, axis: 1)
          |> Nx.as_type(:s32)
        end,
        name: "extract_character_ids"
      )

    # Embed characters
    character_embeddings =
      Axon.embedding(character_ids, @num_characters, character_embed_size,
        name: "character_embedding"
      )

    flat_character_embeddings =
      Axon.flatten(character_embeddings, name: "flatten_character_embeds")

    # Concatenate
    Axon.concatenate([continuous, flat_character_embeddings],
      name: "concat_with_character_embeds"
    )
  end

  @doc """
  Build the action embedding layer that extracts action IDs from the end
  of the input tensor and replaces them with learned embeddings.

  Input: [batch, continuous_size + num_action_ids] where last N are action IDs as floats
  Output: [batch, continuous_size + num_action_ids * action_embed_size]

  ## Parameters
    - `input` - Axon input layer
    - `total_embed_size` - Total input size including action IDs
    - `action_embed_size` - Size of each action's learned embedding
    - `num_action_ids` - Number of action IDs (2 for players, 4 for players + Nana)
  """
  @spec build_action_embedding_layer(
          Axon.t(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: Axon.t()
  def build_action_embedding_layer(input, total_embed_size, action_embed_size, num_action_ids) do
    continuous_size = total_embed_size - num_action_ids

    # Split input into continuous features and action IDs
    # continuous_features: [batch, continuous_size]
    # action_ids: [batch, num_action_ids]
    continuous =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, 0, continuous_size, axis: 1)
        end,
        name: "extract_continuous"
      )

    action_ids =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, continuous_size, num_action_ids, axis: 1)
          # Convert to integers for embedding lookup
          |> Nx.as_type(:s32)
        end,
        name: "extract_action_ids"
      )

    # Create embedding layer for actions
    # Shape: [399, action_embed_size]
    action_embeddings =
      Axon.embedding(action_ids, @num_actions, action_embed_size, name: "action_embedding")

    # Flatten embeddings: [batch, num_action_ids, embed_size] -> [batch, num_action_ids * embed_size]
    flat_action_embeddings = Axon.flatten(action_embeddings, name: "flatten_action_embeds")

    # Concatenate continuous features with embedded actions
    Axon.concatenate([continuous, flat_action_embeddings], name: "concat_with_action_embeds")
  end

  @doc """
  Build temporal embedding preprocessing for sequence inputs.

  Handles 3D inputs [batch, seq_len, features] by extracting IDs along axis 2.

  ## Parameters
    - `original_embed_size` - Size of each frame's embedding
    - `window_size` - Sequence length
    - `action_embed_size` - Size of action embedding (nil = none)
    - `num_action_ids` - Number of action IDs
    - `character_embed_size` - Size of character embedding (nil = none)
    - `num_character_ids` - Number of character IDs

  ## Returns
    Tuple of {input_layer, processed_layer, effective_embed_size}
  """
  @spec build_temporal_embedding_preprocessing(
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer() | nil,
          non_neg_integer(),
          non_neg_integer() | nil,
          non_neg_integer()
        ) :: {Axon.t(), Axon.t(), non_neg_integer()}
  def build_temporal_embedding_preprocessing(
        original_embed_size,
        window_size,
        action_embed_size,
        num_action_ids,
        character_embed_size,
        num_character_ids
      ) do
    # Calculate sizes based on what's present
    num_action_slots = if action_embed_size, do: num_action_ids, else: 0
    num_char_slots = if character_embed_size, do: num_character_ids, else: 0
    continuous_size = original_embed_size - num_action_slots - num_char_slots

    effective_embed_size =
      continuous_size +
        if(action_embed_size, do: num_action_ids * action_embed_size, else: 0) +
        if character_embed_size, do: num_character_ids * character_embed_size, else: 0

    # Input: [batch, seq_len, original_embed_size]
    input = Axon.input("state_sequence", shape: {nil, window_size, original_embed_size})

    # Extract continuous features: [batch, seq_len, continuous_size]
    continuous =
      Axon.nx(
        input,
        fn x ->
          Nx.slice_along_axis(x, 0, continuous_size, axis: 2)
        end,
        name: "temporal_extract_continuous"
      )

    # Build list of embeddings to concatenate
    embeddings = [continuous]

    # Extract and embed action IDs if using learned action embeddings
    embeddings =
      if action_embed_size do
        action_ids =
          Axon.nx(
            input,
            fn x ->
              Nx.slice_along_axis(x, continuous_size, num_action_ids, axis: 2)
              |> Nx.as_type(:s32)
            end,
            name: "temporal_extract_action_ids"
          )

        action_embeddings =
          Axon.embedding(action_ids, @num_actions, action_embed_size,
            name: "temporal_action_embedding"
          )

        flat_action_embeddings =
          Axon.nx(
            action_embeddings,
            fn x ->
              {batch, seq_len, num_ids, emb_size} = Nx.shape(x)
              Nx.reshape(x, {batch, seq_len, num_ids * emb_size})
            end,
            name: "temporal_flatten_action_embeds"
          )

        embeddings ++ [flat_action_embeddings]
      else
        embeddings
      end

    # Extract and embed character IDs if using learned character embeddings
    embeddings =
      if character_embed_size do
        char_offset = continuous_size + num_action_slots

        character_ids =
          Axon.nx(
            input,
            fn x ->
              Nx.slice_along_axis(x, char_offset, num_character_ids, axis: 2)
              |> Nx.as_type(:s32)
            end,
            name: "temporal_extract_character_ids"
          )

        character_embeddings =
          Axon.embedding(character_ids, @num_characters, character_embed_size,
            name: "temporal_character_embedding"
          )

        flat_character_embeddings =
          Axon.nx(
            character_embeddings,
            fn x ->
              {batch, seq_len, num_ids, emb_size} = Nx.shape(x)
              Nx.reshape(x, {batch, seq_len, num_ids * emb_size})
            end,
            name: "temporal_flatten_character_embeds"
          )

        embeddings ++ [flat_character_embeddings]
      else
        embeddings
      end

    # Concatenate all embeddings
    combined =
      if length(embeddings) > 1 do
        Axon.concatenate(embeddings, axis: 2, name: "temporal_concat_with_embeds")
      else
        hd(embeddings)
      end

    {input, combined, effective_embed_size}
  end

  @doc """
  Get default number of action IDs.
  """
  @spec default_num_action_ids() :: non_neg_integer()
  def default_num_action_ids, do: @default_num_action_ids

  @doc """
  Get default number of character IDs.
  """
  @spec default_num_character_ids() :: non_neg_integer()
  def default_num_character_ids, do: @default_num_character_ids
end
