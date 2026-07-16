defmodule ExPhil.Networks.Mamba.Common do
  @moduledoc """
  Shared components for all Mamba architecture variants — thin delegation
  to `Edifice.SSM.Common`.

  This module used to be a full structural clone of the edifice original.
  The clone quietly drifted (2026-07-16 audit): its `build_depthwise_conv1d`
  was a windowed-mean APPROXIMATION of the real depthwise causal conv
  edifice implements, and its scan ranges iterated BACKWARDS at
  `seq_len == 1` instead of being empty. Delegating kills both bugs and the
  duplication; edifice accepts `:embed_size` as an alias for `:embed_dim`,
  so exphil call sites pass through unchanged.

  NOTE checkpoint compatibility: the real depthwise conv registers params
  under `<name>_dw_conv` where the old approximation used
  `<name>_causal`/`<name>_proj` — Mamba-family checkpoints trained before
  2026-07-16 will not load into newly built models (the newera line is
  GRU-backbone and unaffected).

  ## Mamba Variants

  All variants share the same architecture, differing only in the scan
  algorithm:

  | Variant | Scan Algorithm | Notes |
  |---------|---------------|-------|
  | `Mamba` | Blelloch | Work-efficient O(L) work, O(log L) depth |
  | `MambaHillisSteele` | Hillis-Steele | O(L log L) work, more parallelism |
  | `MambaCumsum` | Cumsum-based | Experimental log-space approach |
  | `MambaSSD` | SSD chunked | Mamba-2's matmul approach |
  | `MambaNIF` | CUDA NIF | 5x faster inference via Rust NIF |

  ## See Also

  - `Edifice.SSM.Common` - The implementation
  - `ExPhil.Networks.Mamba` - Main Mamba implementation
  - `ExPhil.Networks.Policy.Backbone` - Uses Mamba as temporal backbone
  """

  alias Edifice.SSM.Common

  # Default hyperparameters
  defdelegate default_hidden_size, to: Common
  defdelegate default_state_size, to: Common
  defdelegate default_expand_factor, to: Common
  defdelegate default_conv_size, to: Common
  defdelegate default_num_layers, to: Common
  defdelegate default_dropout, to: Common
  defdelegate dt_min, to: Common
  defdelegate dt_max, to: Common

  # Model / block builders
  defdelegate build_model(opts, block_builder), to: Common
  defdelegate build_block(input, opts, ssm_builder), to: Common
  defdelegate build_depthwise_conv1d(input, channels, kernel_size, name), to: Common
  defdelegate build_ssm_projections(input, opts), to: Common

  # SSM math
  defdelegate discretize_ssm(x, b, dt, state_size), to: Common
  defdelegate compute_ssm_output(h, c), to: Common
  defdelegate sequential_scan(a, b), to: Common
  defdelegate blelloch_scan(a, b), to: Common

  # Utilities
  defdelegate output_size(opts \\ []), to: Common
  defdelegate param_count(opts), to: Common

  @doc """
  Recommended defaults for Melee gameplay (60fps). Byte-identical to
  `Edifice.SSM.Common.recommended_defaults/0`; kept for call-site clarity.
  """
  @spec melee_defaults() :: keyword()
  defdelegate melee_defaults, to: Common, as: :recommended_defaults
end
