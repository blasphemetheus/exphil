defmodule ExPhil.Training.Config do
  @moduledoc """
  Training configuration parsing and generation.

  Extracts the configuration logic from training scripts into a testable module.
  Handles:
  - Command-line argument parsing
  - Timestamped checkpoint name generation
  - Training config JSON structure

  ## See Also

  - `ExPhil.Training.Imitation` - The main training module that uses this config
  - `ExPhil.Training.Data` - Data loading and batching
  - `ExPhil.Training.Help` - CLI help text generation
  """

  alias ExPhil.Constants
  alias ExPhil.Training.Config.AtomSafety
  alias ExPhil.Training.Config.Checkpoint
  alias ExPhil.Training.Config.Diff
  alias ExPhil.Training.Config.Inference
  alias ExPhil.Training.Config.Parser
  alias ExPhil.Training.Config.Presets
  alias ExPhil.Training.Config.Validator
  alias ExPhil.Training.Config.Yaml

  # Default replays directory - relative path for portability
  # Can be overridden with --replays or --replay-dir
  @default_replays_dir "./replays"
  # Larger network - modern GPUs handle this easily, better capacity
  @default_hidden_sizes [512, 512, 256]

  # Mode allowlists for safe atom conversion (used by YAML module)
  @valid_precision_modes [:f32, :bf16, :f16]

  # Training option allowlists (used by Parser and YAML modules)
  # Note: :hybrid is an alias for :lstm_hybrid (kept for backwards compatibility)
  # INVARIANTS.md item 3 phase B: derived from @backbone_specs (see below).
  # (valid backbones derive from @backbone_specs — see valid_backbones/0)
  @valid_optimizers [:adam, :adamw, :lamb, :radam, :sgd, :rmsprop, :adabelief, :yogi]

  @doc """
  The CLI-selectable backbones. Public so the atom contract between this
  list, the dispatcher, and `backbone_defaults/1` can be tested rather than
  duplicated (test/exphil/harness/backbone_defaults_test.exs) — a typo'd
  defaults key silently trains a backbone with no defaults at all.
  """
  @spec valid_backbones() :: [atom()]
  def valid_backbones, do: Keyword.keys(backbone_specs())
  @valid_lr_schedules [:constant, :cosine, :cosine_restarts, :exponential, :linear]
  # Policy types: how actions are predicted
  # - :autoregressive - Standard 6-head sequential prediction (current default)
  # - :diffusion - DDPM-based iterative denoising (slow but high quality)
  # - :act - Action Chunking with Transformers (fast, predicts sequences)
  # - :flow_matching - ODE-based continuous normalizing flow (fast, simpler than diffusion)
  @valid_policy_types [:autoregressive, :diffusion, :act, :flow_matching]

  # Controller head for :autoregressive policies (AUTOREGRESSIVE_HEAD_PLAN):
  # - :independent - six parallel heads read the trunk (legacy default;
  #   "autoregressive" in older docs meant this)
  # - :autoregressive - residual-stream conditional head: buttons ->
  #   main_x -> main_y -> c_x -> c_y -> shoulder, teacher-forced in training
  @valid_heads [:independent, :autoregressive]

  # Presets are now defined in ExPhil.Training.Config.Presets
  # Use Presets.valid_presets() to get the list

  # All valid CLI flags for argument validation
  # This list is used to detect typos and suggest corrections
  # INVARIANTS.md item 2 phase B (2026-09-09): the accepted-flag list is
  # DERIVED from the parser's flag table (+ its explicit multi-key steps).
  # A flag that is accepted but never parsed, or parsed but rejected at
  # the door, is now unrepresentable. Defaults and docs parity remain
  # pinned by flag_parity_test.
  @valid_flags ExPhil.Training.Config.Parser.flags()

  @doc """
  List of available preset names.

  ## Examples

      iex> presets = ExPhil.Training.Config.available_presets()
      iex> :quick in presets
      true
      iex> :production in presets
      true
      iex> :mewtwo in presets
      true

  """
  @spec available_presets() :: [atom()]
  def available_presets, do: Presets.valid_presets()

  # ---------------------------------------------------------------------------
  # INVARIANTS.md item 3 (phase B, 2026-09-09): THE backbone spec map. Every
  # dispatchable backbone (Networks.Policy.Backbone.build_temporal_backbone)
  # has exactly one row; `@valid_backbones` is DERIVED from these keys and
  # `backbone_defaults/1` is a lookup that RAISES for unknown atoms — so a
  # backbone that is dispatchable-but-rejected, valid-but-undefaulted, or
  # spelled two ways cannot be written. Rows in @untuned_backbones carry
  # the generic baseline and are a RATCHET (backbone_table_parity_test):
  # tune one, remove it from the list; the list may never grow.
  # ---------------------------------------------------------------------------
  @baseline_defaults [
    temporal: true,
    precision: :f32,
    dropout: 0.1,
    window_size: 60,
    num_layers: 2
  ]

  @backbone_specs [
    sliding_window: [temporal: false, precision: :f32, dropout: 0.1],
    attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      num_heads: 4,
      head_dim: 64,
      chunked_attention: true
    ],
    lstm_hybrid: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    jamba: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      learning_rate: 5.0e-6,
      max_grad_norm: 0.25,
      batch_size: 16,
      build:
        {Edifice.SSM.Hybrid, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           expand_factor: 2,
           conv_size: 4,
           num_layers: 6,
           attention_every: 3,
           num_heads: 4,
           head_dim: 64,
           dropout: 0.1,
           window_size: 60,
           use_sliding_window: true,
           seq_len: {:ref, :window_size},
           pre_norm: true,
           qk_layernorm: true
         ]},
      output: {:hidden_size, 256}
    ],
    zamba: [
      temporal: true,
      precision: :bf16,
      dropout: 0.0,
      learning_rate: 1.0e-5,
      max_grad_norm: 0.5,
      build:
        {Edifice.SSM.Zamba, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           expand_factor: 2,
           conv_size: 4,
           num_layers: 6,
           attention_every: 3,
           num_heads: 4,
           head_dim: 64,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    griffin: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      window_size: 60,
      num_layers: 2
    ],
    hawk: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    xlstm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    xlstm_slstm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    xlstm_mlstm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    retnet: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.RetNet, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 6,
           num_heads: 4,
           expand_factor: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    rwkv: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.RWKV, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 6,
           head_size: 64,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    gla: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.GLA, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 6,
           num_heads: 4,
           head_dim: 64,
           expand_factor: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    hgrn: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.HGRN, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 6,
           state_expansion: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    s5: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.S5, :embed_dim,
         [
           hidden_size: 256,
           state_size: 64,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    s4: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.S4, :embed_dim,
         [
           hidden_size: 256,
           state_size: 64,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    s4d: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.S4D, :embed_dim,
         [
           hidden_size: 256,
           state_size: 64,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    h3: [
      temporal: true,
      precision: :f32,
      learning_rate: 5.0e-7,
      max_grad_norm: 0.1,
      build:
        {Edifice.SSM.H3, :embed_dim,
         [
           hidden_size: 256,
           state_size: 64,
           conv_size: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    performer: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Performer, :embed_dim,
         [
           hidden_size: 256,
           num_features: 64,
           num_layers: 4,
           num_heads: 4,
           dropout: 0.1,
           window_size: 60
         ]},
      output: {:hidden_size, 256}
    ],
    deltanet: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.DeltaNet, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    fnet: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.FNet, :embed_dim,
         [hidden_size: 256, num_layers: 4, dropout: 0.1, window_size: 60]},
      output: {:hidden_size, 256}
    ],
    perceiver: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Perceiver, :input_dim,
         [
           latent_dim: 256,
           num_latents: 64,
           num_layers: 4,
           num_cross_layers: 1,
           num_heads: 4,
           dropout: 0.1
         ]},
      output: {:latent_dim, 256}
    ],
    ttt: [
      temporal: true,
      precision: :f32,
      learning_rate: 5.0e-7,
      max_grad_norm: 0.1,
      build:
        {Edifice.Recurrent.TTT, :embed_dim,
         [
           hidden_size: 256,
           inner_size: 64,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    hopfield: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    ntm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    reservoir: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    snn: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    bayesian: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    decision_transformer: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2
    ],
    liquid: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Liquid, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size},
           integration_steps: 1,
           solver: :exact
         ]},
      output: {:hidden_size, 256}
    ],
    kan: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Feedforward.KAN, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 4,
           grid_size: 8,
           basis: :sine,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    transformer_like: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.TransformerLike, :embed_dim,
         [
           hidden_size: 512,
           num_layers: 3,
           cell_type: :lstm,
           ffn_multiplier: 2,
           activation: :gelu,
           dropout: 0.1,
           norm: :layer_norm,
           recurrent_norm: false,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 512}
    ],
    deep_res_lstm: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.DeepResLSTM, :embed_dim,
         [
           hidden_size: 512,
           num_layers: 3,
           dropout: 0.1,
           norm: :layer_norm,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 512}
    ],
    min_gru: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.MinGRU, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    min_lstm: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.MinLSTM, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    tcn: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    mamba3: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.Mamba3, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           expand_factor: 2,
           conv_size: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60
         ]},
      output: {:hidden_size, 256}
    ],
    hyena: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.Hyena, :embed_dim,
         [
           hidden_size: 256,
           order: 2,
           filter_size: 64,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    titans: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.Titans, :embed_dim,
         [
           hidden_size: 256,
           memory_size: 64,
           num_layers: 4,
           momentum: 0.9,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    gated_deltanet: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.GatedDeltaNet, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           conv_size: 4,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]},
      output: {:hidden_size, 256}
    ],
    native_recurrence: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.NativeRecurrence, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    longhorn: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.Longhorn, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    samba: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.Samba, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           num_heads: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    hymba: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.Hymba, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           num_heads: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    gss: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.GSS, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    delta_product: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.DeltaProduct, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    gla_v2: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.GLAv2, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    hgrn_v2: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.HGRNv2, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    ttt_e2e: [
      temporal: true,
      precision: :f32,
      learning_rate: 5.0e-7,
      max_grad_norm: 0.1,
      build:
        {Edifice.Recurrent.TTTE2E, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    gsa: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.GSA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    rla: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.RLA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    nha: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.NHA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    fox: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.FoX, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    log_linear: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.LogLinear, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    laser: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.LASER, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    moba: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.MoBA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    tnn: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.TNN, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 4,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    miras: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.MIRAS, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    mixture_of_mamba: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.MixtureOfMamba, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    huginn: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.Huginn, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    coconut: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Meta.Coconut, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    mega: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Mega, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    based: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Based, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    infini_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.InfiniAttention, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           segment_size: 32,
           dropout: 0.1,
           window_size: 60
         ]}
    ],
    conformer: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Conformer, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           conv_kernel_size: 31,
           dropout: 0.1,
           window_size: 60
         ]}
    ],
    mla: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.MLA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    diff_transformer: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.DiffTransformer, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    megalodon: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.Megalodon, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    lightning_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.LightningAttention, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    flash_linear_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.FlashLinearAttention, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    kda: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.KDA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    sigmoid_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.SigmoidAttention, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    spla: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    retnet_v2: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.RetNetV2, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    rnope_swa: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.RNoPESWA, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    nsa: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.NSA, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    infllm_v2: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    dual_chunk_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.DualChunk, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1]}
    ],
    gated_attention: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.GatedAttention, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    mta: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Attention.MTA, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    slstm: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.SLSTM, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    xlstm_v2: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Recurrent.XLSTMv2, :embed_dim,
         [
           hidden_size: 256,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    bimamba: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.BiMamba, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    hyena_v2: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.HyenaV2, :embed_dim,
         [
           hidden_size: 256,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    ss_transformer: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.SSM.SSTransformer, :embed_dim,
         [
           hidden_size: 256,
           state_size: 16,
           num_heads: 4,
           num_layers: 2,
           dropout: 0.1,
           window_size: 60,
           seq_len: {:ref, :window_size}
         ]}
    ],
    ssmax: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Blocks.SSMax, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    softpick: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2,
      build:
        {Edifice.Blocks.Softpick, :embed_dim,
         [hidden_size: 256, num_heads: 4, num_layers: 2, dropout: 0.1, window_size: 60]}
    ],
    lstm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    gru: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    gated_ssm: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    mamba: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      window_size: 60,
      num_layers: 2,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4
    ],
    mamba_2: [
      temporal: true,
      precision: :f32,
      dropout: 0.0,
      lr_schedule: :cosine_restarts,
      window_size: 60,
      num_layers: 2,
      state_size: 16,
      expand_factor: 2
    ],
    mamba_nif: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    mamba_cumsum: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    mamba_hillis_steele: [
      temporal: true,
      precision: :f32,
      dropout: 0.1,
      window_size: 60,
      num_layers: 2
    ],
    mamba_ssd: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2],
    mlp: [temporal: false, dropout: 0.1, precision: :f32],
    hybrid: [temporal: true, precision: :f32, dropout: 0.1, window_size: 60, num_layers: 2]
  ]

  @untuned_backbones ~w(lstm_hybrid hawk s5 s4 s4d fnet perceiver hopfield ntm reservoir snn bayesian decision_transformer liquid kan transformer_like deep_res_lstm tcn hyena titans native_recurrence longhorn samba hymba gss delta_product gla_v2 hgrn_v2 gsa rla nha fox log_linear laser moba tnn miras mixture_of_mamba huginn coconut mega based infini_attention conformer mla diff_transformer megalodon lightning_attention flash_linear_attention kda sigmoid_attention spla retnet_v2 rnope_swa nsa infllm_v2 dual_chunk_attention gated_attention mta slstm xlstm_v2 bimamba hyena_v2 ss_transformer ssmax softpick gated_ssm mamba_nif mamba_cumsum mamba_hillis_steele mamba_ssd)a

  @doc "The backbone spec map (atom => training defaults) — the single source for valid_backbones/0 and backbone_defaults/1."
  @spec backbone_specs() :: keyword()
  def backbone_specs, do: @backbone_specs

  @doc "Backbones carrying only the generic baseline defaults (ratchet: may only shrink)."
  @spec untuned_backbones() :: [atom()]
  def untuned_backbones, do: @untuned_backbones

  @doc "True when `backbone` has no tuned defaults (baseline only)."
  @spec backbone_defaults_baseline?(atom()) :: boolean()
  def backbone_defaults_baseline?(backbone), do: backbone in @untuned_backbones

  # INVARIANTS.md item 3 phase C: a row may also carry the CONSTRUCTION
  # recipe (`build: {module, embed_key, params}`) and the output-size rule
  # (`output: {opt_key, default}`) that Networks.Policy.Backbone reads for
  # every backbone without a bespoke dispatcher clause. They are stripped
  # from the training defaults so they never reach opts / the checkpoint JSON.
  @spec_only_keys [:build, :output]

  @doc "The full spec row for a backbone (training defaults + build recipe + output rule); unknown atoms raise."
  @spec backbone_spec(atom()) :: keyword()
  def backbone_spec(backbone) do
    case Keyword.fetch(@backbone_specs, backbone) do
      {:ok, row} ->
        row

      :error ->
        raise ArgumentError,
              "unknown backbone #{inspect(backbone)} — not in @backbone_specs " <>
                "(valid: #{inspect(Keyword.keys(@backbone_specs))})"
    end
  end

  @doc "Per-backbone training defaults (CLI args win). Every valid backbone has a row in @backbone_specs; an unknown atom raises. Tuned rows' source: benchmark_architectures.exs."
  @spec backbone_defaults(atom()) :: keyword()
  def backbone_defaults(backbone),
    do: backbone |> backbone_spec() |> Keyword.drop(@spec_only_keys)

  @doc "The `{module, embed_key, params}` construction recipe for a spec-built backbone, or nil for bespoke ones."
  @spec backbone_recipe(atom()) :: {module(), atom(), keyword()} | nil
  def backbone_recipe(backbone), do: backbone |> backbone_spec() |> Keyword.get(:build)

  @doc "The `{opt_key, default}` output-size rule for a spec-built backbone, or nil."
  @spec backbone_output_rule(atom()) :: {atom(), pos_integer()} | nil
  def backbone_output_rule(backbone), do: backbone |> backbone_spec() |> Keyword.get(:output)

  @doc "Backbones whose construction is a spec recipe (no bespoke dispatcher clause)."
  @spec spec_built_backbones() :: [atom()]
  def spec_built_backbones,
    do: for({b, row} <- @backbone_specs, Keyword.has_key?(row, :build), do: b)

  @doc """
  Default training options.

  ## Examples

      iex> opts = ExPhil.Training.Config.defaults()
      iex> opts[:epochs]
      10
      iex> opts[:batch_size]
      64
      iex> opts[:temporal]
      false

  """
  @spec defaults() :: keyword()
  def defaults do
    [
      replays: @default_replays_dir,
      # Pre-built MmapCorpus directory (scripts/build_corpus.exs) —
      # bypasses parse/embed entirely; overrides :replays when set
      corpus: nil,
      epochs: 10,
      batch_size: 64,
      hidden_sizes: @default_hidden_sizes,
      max_files: nil,
      # Error handling for bad replay files
      # Continue past bad files (default: true for convenience)
      skip_errors: true,
      # Show individual file errors (default: true)
      show_errors: true,
      # Optional file path to log errors
      error_log: nil,
      checkpoint: nil,
      player_port: 1,
      # Auto-select port based on character (e.g., :mewtwo)
      train_character: nil,
      # With --train-character: resolve the imitated port per file to that
      # character's player (E1 fix — without it the streaming loader imitates
      # port 1 regardless of who sits there). Streaming pipeline only.
      select_character_port: false,
      # Train on both ports (doubles training data)
      dual_port: false,
      # Weight sampling by inverse character frequency
      balance_characters: false,
      wandb: false,
      wandb_project: "exphil",
      wandb_name: nil,
      temporal: false,
      backbone: :sliding_window,
      # Policy type: :autoregressive, :diffusion, :act, :flow_matching
      policy_type: :autoregressive,
      # Controller head: :independent (six parallel heads) or :autoregressive
      # (residual-stream conditional head, AUTOREGRESSIVE_HEAD_PLAN)
      head: :independent,
      # Action horizon for ACT/generative policies (frames to predict at once)
      action_horizon: 8,
      # Number of steps for diffusion/flow inference (more = higher quality, slower)
      num_inference_steps: 20,
      # KL weight (β) for ACT CVAE regularization
      kl_weight: 10.0,
      window_size: 60,
      stride: 5,
      num_layers: 2,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      # Jamba stability options (Pre-LN + QK LayerNorm prevent NaN)
      pre_norm: true,
      qk_layernorm: true,
      # Chunked attention for reduced memory (20-30% savings)
      # Processes queries in chunks against all keys - same results, lower peak memory
      chunked_attention: false,
      chunk_size: 32,
      # Memory-efficient attention (true O(n) memory via online softmax)
      # Slower than chunked but uses less memory for very long sequences
      memory_efficient_attention: false,
      # FlashAttention NIF for inference (forward-only, no gradients)
      # Requires Ampere+ GPU (RTX 30xx/40xx, A100, H100) for CUDA acceleration
      # Falls back to CPU if CUDA unavailable (slower than Pure Nx due to copy overhead)
      flash_attention_nif: false,
      truncate_bptt: nil,
      # Contiguous-BPTT training (BPTT_LOADER_DESIGN.md): cursors walk
      # replays in order, GRU carry flows across chunks, per-timestep loss
      bptt: false,
      unroll: 80,
      bptt_overlap: 1,
      bptt_val_files: 16,
      # FP32 is default - benchmarks show BF16 is 2x SLOWER on RTX 4090 due to
      # XLA issues: dimension misalignment (287 dims not divisible by 16),
      # type casting overhead, and fallback to FP32 kernels internally.
      # See: https://github.com/openxla/xla/issues/12429
      precision: :f32,
      # Mixed precision training (FP32 master weights + BF16 compute)
      # Not recommended - adds overhead without tensor core benefits on current XLA
      mixed_precision: false,
      # INVARIANTS.md item 1 / GOTCHA #113: REACTION delay for the streaming
      # /bptt path, on top of the parser's causal pairing (state[t] ->
      # controller[t+1] is what Peppi emits; k adds k frames). 0 = the
      # causal pairing. The leaked pairing is unrepresentable since
      # 2026-09-09 (ExPhil.Data.LabelConvention); live --frame-delay N
      # plays reaction delay N-1.
      frame_delay: nil,
      label_delay: nil,
      # Stage internals in the embedding (FoD platform heights + PS
      # transformation; W4 2026-08-24 stage-blindness verdict). +7 raw
      # dims, zero-gated by stage. Enable with --stage-internals.
      stage_internals: false,
      # Bucketized action-frame one-hot per player (0 = off; the jab-chain
      # lever, V2_PREP 7b). Layout key: stamped in the checkpoint, rebuilt
      # by the Agent. Enable with --action-frame-buckets N (24 = f0..22 +
      # overflow; covers every startup/chain window that matters).
      action_frame_buckets: 0,
      # Frame delay augmentation for online robustness
      # Enable with --frame-delay-augment or --online-robust
      frame_delay_augment: false,
      # Local play - no delay
      frame_delay_min: 0,
      # Online play - typical Slippi delay (uses Constants.online_frame_delay())
      frame_delay_max: Constants.online_frame_delay(),
      preset: nil,
      character: nil,
      # Early stopping
      early_stopping: false,
      patience: 5,
      min_delta: 0.01,
      # Checkpointing
      save_best: true,
      save_every: nil,
      # Save checkpoint every N batches (useful for streaming mode)
      save_every_batches: nil,
      # Learning rate
      learning_rate: 1.0e-4,
      lr_schedule: :constant,
      # 1 instead of 0 to avoid Polaris/Nx 0.10 compatibility bug
      warmup_steps: 1,
      decay_steps: nil,
      # Cosine restarts (SGDR)
      # Initial period before first restart (T_0)
      restart_period: 1000,
      # Multiply period by this after each restart (T_mult)
      restart_mult: 2,
      # Gradient clipping
      # Clip gradients by global norm (0 = disabled)
      max_grad_norm: 1.0,
      # Resumption
      resume: nil,
      # With --resume: re-initialise the controller head from scratch while
      # loading the trunk (the v1.1-IND control; automatic when the heads
      # differ, e.g. --resume ep10 --head autoregressive)
      reinit_head: false,
      # Model naming
      name: nil,
      # Gradient accumulation
      accumulation_steps: 1,
      # Validation split (10% held out by default for generalization tracking)
      val_split: 0.1,
      # Data augmentation - mirror states doubles effective data
      # Off by default since it disables precompute; use --augment for serious training
      augment: false,
      mirror_prob: 0.5,
      noise_prob: 0.3,
      noise_scale: 0.01,
      # Label smoothing - prevents overconfidence, standard best practice
      # 0.0 = no smoothing, 0.1 = typical value
      label_smoothing: 0.1,
      # Dropout rate (0.0 = no dropout, 0.1 = typical)
      dropout: 0.0,
      # Focal loss for rare actions (Z, L, R buttons)
      # Enabled by default to prevent mode collapse on button predictions
      focal_loss: true,
      use_prev_action: false,
      # Fraction of frames whose prev-action channel is zeroed during training
      # (exposure-bias mitigation: live the model eats its own outputs, which
      # drift from ground truth — dropout stops it over-relying on the channel)
      prev_action_dropout: 0.0,
      # Scheduled sampling (exposure bias): fraction of samples whose LAST
      # window position's prev-action slice is replaced by the model's own
      # decoded prediction (ExPhil.Training.ScheduledSampling). Requires
      # --temporal and --prev-action. 0.0 = off. Ramped 0 -> P over
      # ss_ramp epochs by the drill loop; the main pipeline applies P flat.
      scheduled_sampling: 0.0,
      ss_ramp: 10,
      # Curriculum mixing: comma/glob list of drill .frames exports
      # (scripts/export_drill_frames.exs) concatenated into training
      mix_frames: nil,
      # Corpus-mode curriculum mixing: a second MmapCorpus dir
      # (scripts/build_snippet_corpus.exs) whose batches are interleaved
      # into the training stream — corpus mode ignores :mix_frames, and
      # unlike it, mini-corpus windows never cross snippet boundaries
      mix_corpus: nil,
      # How many passes of the mix corpus to interleave per epoch (the
      # mix is typically ~0.05% of the main corpus — oversample to give
      # corrections a meaningful gradient share, e.g. 20)
      mix_oversample: 1,
      # Task #25: real per-stage edge x in the ledge-distance feature
      # (default-off; existing checkpoints are calibrated to the
      # 85-everywhere constant — only fresh v3-edge arms opt in, and a
      # corpus must be REBUILT with this flag for corpus-mode training)
      per_stage_ledge: false,
      # Same concept for the standard (non-streaming) path: Data.shift_actions
      # adds action_delay frames of reaction delay on top of the causal
      # pairing. 0 = causal. See frame_delay above; the two keys are one
      # concept (FIXES.md P1).
      action_delay: nil,
      # Attention geometry (INVARIANTS.md item 2, 2026-09-09): these were
      # absent here, so `--num-heads` was accepted-and-ignored and
      # Trainer's private table (2/32) silently won over the documented
      # 4/64 for every backbone without a backbone_defaults clause.
      num_heads: 4,
      head_dim: 64,
      # --log-file PATH: tee Output to a file (train.exs consumes it)
      log_file: nil,
      # Higher = more focus on hard examples
      focal_gamma: 3.0,
      # Button loss weight: multiply button loss to balance vs 5 stick/shoulder losses
      # 2.0 fixes typical under-prediction of buttons; use 3.0+ for action-heavy characters
      button_weight: 2.0,
      # Per-button positive class weights for BCE loss (addresses mode collapse)
      # :auto = compute from training data (recommended), nil = equal weights, or [8] float list
      # e.g., [9,19,15,33,49,8,19,99] for manual inverse-frequency weighting
      button_pos_weight: :auto,
      # Stick edge bucket weight: weight edge buckets (0, 16) higher than center (8)
      # Addresses center-bias where model predicts neutral 95%+ of the time
      # 2.0 = edges weighted 2x center, linearly interpolated
      stick_edge_weight: 2.0,
      # Entropy regularization: penalize collapsed output distributions
      # Prevents mode collapse on large datasets (200+ files) where the model
      # defaults to predicting neutral for everything. 0.01 is the tested value.
      entropy_weight: 0.01,
      # Per-frame neutral weight: action frames get 1.0, neutral frames get this value
      # Lower = stronger anti-collapse signal. 0.0 = skip neutral frames entirely.
      neutral_weight: 0.25,
      # Per-frame DECISION weight: frames whose controller differs from the
      # previous frame get max(weight, transition_weight) — emphasizes WHEN
      # to change action (leaving WAIT, committing) instead of blanket
      # downweighting neutral frames. nil = off. Was plumbed through the
      # pipeline and drills but had no train.exs flag until 2026-09-07.
      transition_weight: nil,
      # Per-frame OFFSTAGE weight (bptt path): frames where the subject is
      # airborne beyond the ledge get max(weight, offstage_weight) —
      # rare-state coverage for recovery (V2_PREP 09-08). nil = off.
      offstage_weight: nil,
      # AWBC (advantage-weighted BC) loss weights: reweight the imitation loss
      # by observed outcomes. --awbc-reward standard uses Rewards.Standard
      # (stock + damage); default :shine is the multishine specialist signal.
      awbc: false,
      awbc_reward: :shine,
      awbc_beta: nil,
      awbc_shuffle: false,
      # Per-head loss normalization: equalize gradient contribution from each head
      head_normalize: false,
      # Action-conditional oversampling: frames with button presses appear N× more often
      # nil = disabled, 3.0 = button-press frames sampled 3× more often
      action_oversample: 3.0,
      # Lazy sequence batching: slice from frame embeddings on-the-fly instead of
      # pre-building all sequences in RAM. Trades ~10-20% speed for massive RAM savings.
      lazy_sequences: true,
      # Use Nx.Batch for lazy batch assembly (defers concat to JIT boundary)
      use_batch: false,
      # Registry
      no_register: false,
      # Checkpoint pruning
      # nil = no pruning, N = keep best N epoch checkpoints
      keep_best: nil,
      # Model EMA
      ema: false,
      ema_decay: 0.999,
      # Embedding precomputation (2-3x speedup for MLP training)
      # Precompute embeddings for 2-3x speedup (auto-disabled with augmentation)
      precompute: true,
      # Override for explicitly disabling precomputation
      no_precompute: false,
      # Embedding disk caching (save precomputed embeddings to disk for reuse)
      # Default: true - caches embeddings to disk for faster subsequent runs
      cache_embeddings: true,
      # Force recompute even if cache exists
      no_cache: false,
      # Directory for embedding cache files
      cache_dir: "cache/embeddings",
      # Augmented embedding cache (precompute original + mirrored + noisy variants)
      # When true, enables ~100x speedup for --augment training
      cache_augmented: false,
      # Number of noisy variants to precompute (only used with cache_augmented)
      num_noisy_variants: 2,
      # Data prefetching (load next batch while GPU trains)
      # Only effective with --stream-chunk-size (streaming mode)
      prefetch: false,
      # Number of batches to prefetch
      prefetch_buffer: 2,
      # Layer normalization for MLP backbone
      layer_norm: false,
      # Residual connections for MLP backbone (enables deeper networks, +5-15% accuracy)
      residual: false,
      # Optimizer selection
      # :adam, :adamw, :lamb, :radam
      optimizer: :adam,
      # Gradient checkpointing (memory vs compute trade-off)
      gradient_checkpoint: false,
      # Checkpoint every N layers (1 = every layer, 2 = every other)
      checkpoint_every: 1,
      # Dry run mode - validate config without training
      dry_run: false,
      # Replay filtering
      # Filter replays by character (e.g., [:mewtwo, :fox])
      characters: [],
      # Filter replays by stage (e.g., [:battlefield, :fd])
      stages: [],
      # K-means stick discretization
      # Path to K-means cluster centers file (.nx)
      kmeans_centers: nil,
      # Streaming data loading (process files in chunks to bound memory)
      # nil = load all at once, N = process N files per chunk
      stream_chunk_size: nil,
      # Pipeline chunk preparation (prepare N+1 while training on N)
      pipeline_chunks: true,
      # Cache streaming embeddings to disk (reuse across epochs)
      # No downside, massive speedup on epoch 2+
      cache_streaming: true,
      # Embedding options
      # Stage: :one_hot_full (64 dims), :one_hot_compact (7 dims), :learned (1 ID)
      stage_mode: :one_hot_compact,
      # Action: :one_hot (399 dims per player) or :learned (64-dim trainable, 2 IDs)
      action_mode: :learned,
      # Character: :one_hot (33 dims per player) or :learned (64-dim trainable, 2 IDs)
      character_mode: :learned,
      # Nana (Ice Climbers): :compact (39 dims), :enhanced (14 + ID), :full (449 dims)
      nana_mode: :compact,
      # Jumps: true = normalized (1 dim), false = one_hot (7 dims)
      jumps_normalized: true,
      # Player name embedding dims (0 = disable, 112 = slippi-ai compatible)
      num_player_names: 112,
      # Enable style-conditional training
      learn_player_styles: false,
      # Path to save/load player registry JSON
      player_registry: nil,
      # Minimum games for player to be in registry
      min_player_games: 1,
      # Verbosity control
      # 0 = quiet (errors only), 1 = normal, 2 = verbose (debug)
      verbosity: 1,
      # Progress bar update interval (batches between updates)
      # Higher = less log spam, faster training (less IO)
      # Default 100 keeps logs readable while still showing progress
      log_interval: 100,
      # Reproducibility
      # Random seed (nil = generate from entropy)
      seed: nil,
      # Checkpoint safety
      # Allow overwriting existing checkpoints
      overwrite: false,
      # Create .bak before overwrite
      backup: true,
      # Number of backup versions to keep
      backup_count: 3,
      # Duplicate detection
      # Skip duplicate replay files by hash
      skip_duplicates: true,
      # Replay quality filtering
      # nil = no quality filtering, N = minimum score (0-100)
      min_quality: nil,
      # Show quality distribution after filtering
      show_quality_stats: false,
      # Memory management
      # Run garbage collection every N batches (0 = disabled)
      gc_every: 100,
      # Profiling
      # Enable detailed timing profiler
      profile: false,
      # Parallel validation
      # Number of concurrent batches during validation (1 = sequential)
      val_concurrency: 4,
      # Memory-mapped embeddings for datasets larger than RAM
      # false = disabled, true = auto path, string = custom path
      mmap_embeddings: false,
      # Explicit path for mmap file (overrides auto-generated path)
      mmap_path: nil,
      # Batch size auto-tuning (find optimal batch size for GPU)
      auto_batch_size: false,
      # Minimum batch size to test
      auto_batch_min: 32,
      # Maximum batch size to test
      auto_batch_max: 4096,
      # Safety factor after finding largest working size (0.8 = 20% headroom)
      auto_batch_backoff: 0.8
    ]
    |> apply_env_defaults()
  end

  # Apply environment variable defaults (lower priority than CLI args)
  defp apply_env_defaults(opts) do
    opts
    |> maybe_env(:replays, "EXPHIL_REPLAYS_DIR")
    |> maybe_env(:wandb_project, "EXPHIL_WANDB_PROJECT")
    |> maybe_env_preset("EXPHIL_DEFAULT_PRESET")
  end

  defp maybe_env(opts, key, env_var) do
    case System.get_env(env_var) do
      nil -> opts
      # Override default with env var
      value -> Keyword.put(opts, key, value)
    end
  end

  defp maybe_env_preset(opts, env_var) do
    case System.get_env(env_var) do
      nil ->
        opts

      value ->
        case AtomSafety.safe_to_atom(value, Presets.valid_presets()) do
          {:ok, preset_atom} -> Keyword.put_new(opts, :preset, preset_atom)
          {:error, _} -> opts
        end
    end
  end

  # Character name mappings (atom -> display name, also accepts aliases)
  @character_map %{
    captain_falcon: "Captain Falcon",
    falcon: "Captain Falcon",
    donkey_kong: "Donkey Kong",
    dk: "Donkey Kong",
    fox: "Fox",
    game_and_watch: "Game & Watch",
    gnw: "Game & Watch",
    gameandwatch: "Game & Watch",
    kirby: "Kirby",
    bowser: "Bowser",
    link: "Link",
    luigi: "Luigi",
    mario: "Mario",
    marth: "Marth",
    mewtwo: "Mewtwo",
    ness: "Ness",
    peach: "Peach",
    pikachu: "Pikachu",
    pika: "Pikachu",
    ice_climbers: "Ice Climbers",
    ics: "Ice Climbers",
    icies: "Ice Climbers",
    jigglypuff: "Jigglypuff",
    puff: "Jigglypuff",
    jiggs: "Jigglypuff",
    samus: "Samus",
    yoshi: "Yoshi",
    zelda: "Zelda",
    sheik: "Sheik",
    falco: "Falco",
    young_link: "Young Link",
    ylink: "Young Link",
    dr_mario: "Dr. Mario",
    doc: "Dr. Mario",
    roy: "Roy",
    pichu: "Pichu",
    ganondorf: "Ganondorf",
    ganon: "Ganondorf"
  }

  # Stage name mappings (atom -> {display name, stage ID})
  @stage_map %{
    fountain_of_dreams: {"Fountain of Dreams", 2},
    fod: {"Fountain of Dreams", 2},
    fountain: {"Fountain of Dreams", 2},
    pokemon_stadium: {"Pokemon Stadium", 3},
    ps: {"Pokemon Stadium", 3},
    stadium: {"Pokemon Stadium", 3},
    yoshis_story: {"Yoshi's Story", 8},
    yoshis: {"Yoshi's Story", 8},
    ys: {"Yoshi's Story", 8},
    dream_land: {"Dream Land", 28},
    dreamland: {"Dream Land", 28},
    dl: {"Dream Land", 28},
    battlefield: {"Battlefield", 31},
    bf: {"Battlefield", 31},
    final_destination: {"Final Destination", 32},
    fd: {"Final Destination", 32}
  }

  @doc "Get display name for a character atom"
  def character_name(char) when is_atom(char), do: Map.get(@character_map, char, to_string(char))

  @doc "Get display name and ID for a stage atom"
  def stage_info(stage) when is_atom(stage), do: Map.get(@stage_map, stage)

  @doc "Get stage ID for a stage atom"
  def stage_id(stage) when is_atom(stage) do
    case Map.get(@stage_map, stage) do
      {_name, id} -> id
      nil -> nil
    end
  end

  @doc "List of valid character atoms"
  def valid_characters, do: Map.keys(@character_map)

  @doc "List of valid stage atoms"
  def valid_stages, do: Map.keys(@stage_map)

  # ============================================================================
  # Config File Loading (YAML)
  # ============================================================================

  # Delegated to ExPhil.Training.Config.Yaml
  # See that module for implementation details

  @doc """
  Load training configuration from a YAML file.
  Delegates to `ExPhil.Training.Config.Yaml.load/2`.
  """
  @spec load_yaml(String.t()) :: {:ok, keyword()} | {:error, atom() | String.t()}
  def load_yaml(path) do
    Yaml.load(path, yaml_context())
  end

  @doc """
  Load and merge YAML config with CLI args.
  CLI args take precedence over YAML config.
  """
  @spec load_with_yaml(String.t(), [String.t()]) :: {:ok, keyword()} | {:error, any()}
  def load_with_yaml(yaml_path, cli_args) do
    with {:ok, _yaml_opts} <- load_yaml(yaml_path) do
      {:ok, parse_args(["--config", yaml_path | cli_args])}
    end
  end

  @doc """
  Parse YAML content into training options.
  Delegates to `ExPhil.Training.Config.Yaml.parse/2`.
  """
  @spec parse_yaml(String.t()) :: {:ok, keyword()} | {:error, any()}
  def parse_yaml(content) do
    Yaml.parse(content, yaml_context())
  end

  @doc """
  Save current configuration to a YAML file.
  Delegates to `ExPhil.Training.Config.Yaml.save/2`.
  """
  @spec save_yaml(keyword(), String.t()) :: :ok | {:error, any()}
  def save_yaml(opts, path) do
    Yaml.save(opts, path)
  end

  # Build context for YAML parsing with allowlists
  defp yaml_context do
    %{
      valid_backbones: valid_backbones(),
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules,
      valid_precision_modes: @valid_precision_modes,
      valid_presets: Presets.valid_presets(),
      valid_characters: Map.keys(@character_map),
      valid_stages: Map.keys(@stage_map)
    }
  end

  # ============================================================================
  # Training Presets
  # ============================================================================

  @doc """
  Get training options for a preset.

  ## Available Presets

  ### CPU Presets (No GPU Required)
  - `:quick` - Fast iteration for testing (1 epoch, 5 files, small MLP)
  - `:standard` - Balanced CPU training (10 epochs, 50 files, augmentation)
  - `:full_cpu` - Maximum CPU quality (20 epochs, 100 files, all regularization)

  ### GPU Presets (Requires CUDA/ROCm)
  - `:gpu_quick` - Fast GPU test (3 epochs, 20 files, Mamba temporal)
  - `:gpu_mlp_quick` - Fastest GPU test (5 epochs, 50 files, MLP + precompute, 2-3x faster)
  - `:gpu_lstm_quick` - LSTM backbone test (3 epochs, 30 files)
  - `:gpu_gru_quick` - GRU backbone test (3 epochs, 30 files)
  - `:gpu_attention_quick` - Attention backbone test (3 epochs, 30 files)
  - `:gpu_standard` - Standard GPU training (20 epochs, Mamba, all features)
  - `:full` - High quality GPU (50 epochs, Mamba, temporal, full regularization)
  - `:production` - Maximum quality (100 epochs, Mamba, all optimizations, EMA)

  ### Character Presets (Built on :production)
  - `:mewtwo` - Longer context (90 frames) for teleport recovery tracking
  - `:ganondorf` - Standard context (60 frames) for spacing-focused play
  - `:link` - Extended context (75 frames) for projectile tracking
  - `:gameandwatch` - Shorter context (45 frames) since no L-cancel
  - `:zelda` - Standard context (60 frames) for transform mechanics

  ## Best Practices Applied

  | Feature | quick | standard | full | production |
  |---------|-------|----------|------|------------|
  | Augmentation | ✗ | ✓ | ✓ | ✓ |
  | Label Smoothing | ✗ | 0.05 | 0.1 | 0.1 |
  | EMA | ✗ | ✗ | ✓ | ✓ |
  | LR Schedule | constant | cosine | cosine | cosine_restarts |
  | Val Split | ✗ | 0.1 | 0.1 | 0.15 |
  | Early Stopping | ✗ | ✓ | ✓ | ✓ |

  ## Examples

      iex> opts = ExPhil.Training.Config.preset(:quick)
      iex> opts[:epochs]
      1
      iex> opts[:max_files]
      5
      iex> opts[:temporal]
      false

      iex> opts = ExPhil.Training.Config.preset(:production)
      iex> opts[:epochs]
      100
      iex> opts[:ema]
      true
      iex> opts[:lr_schedule]
      :cosine_restarts

      iex> opts = ExPhil.Training.Config.preset(:mewtwo)
      iex> opts[:character]
      :mewtwo
      iex> opts[:window_size]
      90

  """
  @spec preset(atom() | String.t()) :: keyword() | no_return()
  defdelegate preset(name), to: Presets, as: :get

  # ============================================================================
  # Config Diff Display
  # ============================================================================

  # Delegated to ExPhil.Training.Config.Diff
  # See that module for implementation details

  @doc """
  Get a list of options that differ from defaults.
  Delegates to `ExPhil.Training.Config.Diff.from_defaults/3`.
  """
  @spec diff_from_defaults(keyword(), keyword()) :: [{atom(), any(), any()}]
  def diff_from_defaults(opts, diff_opts \\ []) do
    Diff.from_defaults(opts, &defaults/0, diff_opts)
  end

  @doc """
  Format config diff as a human-readable string.
  Delegates to `ExPhil.Training.Config.Diff.format/2`.
  """
  @spec format_diff(keyword()) :: String.t() | nil
  def format_diff(opts) do
    Diff.format(opts, &defaults/0)
  end

  # ============================================================================
  # Validation
  # ============================================================================
  # Validation logic is in ExPhil.Training.Config.Validator

  @doc """
  Validate training options and return errors/warnings.

  Returns `{:ok, opts}` if valid, or `{:error, errors}` if invalid.
  Warnings are logged but don't cause validation to fail.

  ## Examples

      iex> Config.validate(epochs: 10, batch_size: 64)
      {:ok, [epochs: 10, batch_size: 64]}

      iex> Config.validate(epochs: -1)
      {:error, ["epochs must be positive, got: -1"]}

  """
  @spec validate(keyword()) :: {:ok, keyword()} | {:error, [String.t()]}
  def validate(opts) do
    with {:ok, opts} <- Validator.validate(opts, validation_context()) do
      try do
        ExPhil.Training.LabelDelay.resolve!(opts)
        {:ok, opts}
      rescue
        error in ArgumentError -> {:error, [Exception.message(error)]}
      end
    end
  end

  @doc """
  Validate training options, raising on errors.

  Returns opts if valid, raises `ArgumentError` if invalid.
  Warnings are logged but don't cause validation to fail.

  ## Examples

      iex> ExPhil.Training.Config.validate!(epochs: 10, batch_size: 64)
      [epochs: 10, batch_size: 64]

  Invalid configurations raise an ArgumentError:

      Config.validate!(epochs: -1)
      # => raises ArgumentError with "Invalid training configuration..."

  """
  @spec validate!(keyword()) :: keyword()
  def validate!(opts) do
    opts = Validator.validate!(opts, validation_context())
    ExPhil.Training.LabelDelay.resolve!(opts)
    opts
  end

  # Build the validation context with allowlists
  defp validation_context do
    %{
      valid_backbones: valid_backbones(),
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules,
      valid_policy_types: @valid_policy_types,
      valid_heads: @valid_heads
    }
  end

  @doc """
  Apply a preset to options, allowing CLI args to override preset values.

  Preset values serve as defaults, but any explicitly provided CLI arguments
  take precedence.

  ## Examples

      # Preset provides epochs: 1, but CLI overrides with epochs: 5
      iex> opts = Config.parse_args(["--preset", "quick", "--epochs", "5"])
      iex> opts[:epochs]
      5
      iex> opts[:hidden_sizes]
      [32, 32]  # From preset

  """
  def apply_preset(opts, args) do
    case Parser.get_arg_value(args, "--preset") do
      nil ->
        opts

      preset_name ->
        preset_opts = preset(preset_name)

        # Merge: defaults < preset < CLI args
        # Use parse_args_standard with empty base to get only CLI-specified values.
        # This is the single source of truth for all CLI flag parsing — no duplication.
        cli_overrides = Parser.parse(args, [], parser_context())

        defaults()
        |> Keyword.merge(preset_opts)
        |> Keyword.merge(cli_overrides)
    end
  end

  @doc """
  Parse command-line arguments into a keyword list of options.

  If `--preset` is provided, the preset values are used as a base,
  with any explicit CLI arguments overriding the preset.

  ## Examples

      iex> opts = ExPhil.Training.Config.parse_args(["--epochs", "5", "--temporal"])
      iex> opts[:epochs]
      5
      iex> opts[:temporal]
      true

      iex> opts = ExPhil.Training.Config.parse_args(["--preset", "quick"])
      iex> opts[:epochs]
      1
      iex> opts[:max_files]
      5

      iex> opts = ExPhil.Training.Config.parse_args(["--preset", "quick", "--epochs", "3"])
      iex> opts[:epochs]
      3
      iex> opts[:max_files]
      5

  """
  @spec parse_args([String.t()]) :: keyword()
  def parse_args(args) when is_list(args) do
    # Check if config file is specified first
    yaml_opts =
      if Parser.has_flag_value?(args, "--config") do
        config_path = Parser.get_arg_value(args, "--config")

        case load_yaml(config_path) do
          {:ok, yaml_opts} ->
            # Merge YAML opts on top of defaults
            yaml_opts

          {:error, reason} ->
            IO.puts(:stderr, "Error loading config file: #{inspect(reason)}")
            System.halt(1)
        end
      else
        []
      end

    base_opts = Keyword.merge(defaults(), yaml_opts)

    # Check if preset is specified - if so, use apply_preset flow
    opts =
      if Parser.has_flag_value?(args, "--preset") do
        apply_preset(base_opts, args)
      else
        # No preset - standard parsing flow
        Parser.parse(args, base_opts, parser_context())
      end

    # Apply per-backbone safe defaults for values the user didn't explicitly set.
    # E.g., Jamba auto-gets lr=5e-6 and grad_clip=0.25 unless you passed --learning-rate.
    opts = apply_backbone_defaults(opts, args)

    # Apply scale-dependent adjustments based on dataset size
    preset_opts =
      case Parser.get_arg_value(args, "--preset") do
        nil -> []
        name -> preset(name)
      end

    cli_opts = Parser.parse(args, [], parser_context())
    layers = [preset_opts, yaml_opts, cli_opts]

    resume_delay =
      if opts[:resume] && File.regular?(opts[:resume]) &&
           not Enum.any?(layers, &ExPhil.Training.LabelDelay.explicit?/1) do
        case ExPhil.Training.Checkpoint.load(opts[:resume]) do
          {:ok, checkpoint} ->
            delay = ExPhil.Data.LabelConvention.reaction_delay(checkpoint.config)
            if delay < 0,
              do: raise(ArgumentError, "Resume checkpoint used leaked labels; explicitly choose a causal --label-delay")
            [label_delay: delay]
          {:error, reason} ->
            raise ArgumentError, "Cannot read resume delay: #{inspect(reason)}"
        end
      else
        []
      end

    delay_opts = ExPhil.Training.LabelDelay.merge_layers!([resume_delay | layers])
    opts |> adjust_for_scale(args) |> Keyword.merge(delay_opts)
  end

  # Adjust defaults based on dataset scale (max_files).
  # Larger datasets need different hyperparameters to avoid mode collapse.
  defp adjust_for_scale(opts, args) do
    max_files = opts[:max_files]

    if max_files && max_files > 100 do
      overrides = [
        # Entropy regularization prevents mode collapse on large datasets
        entropy_weight: 0.01
      ]

      Enum.reduce(overrides, opts, fn {key, value}, acc ->
        cli_flag = %{entropy_weight: "--entropy-weight"}[key]

        if cli_flag && ExPhil.Training.Config.Parser.has_flag_value?(args, cli_flag) do
          # User explicitly set — don't override
          acc
        else
          if acc[key] == 0.0 or acc[key] == nil do
            Keyword.put(acc, key, value)
          else
            acc
          end
        end
      end)
    else
      opts
    end
  end

  # Map from backbone_defaults keys to their CLI flag names
  # Keys listed here can be overridden by explicit CLI args
  @backbone_default_flags %{
    learning_rate: "--learning-rate",
    max_grad_norm: "--max-grad-norm",
    batch_size: "--batch-size",
    temporal: "--temporal",
    precision: "--precision",
    dropout: "--dropout",
    lr_schedule: "--lr-schedule",
    window_size: "--window-size",
    num_layers: "--num-layers",
    state_size: "--state-size",
    expand_factor: "--expand-factor",
    conv_size: "--conv-size",
    num_heads: "--num-heads",
    head_dim: "--head-dim",
    chunked_attention: "--chunked-attention"
  }

  # Apply per-backbone training defaults for values the user didn't explicitly pass via CLI.
  # This prevents NaN/OOM for sensitive architectures (Jamba, H3, TTT, Zamba) while
  # still letting explicit CLI args take priority.
  defp apply_backbone_defaults(opts, args) do
    backbone = opts[:backbone]
    overrides = backbone_defaults(backbone)

    if overrides == [] do
      opts
    else
      # Only apply defaults for keys the user didn't explicitly set
      applied =
        Enum.reduce(overrides, opts, fn {key, value}, acc ->
          cli_flag = @backbone_default_flags[key]

          cond do
            # User explicitly set via CLI — don't override
            cli_flag && Parser.has_flag_value?(args, cli_flag) -> acc
            # Preset already set a non-default value — don't override
            acc[key] != nil && acc[key] != defaults()[key] -> acc
            # Apply backbone default
            true -> Keyword.put(acc, key, value)
          end
        end)

      # Log what we changed so the user knows
      changed =
        Enum.filter(overrides, fn {key, _} ->
          cli_flag = @backbone_default_flags[key]
          !(cli_flag && Parser.has_flag_value?(args, cli_flag))
        end)

      if changed != [] do
        changes = Enum.map_join(changed, ", ", fn {k, v} -> "#{k}=#{v}" end)
        IO.puts(:stderr, "  [#{backbone}] Auto-applied safe defaults: #{changes}")
        IO.puts(:stderr, "    (override with explicit CLI flags)")
      end

      applied
    end
  end

  # Build context for argument parsing with allowlists
  defp parser_context do
    %{
      valid_backbones: valid_backbones(),
      valid_optimizers: @valid_optimizers,
      valid_lr_schedules: @valid_lr_schedules,
      valid_characters: Map.keys(@character_map),
      valid_stages: Map.keys(@stage_map),
      valid_flags: @valid_flags
    }
  end

  @doc """
  List of valid CLI flags.

  ## Examples

      iex> flags = ExPhil.Training.Config.valid_flags()
      iex> "--epochs" in flags
      true
      iex> "--batch-size" in flags
      true
      iex> "--preset" in flags
      true

  """
  @spec valid_flags() :: [String.t()]
  def valid_flags, do: @valid_flags

  @doc """
  List of valid policy types.

  ## Examples

      iex> types = ExPhil.Training.Config.valid_policy_types()
      iex> :autoregressive in types
      true
      iex> :diffusion in types
      true

  """
  @spec valid_policy_types() :: [atom()]
  def valid_policy_types, do: @valid_policy_types

  @doc """
  Validate command-line arguments for unrecognized flags.
  Delegates to `ExPhil.Training.Config.Parser.validate_args/2`.
  """
  @spec validate_args(list(String.t())) :: {:ok, list(String.t())}
  def validate_args(args) when is_list(args) do
    Parser.validate_args(args, @valid_flags)
  end

  @doc """
  Validate args and print warnings if any.
  Delegates to `ExPhil.Training.Config.Parser.validate_args!/2`.
  """
  @spec validate_args!(list(String.t())) :: :ok
  def validate_args!(args) do
    Parser.validate_args!(args, @valid_flags)
  end

  @doc """
  Parse hidden sizes string into list of integers.
  Delegates to `ExPhil.Training.Config.Parser.parse_hidden_sizes/1`.
  """
  @spec parse_hidden_sizes(String.t()) :: [integer()]
  defdelegate parse_hidden_sizes(str), to: Parser

  # =============================================================================
  # Checkpoint Naming and Path Utilities
  # =============================================================================

  @doc """
  Generate a checkpoint name with memorable naming if not already specified.

  Format: `checkpoints/{character_}{backbone}_{name}_{timestamp}.axon`
  """
  def ensure_checkpoint_name(opts) do
    if opts[:checkpoint] do
      opts
    else
      alias ExPhil.Training.Naming

      timestamp = generate_timestamp()
      backbone = if opts[:temporal], do: opts[:backbone], else: :mlp
      auto_name = Naming.generate()
      user_name = opts[:name]
      character = opts[:character]

      checkpoint_name =
        cond do
          user_name && character ->
            "checkpoints/#{character}_#{user_name}_#{timestamp}.axon"

          user_name ->
            "checkpoints/#{user_name}_#{timestamp}.axon"

          character ->
            "checkpoints/#{character}_#{backbone}_#{auto_name}_#{timestamp}.axon"

          true ->
            "checkpoints/#{backbone}_#{auto_name}_#{timestamp}.axon"
        end

      display_name = user_name || auto_name

      opts
      |> Keyword.put(:checkpoint, checkpoint_name)
      |> Keyword.put(:name, display_name)
    end
  end

  @doc """
  Generate a timestamp string for checkpoint naming.
  Format: YYYYMMDD_HHMMSS in UTC
  """
  def generate_timestamp do
    DateTime.utc_now() |> Calendar.strftime("%Y%m%d_%H%M%S")
  end

  @doc """
  Generate a timestamp string using a specific DateTime (for testing).
  """
  def generate_timestamp(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y%m%d_%H%M%S")
  end

  @doc """
  Derive the policy path from a checkpoint path.
  """
  def derive_policy_path(nil), do: nil

  def derive_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_policy.bin")
  end

  @doc """
  Derive the config JSON path from a checkpoint path.
  """
  def derive_config_path(nil), do: nil

  def derive_config_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_config.json")
  end

  @doc """
  Derive the best checkpoint path from a checkpoint path.
  """
  def derive_best_checkpoint_path(nil), do: nil

  def derive_best_checkpoint_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best.axon")
  end

  @doc """
  Derive the best policy path from a checkpoint path.
  """
  def derive_best_policy_path(nil), do: nil

  def derive_best_policy_path(checkpoint_path) do
    String.replace(checkpoint_path, ".axon", "_best_policy.bin")
  end

  @doc """
  Compute a SHA256 hash of a list of file paths for replay manifest.
  """
  @spec compute_manifest_hash([String.t()]) :: String.t() | nil
  def compute_manifest_hash([]), do: nil

  def compute_manifest_hash(paths) when is_list(paths) do
    paths
    |> Enum.sort()
    |> Enum.join("\n")
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> then(&"sha256:#{&1}")
  end

  @doc """
  Build the training config map that gets saved as JSON alongside the model.
  """
  def build_config_json(opts, results \\ %{}) do
    %{
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      replays_dir: opts[:replays],
      max_files: opts[:max_files],
      player_port: opts[:player_port],
      characters: format_atom_list(opts[:characters]),
      stages: format_atom_list(opts[:stages]),
      replay_count: results[:replay_count],
      replay_files: results[:replay_files],
      replay_manifest_hash: results[:replay_manifest_hash],
      character_distribution: results[:character_distribution],
      temporal: opts[:temporal],
      backbone: if(opts[:temporal], do: to_string(opts[:backbone]), else: "mlp"),
      policy_type: to_string(opts[:policy_type] || :autoregressive),
      head: to_string(opts[:head] || :independent),
      action_horizon: opts[:action_horizon],
      num_inference_steps: opts[:num_inference_steps],
      kl_weight: opts[:kl_weight],
      hidden_sizes: opts[:hidden_sizes],
      embed_size: results[:embed_size],
      layer_norm: opts[:layer_norm],
      residual: opts[:residual],
      kmeans_centers: opts[:kmeans_centers],
      stage_mode: get_embedding_mode(opts, :stage_mode),
      action_mode: get_embedding_mode(opts, :action_mode),
      character_mode: get_embedding_mode(opts, :character_mode),
      nana_mode: get_embedding_mode(opts, :nana_mode),
      # INVARIANTS.md item 4: which channels the training source provided
      # and whether the projectile block therefore exists in this checkpoint's
      # embedding. The agent builds its live embed config from these.
      provided_channels: ExPhil.Data.Peppi.provides(),
      with_projectiles:
        ExPhil.Embeddings.resolve_with_projectiles(opts, ExPhil.Data.Peppi.provides()),
      jumps_normalized: Keyword.get(opts, :jumps_normalized, defaults()[:jumps_normalized]),
      window_size: opts[:window_size],
      stride: opts[:stride],
      num_layers: opts[:num_layers],
      truncate_bptt: opts[:truncate_bptt],
      state_size: opts[:state_size],
      expand_factor: opts[:expand_factor],
      conv_size: opts[:conv_size],
      attention_every: opts[:attention_every],
      num_heads: opts[:num_heads],
      head_dim: opts[:head_dim],
      epochs: opts[:epochs],
      batch_size: opts[:batch_size],
      precision: to_string(opts[:precision]),
      frame_delay: opts[:frame_delay],
      label_delay: ExPhil.Training.LabelDelay.resolve!(opts)[:label_delay],
      learning_rate: opts[:lr],
      lr_schedule: opts[:lr_schedule] && to_string(opts[:lr_schedule]),
      warmup_steps: opts[:warmup_steps],
      optimizer: opts[:optimizer] && to_string(opts[:optimizer]),
      max_grad_norm: opts[:max_grad_norm],
      accumulation_steps: opts[:accumulation_steps],
      label_smoothing: opts[:label_smoothing],
      dropout: opts[:dropout],
      focal_loss: opts[:focal_loss],
      use_prev_action: opts[:use_prev_action],
      prev_action_dropout: opts[:prev_action_dropout],
      scheduled_sampling: opts[:scheduled_sampling],
      ss_ramp: opts[:ss_ramp],
      action_delay: opts[:action_delay],
      focal_gamma: opts[:focal_gamma],
      button_weight: opts[:button_weight],
      button_pos_weight:
        case opts[:button_pos_weight] do
          %Nx.Tensor{} = t -> Nx.to_flat_list(t)
          other -> other
        end,
      stick_edge_weight: opts[:stick_edge_weight],
      entropy_weight: opts[:entropy_weight] || 0.0,
      ema: opts[:ema],
      ema_decay: opts[:ema_decay],
      train_character: opts[:train_character] && to_string(opts[:train_character]),
      augment: opts[:augment],
      val_split: opts[:val_split],
      seed: opts[:seed],
      early_stopping: opts[:early_stopping],
      patience: opts[:patience],
      min_delta: opts[:min_delta],
      training_frames: results[:training_frames],
      validation_frames: results[:validation_frames],
      total_time_seconds: results[:total_time_seconds],
      final_training_loss: results[:final_training_loss],
      epochs_completed: results[:epochs_completed],
      stopped_early: results[:stopped_early],
      checkpoint_path: opts[:checkpoint],
      policy_path: derive_policy_path(opts[:checkpoint]),
      # INVARIANTS.md item 1: which label pairing this checkpoint was
      # trained under. Unstamped checkpoints are legacy (:producing);
      # readers go through ExPhil.Data.LabelConvention, never the raw
      # delay numbers.
      label_convention: ExPhil.Data.LabelConvention.current()
    }
    # INVARIANTS.md item 11: every checkpoint carries its comparability key
    # (label_delay / embed canary / loss recipe / train_delays) so tools can
    # refuse to rank losses that don't mean the same thing.
    |> then(fn map ->
      Map.put(map, :comparability_key, ExPhil.Training.Comparability.key(map))
    end)
  end

  defp format_atom_list(nil), do: nil
  defp format_atom_list([]), do: nil

  defp format_atom_list(atoms) when is_list(atoms) do
    Enum.map(atoms, &to_string/1)
  end

  defp get_embedding_mode(opts, key) do
    value = Keyword.get(opts, key, defaults()[key])

    case value do
      nil -> nil
      atom when is_atom(atom) -> to_string(atom)
      other -> other
    end
  end

  # =============================================================================
  # Smart Flag Inference
  # =============================================================================

  # Delegated to ExPhil.Training.Config.Inference
  # See that module for implementation details

  @doc """
  Apply smart defaults based on flag combinations.

  Delegates to `ExPhil.Training.Config.Inference.infer_smart_defaults/1`.
  See that module for details on what inferences are applied.
  """
  @spec infer_smart_defaults(keyword()) :: {keyword(), list(String.t())}
  defdelegate infer_smart_defaults(opts), to: Inference

  # =============================================================================
  # Checkpoint Safety Functions
  # =============================================================================

  # Delegated to ExPhil.Training.Config.Checkpoint
  # See that module for implementation details

  @doc """
  Check if a checkpoint path would overwrite an existing file.
  Delegates to `ExPhil.Training.Config.Checkpoint.check_checkpoint_path/2`.
  """
  defdelegate check_checkpoint_path(path, opts \\ []), to: Checkpoint

  @doc """
  Format file info for display in collision warnings.
  Delegates to `ExPhil.Training.Config.Checkpoint.format_file_info/1`.
  """
  defdelegate format_file_info(info), to: Checkpoint

  @doc """
  Backup an existing checkpoint before overwriting.
  Delegates to `ExPhil.Training.Config.Checkpoint.backup_checkpoint/2`.
  """
  defdelegate backup_checkpoint(path, opts \\ []), to: Checkpoint

  # =============================================================================
  # Reproducibility Functions
  # =============================================================================

  @doc """
  Initialize random seed for reproducibility.

  If seed is provided, uses it directly. Otherwise generates a seed from system entropy.
  Returns the seed used (for logging).

  Sets seeds for:
  - Erlang's :rand module
  - Nx global default seed (for parameter initialization, dropout)
  """
  @spec init_seed(integer() | nil) :: integer()
  def init_seed(nil) do
    # Generate seed from system entropy
    seed = :rand.uniform(2_147_483_647)
    init_seed(seed)
  end

  def init_seed(seed) when is_integer(seed) do
    # Seed Erlang's random module
    :rand.seed(:exsss, {seed, seed, seed})

    # Seed Nx's global key (affects Nx.Random operations)
    # Note: Nx uses a PRNG key system, this sets the default
    Nx.default_backend(EXLA.Backend)
    # Merge into default_defn_options rather than replacing them: a plain
    # put_env with [seed: ...] drops the compiler, silently sending every
    # bare Nx.Defn.jit/value_and_grad in the app to Nx.Defn.Evaluator
    # (pure-Elixir, no XLA fusion).
    defn_opts =
      Application.get_env(:nx, :default_defn_options, [])
      |> Keyword.put_new(:compiler, EXLA)
      |> Keyword.put(:seed, seed)

    Application.put_env(:nx, :default_defn_options, defn_opts)

    seed
  end

  @doc """
  Get verbosity level description.
  """
  @spec verbosity_name(integer()) :: String.t()
  def verbosity_name(0), do: "quiet"
  def verbosity_name(1), do: "normal"
  def verbosity_name(2), do: "verbose"
  def verbosity_name(_), do: "unknown"
end
