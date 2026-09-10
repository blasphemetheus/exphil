defmodule ExPhil.Training.Imitation.LossConfig do
  @moduledoc """
  INVARIANTS.md item 8: loss configuration as ONE typed value.

  Before 2026-09-09 four loss builders (`build_autoregressive_loss_and_grad_fn`,
  `build_bptt_loss_and_grad_fn`, `build_bptt_eval_loss_fn`,
  `build_autoregressive_eval_loss_fn`) each re-extracted the same knobs
  from the config with their own inline defaults — and they had drifted:
  `head` defaulted to `:independent` on the windowed path and
  `:autoregressive` on the bptt path, precision to `:bf16` vs `:f32`,
  `button_weight` 1.0 vs `Config.defaults` 2.0, `focal_gamma` 2.0 vs 3.0.
  Three more builders re-derived precision alone.

  Now every builder calls `from_config/1` and reads fields off the struct;
  the sampler-side keyword list is produced by `to_loss_opts/1` at the
  boundary to `Policy.Loss.imitation_loss/3`. Fallback values come from
  ONE place (`ExPhil.Training.Config.defaults/0`), and only when a key is
  ABSENT — a present `nil`/`false` is honored, so callers that build their
  own config (drills, tests) keep their exact semantics.

  Type-level fact encoded here: the button loss has NO smoothing field.
  `label_smoothing` belongs to the categorical heads only; the button
  head's BCE is either plain or positive-weighted (`button: %{pos_weight,
  focal}`). The smoothing x pos_weight optimum-shift pathology (July 2026)
  is unrepresentable in this struct.
  """

  alias ExPhil.Training.Config

  @enforce_keys [:precision, :head]
  defstruct label_smoothing: 0.0,
            button: %{pos_weight: nil, focal: nil, weight: 1.0},
            stick_edge_weight: nil,
            entropy_weight: 0.0,
            head_normalize: false,
            precision: :f32,
            head: :independent,
            temporal: false

  @type button :: %{
          pos_weight: Nx.Tensor.t() | nil,
          focal: %{gamma: number()} | nil,
          weight: number()
        }

  @type t :: %__MODULE__{
          label_smoothing: number(),
          button: button(),
          stick_edge_weight: number() | nil,
          entropy_weight: number(),
          head_normalize: boolean(),
          precision: :f32 | :bf16 | :f16,
          head: :independent | :autoregressive,
          temporal: boolean()
        }

  @doc "Build from a training config (map or keyword). Absent keys fall back to Config.defaults/0."
  @spec from_config(map() | keyword()) :: t()
  def from_config(config) do
    get = fn key -> fetch(config, key) end

    focal =
      if get.(:focal_loss) in [true, "true"],
        do: %{gamma: get.(:focal_gamma) || 2.0},
        else: nil

    %__MODULE__{
      label_smoothing: get.(:label_smoothing) || 0.0,
      button: %{
        pos_weight: normalize_pos_weight(get.(:button_pos_weight)),
        focal: focal,
        weight: get.(:button_weight) || 1.0
      },
      stick_edge_weight: get.(:stick_edge_weight),
      entropy_weight: get.(:entropy_weight) || 0.0,
      head_normalize: get.(:head_normalize) || false,
      precision: get.(:precision) || Config.defaults()[:precision],
      head: get.(:head) || Config.defaults()[:head],
      temporal: get.(:temporal) || false
    }
  end

  @doc "The keyword list `Policy.Loss.imitation_loss/3` consumes (boundary conversion)."
  @spec to_loss_opts(t()) :: keyword()
  def to_loss_opts(%__MODULE__{} = lc) do
    [
      label_smoothing: lc.label_smoothing,
      focal_loss: lc.button.focal != nil,
      focal_gamma: (lc.button.focal && lc.button.focal.gamma) || 2.0,
      button_weight: lc.button.weight,
      button_pos_weight: lc.button.pos_weight,
      stick_edge_weight: lc.stick_edge_weight,
      entropy_weight: lc.entropy_weight,
      head_normalize: lc.head_normalize
    ]
  end

  # Present key (even nil/false) wins; absent key -> Config default.
  defp fetch(config, key) when is_map(config) do
    case Map.fetch(config, key) do
      {:ok, v} -> v
      :error -> Config.defaults()[key]
    end
  end

  defp fetch(config, key) when is_list(config) do
    case Keyword.fetch(config, key) do
      {:ok, v} -> v
      :error -> Config.defaults()[key]
    end
  end

  # :auto is resolved by Pipeline from data stats (nil here = unweighted);
  # an explicit CLI list arrives as a plain Elixir list — tensorize once on
  # BinaryBackend (captured loss constants must not be device tensors).
  @doc false
  def normalize_pos_weight(:auto), do: nil
  def normalize_pos_weight("auto"), do: nil

  def normalize_pos_weight(list) when is_list(list),
    do: Nx.tensor(list, type: :f32, backend: Nx.BinaryBackend)

  def normalize_pos_weight(other), do: other
end
