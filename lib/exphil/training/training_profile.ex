defmodule ExPhil.Training.TrainingProfile do
  @moduledoc """
  Edifice training profiles for ExPhil's optimizers.

  An `Edifice.Training.Profile` is the architecture's own statement of which
  parameters must not receive weight decay (biases, norm gains, Mamba-2's
  `A_log`, `dt_bias` and `D`, ...), plus any learning-rate multipliers it asks
  for. Before this module ExPhil built `Polaris.Optimizers.adamw/1` directly,
  which decays every float parameter, and nothing recorded which parameters
  the decay reached.

  `for_config/1` resolves the profile for a trainer config (the backbone's
  Edifice architecture when the policy is temporal, the generic rank-rule
  profile otherwise), `adamw/3` builds the decay-aware optimizer, and
  `record_groups/3` writes the chosen parameter groups into the trainer config
  under `:optimizer_groups` so checkpoints and logs carry what the optimizer
  actually did.

  ## Backbone to architecture

  ExPhil backbone names mostly equal Edifice registry names. The ones that do
  not are mapped explicitly below; spec-built backbones resolve through their
  `build:` recipe module. Anything unresolvable (MLP, sliding-window attention
  built in ExPhil) gets `Edifice.Training.Profile.generic/0`: rank <= 1
  parameters are excluded from decay, nothing else is claimed.
  """

  require Logger

  alias Edifice.Training.Profile
  alias ExPhil.Training.Config

  # Backbones ExPhil builds itself even though Edifice registers the same
  # name; their parameters are not an Edifice architecture's.
  @exphil_built [:mlp, :sliding_window]

  # ExPhil backbone atoms whose Edifice registry name differs.
  @arch_aliases %{
    mamba_2: :mamba_ssd,
    mamba_nif: :mamba,
    mamba_cumsum: :mamba_cumsum,
    mamba_hillis_steele: :mamba_hillis_steele
  }

  # Options the profile callbacks may read; everything else in a trainer
  # config is ExPhil's business.
  @build_opt_keys [
    :hidden_size,
    :num_layers,
    :state_size,
    :expand_factor,
    :conv_size,
    :window_size,
    :head_dim,
    :num_heads,
    :dropout,
    :recurrent_state,
    :rnn_layout
  ]

  @doc """
  The Edifice architecture (registry name or module) behind an ExPhil backbone,
  or `nil` when the backbone is built inside ExPhil.
  """
  @spec edifice_arch(atom()) :: atom() | module() | nil
  def edifice_arch(backbone) when is_atom(backbone) do
    arch = Map.get(@arch_aliases, backbone, backbone)

    cond do
      backbone in @exphil_built ->
        nil

      registered?(arch) ->
        arch

      recipe_module(backbone) != nil ->
        recipe_module(backbone)

      true ->
        nil
    end
  end

  @doc """
  The profile for a trainer config: the backbone's architecture profile when
  `config.temporal` is true and the backbone resolves to an Edifice
  architecture, otherwise the generic profile.
  """
  @spec for_config(map()) :: Profile.t()
  def for_config(config) when is_map(config) do
    backbone = Map.get(config, :backbone)

    with true <- Map.get(config, :temporal, false) == true,
         arch when arch != nil <- edifice_arch(backbone) do
      Profile.for(arch, build_opts(config))
    else
      _ -> Profile.generic()
    end
  end

  @doc """
  AdamW whose decay honours the profile's groups. `learning_rate` may be a
  schedule function; `opts` are `Polaris.Optimizers.adamw/1` options
  (`:b1`, `:b2`, `:eps`, `:decay`).
  """
  @spec adamw(Profile.t(), (Nx.Tensor.t() -> Nx.Tensor.t()) | number(), keyword()) ::
          {function(), function()}
  def adamw(%Profile{} = profile, learning_rate, opts \\ []) do
    Profile.adamw(profile, Keyword.put(opts, :learning_rate, learning_rate))
  end

  @doc """
  The parameter groups the optimizer will use, as plain data for a config or
  checkpoint: counts per group, the profile's source, and the paths of the
  `:no_decay` and `:skipped` leaves (the decayed set is the rest).
  """
  @spec groups(Profile.t(), map() | struct(), keyword()) :: map()
  def groups(%Profile{} = profile, params, opts \\ []) do
    g = Profile.parameter_groups(profile, params)
    decay = Keyword.get(opts, :decay, 0.0)

    %{
      profile: profile_name(profile),
      source: inspect(profile.source),
      weight_decay: decay,
      decay_count: length(g.decay),
      no_decay_count: length(g.no_decay),
      skipped_count: length(g.skipped),
      no_decay: g.no_decay,
      skipped: g.skipped,
      rules: Enum.map(profile.no_weight_decay, &inspect/1)
    }
  end

  @doc """
  Put the optimizer's parameter groups into `config` under `:optimizer_groups`
  and log a one-line summary. Returns the config.
  """
  @spec record_groups(map(), Profile.t(), map() | struct(), keyword()) :: map()
  def record_groups(config, %Profile{} = profile, params, opts \\ []) do
    g = groups(profile, params, opts)

    Logger.info(
      "[Optimizer] profile #{g.profile} (#{g.source}): " <>
        "#{g.decay_count} decayed, #{g.no_decay_count} without decay, " <>
        "#{g.skipped_count} non-float skipped; weight_decay=#{g.weight_decay}"
    )

    Map.put(config, :optimizer_groups, g)
  end

  # ---------------------------------------------------------------------------

  defp registered?(arch) when is_atom(arch), do: arch in Edifice.list_architectures()

  defp recipe_module(backbone) do
    case Config.backbone_specs() |> Keyword.get(backbone, []) |> Keyword.get(:build) do
      {module, _embed_key, _params} when is_atom(module) ->
        if edifice_module?(module), do: module, else: nil

      _ ->
        nil
    end
  end

  defp edifice_module?(module), do: String.starts_with?(Atom.to_string(module), "Elixir.Edifice.")

  defp build_opts(config) do
    for key <- @build_opt_keys, {:ok, value} <- [Map.fetch(config, key)], value != nil do
      {key, value}
    end
  end

  defp profile_name(%Profile{arch: arch}) when arch != nil, do: inspect(arch)
  defp profile_name(%Profile{module: module}) when module != nil, do: inspect(module)
  defp profile_name(_), do: "generic"
end
