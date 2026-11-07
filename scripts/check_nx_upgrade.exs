# AI-generated. Method and authorship marker: docs/research/NX_CURRENT_VALIDATION.md.
# Bare-VM check: retain ExPhil's compiled native dependencies, load the candidate
# Edifice/Nx/EXLA build ahead of them, and execute current regression sources.
{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [edifice_build: :string, client: :string, optimizer_only: :boolean, out: :string]
  )

build = opts |> Keyword.fetch!(:edifice_build) |> Path.expand()
Code.prepend_paths(Path.wildcard(Path.join(build, "lib/*/ebin")))
{:ok, _} = Application.ensure_all_started(:exphil)

client =
  case opts[:client] || "host" do
    "host" -> :host
    "cuda" -> :cuda
  end

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.25]
)

Application.put_env(:exla, :default_client, client)
Nx.global_default_backend({EXLA.Backend, client: client})
Nx.Defn.global_default_options(compiler: EXLA, client: client, precision: :highest)
Logger.configure(level: :warning)

files =
  if opts[:optimizer_only] do
    ExUnit.start(autorun: false, exclude: [:test], include: [:nx_upgrade])
    ["test/exphil/training/imitation_test.exs"]
  else
    ExUnit.start(autorun: false)

    [
      "test/exphil/networks/policy/sampling_rng_parity_test.exs",
      "test/exphil/agents/agent_stateful_mamba_test.exs",
      "test/exphil/agents/agent_stateful_mamba2_full_test.exs"
    ]
  end

for file <- files, do: Code.require_file(Path.expand("../" <> file, __DIR__))
result = ExUnit.run()

beams =
  Map.new(
    [
      Nx,
      Nx.Random,
      EXLA,
      EXLA.Defn,
      Edifice.SSM.Mamba2,
      ExPhil.Training.Imitation,
      ExPhil.Training.Imitation.TrainLoop
    ],
    fn module ->
      path = :code.which(module) |> List.to_string()

      {inspect(module),
       %{path: path, sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower)}}
    end
  )

if opts[:out],
  do:
    File.write!(
      opts[:out],
      Jason.encode!(
        %{
          client: client,
          result: result,
          candidate_build: build,
          beams: beams,
          otp: System.otp_release(),
          elixir: System.version()
        },
        pretty: true
      )
    )

if result.failures > 0, do: System.halt(1)
