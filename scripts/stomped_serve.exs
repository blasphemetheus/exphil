#!/usr/bin/env elixir
# Serve a bot for the stomped.dev arena (https://stomped.dev).
#
# A stomped bot is a web server: the arena opens a WebSocket to the URL you
# register, streams the game as raw .slp events, and reads 12-byte pads back.
# This script starts that server (ExPhil.Stomped.Server) and stays up.
#
# Usage (from the repo root, inside `devenv shell`):
#
#   mix run scripts/stomped_serve.exs --path-secret "$(uuidgen)"
#   curl localhost:3000/<secret>/ping          # -> pong
#   cloudflared tunnel --url http://localhost:3000
#   # register https://<tunnel-host>/<secret> at https://stomped.dev/bots,
#   # then restart with --url set to that exact URL to verify handshakes.
#
# Flags:
#   --path-secret S   secret path of the registered URL (default: random, printed)
#   --port N          TCP port (default 3000)
#   --bind ADDR       loopback (default; right for a tunnel) or any
#   --url URL         the registered URL, byte for byte; turns on signed-handshake
#                     verification (fetches the platform key at startup)
#   --lead-extra N    extra frames to aim ahead when the round trip is over the
#                     11 ms budget. Default `auto`: the server times the arena's
#                     ping each match and aims just far enough ahead. A number
#                     pins it (reaction delay k = 1 + N). The lead and the
#                     measured round trip are logged when a match closes.
#   --policy PATH     exported policy to play (ExPhil.Stomped.Bots.Policy). Without it
#                     the quickstart walker plays, which only proves the deploy.
#   --hold-ahead N    repeat each pad N frames ahead as cover for a late decision
#                     (default 1 with --policy, 0 for the walker)
#   --windowed        windowed inference instead of stateful step
#   --deterministic   argmax instead of sampling
#   --temperature T   sampling temperature (default 1.0)
#
# GPU note: on the CUDA client EXLA reserves EXPHIL_GPU_MEMORY_FRACTION of VRAM up
# front (default 0.75). A policy bot needs very little: export a small fraction
# (0.1) before serving, and remember the server is a live exphil beam (no `mix`
# next to it, as with a training run).
#   --character C     character name or slp external id (default fox)
#   --costume N       costume index (default 0)
#
# WARNING: with a tunnel this opens a port on this machine to the internet.
# The server answers only /<secret>/ping and the socket at /<secret>.

alias ExPhil.Stomped.{Bot, Server, Signature}
alias ExPhil.Training.Output

{opts, rest, invalid} =
  OptionParser.parse(System.argv(),
    strict: [
      path_secret: :string,
      port: :integer,
      bind: :string,
      url: :string,
      lead_extra: :string,
      policy: :string,
      hold_ahead: :integer,
      windowed: :boolean,
      deterministic: :boolean,
      temperature: :float,
      character: :string,
      costume: :integer
    ]
  )

# Unknown flags stop the script: a silently dropped flag is a wrong deploy.
if invalid != [] or rest != [] do
  Output.error("Unknown arguments: #{inspect(invalid ++ rest)}")
  System.halt(1)
end

Output.banner("ExPhil — stomped.dev bot server")

port = Keyword.get(opts, :port, 3000)
# auto (default): the server times the arena's `ping` and aims just far enough
# ahead for the measured round trip. A number pins it.
lead_extra =
  case Keyword.get(opts, :lead_extra, "auto") do
    "auto" ->
      :auto

    value ->
      case Integer.parse(value) do
        {n, ""} when n >= 0 ->
          n

        _ ->
          Output.error("--lead-extra must be auto or a whole number >= 0")
          System.halt(1)
      end
  end

ip =
  case Keyword.get(opts, :bind, "loopback") do
    "loopback" ->
      :loopback

    "any" ->
      :any

    other ->
      Output.error("--bind must be loopback or any, got #{inspect(other)}")
      System.halt(1)
  end

character =
  case Bot.character_id(Keyword.get(opts, :character, "fox")) do
    {:ok, id} ->
      id

    :error ->
      Output.error(
        "Unknown --character. Names: #{Bot.characters() |> Map.keys() |> Enum.sort() |> Enum.join(", ")}"
      )

      System.halt(1)
  end

costume = Keyword.get(opts, :costume, 0)

# The agent is warmed BEFORE the server listens: JIT compilation takes far
# longer than the arena's step deadlines, and a registered bot can be seated
# within minutes.
{bot, hold_ahead} =
  case opts[:policy] do
    nil ->
      {{ExPhil.Stomped.Bots.Walker, character: character, costume: costume},
       Keyword.get(opts, :hold_ahead, 0)}

    policy_path ->
      Output.puts("⏳ JIT compiling model (warm-up)... this may take 2-5 minutes")

      case ExPhil.Stomped.Bots.Policy.start(
             policy: policy_path,
             lead_extra: if(lead_extra == :auto, do: 0, else: lead_extra),
             stateful_step: not Keyword.get(opts, :windowed, false),
             deterministic: Keyword.get(opts, :deterministic, false),
             temperature: Keyword.get(opts, :temperature, 1.0)
           ) do
        {:ok, policy} ->
          Output.success(
            "Agent warm in #{policy.warmup_ms} ms; reaction delay " <>
              if(lead_extra == :auto,
                do: "set per match from the round trip (at least 1)",
                else: "#{policy.reaction_delay}"
              )
          )

          {{ExPhil.Stomped.Bots.Policy,
            seat: policy.seat, character: character, costume: costume},
           Keyword.get(opts, :hold_ahead, 1)}

        {:error, reason} ->
          Output.error("Could not start the policy agent: #{inspect(reason)}")
          System.halt(1)
      end
  end

url = Keyword.get(opts, :url)

secret =
  cond do
    opts[:path_secret] -> String.trim(opts[:path_secret], "/")
    url -> url |> URI.parse() |> Map.get(:path) |> to_string() |> String.trim("/")
    true -> Base.url_encode64(:crypto.strong_rand_bytes(18), padding: false)
  end

if secret == "" do
  Output.error("The registered URL needs an unguessable path: it is the bot's password.")
  System.halt(1)
end

path = "/" <> secret

public_key =
  if url do
    case Signature.fetch_public_key() do
      {:ok, key} ->
        key

      {:error, reason} ->
        Output.error(
          "Could not fetch the platform key from #{Signature.key_url()}: #{inspect(reason)}"
        )

        System.halt(1)
    end
  end

Output.config([
  {"Bot", inspect(elem(bot, 0))},
  {"Character id", character},
  {"Port", port},
  {"Bind", ip},
  {"Path", path},
  {"Policy", opts[:policy] || "(none — quickstart walker)"},
  {"Lead",
   if(lead_extra == :auto,
     do: "auto (from the measured round trip)",
     else: "1 + inputDelayFrames + #{lead_extra}"
   )},
  {"Hold ahead", hold_ahead},
  {"EXLA client", inspect(Application.get_env(:exla, :default_client))},
  {"Handshake verification",
   if(url, do: "on (#{url})", else: "OFF — the URL alone is the credential")}
])

{:ok, _server} =
  Server.start_link(
    port: port,
    ip: ip,
    path: path,
    bot: bot,
    lead_extra: lead_extra,
    hold_ahead: hold_ahead,
    url: url,
    public_key: public_key
  )

Output.success("Serving. Check: curl localhost:#{port}#{path}/ping")

if url == nil do
  Output.warning(
    "Handshake verification is off. Pass --url <registered URL> once the bot is registered."
  )
end

Process.sleep(:infinity)
