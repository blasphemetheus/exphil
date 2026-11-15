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
#   --lead-extra N    extra frames to aim ahead; reaction delay k = 1 + N on the
#                     arena (see "Timing" in ExPhil.Stomped.Match). With --policy
#                     the default is the checkpoint's trained delay, pinned for
#                     the whole match. `auto` times the arena's ping and aims
#                     just far enough ahead for the measured round trip; the
#                     lead can grow mid-match, which changes the policy's task,
#                     so it is the default only for the walker.
#   --timing-log PATH append one JSON line per match: lead, reaction delay,
#                     round-trip and compute distributions, landed/late pads
#   --pad-trace PATH  append one line per scored frame (frame, outcome, arrival
#                     time) for reading the time structure of late pads
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
      timing_log: :string,
      pad_trace: :string,
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

# The lead decides which reaction delay the policy plays at (see the "Timing"
# section of ExPhil.Stomped.Match): k = inputDelayFrames + lead_extra.
#
#   * (no flag) with --policy: pinned to the checkpoint's trained delay, so
#     the live task is the training task for the whole match
#   * (no flag) without --policy, or `auto`: the server times the arena's
#     `ping` and aims just far enough ahead for the measured round trip. The
#     lead can grow mid-match, which changes a policy's task: not for
#     measuring a policy.
#   * a number pins it.
lead_extra_flag =
  case Keyword.get(opts, :lead_extra) do
    nil ->
      nil

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

# The arena shows the game one frame behind the engine. A match whose header
# says otherwise is counted under the policy bot's delay_mismatch.
arena_input_delay = 1

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
{bot, hold_ahead, lead_extra} =
  case opts[:policy] do
    nil ->
      {{ExPhil.Stomped.Bots.Walker, character: character, costume: costume},
       Keyword.get(opts, :hold_ahead, 0), lead_extra_flag || :auto}

    policy_path ->
      # The checkpoint's trained delay is the default lead; a pinned number
      # that differs is allowed (it is a deliberate off-delay experiment) and
      # the bot counts every live frame of it under delay_mismatch.
      # `trained` is the lead_extra that plays the checkpoint's trained delay,
      # or nil when the arena cannot play that delay at all (a checkpoint
      # trained at reaction delay 0: the arena's first open pad is one frame
      # later). Such a checkpoint can still be served at a pinned lead to
      # measure the host; its play is then one or more frames slower than it
      # was trained for, and every live frame counts under delay_mismatch.
      {trained, trained_k} =
        case ExPhil.Training.Checkpoint.load_policy(policy_path) do
          {:ok, %{config: config}} ->
            k = max(ExPhil.Data.LabelConvention.reaction_delay(config), 0)

            case ExPhil.Eval.HarnessRung.deploy_knob(:stomped, config) do
              {:ok, knob} ->
                {knob, k}

              {:error, {:unreachable, min}} ->
                if lead_extra_flag == nil do
                  Output.error(
                    "The checkpoint was trained at reaction delay #{k}, which the arena " <>
                      "cannot play (its floor is latency #{min}, reaction delay #{min - 1}). " <>
                      "Train at --label-delay >= #{min - 1}, or pass --lead-extra " <>
                      "#{min - 1 - arena_input_delay} to measure the host with this policy " <>
                      "playing one frame slower than trained."
                  )

                  System.halt(1)
                end

                {nil, k}
            end

          {:error, reason} ->
            Output.error("Could not read #{policy_path}: #{inspect(reason)}")
            System.halt(1)
        end

      lead_extra = lead_extra_flag || trained

      reaction_delay =
        case lead_extra do
          :auto ->
            Output.warning(
              "--lead-extra auto with a policy: the lead can grow during a match, which " <>
                "changes the delay the policy plays at. Pin --lead-extra " <>
                "#{trained || 0} to measure the policy."
            )

            arena_input_delay + (trained || 0)

          n ->
            if n != trained do
              Output.warning(
                "--lead-extra #{n} plays reaction delay #{arena_input_delay + n}; the " <>
                  "checkpoint was trained at #{trained_k}" <>
                  if(trained, do: " (--lead-extra #{trained})", else: ", unreachable here") <>
                  ". Round trip, compute and late pads are still a measurement of the host; " <>
                  "the policy's play is not."
              )
            end

            arena_input_delay + n
        end

      Output.puts("⏳ JIT compiling model (warm-up)... this may take 2-5 minutes")

      case ExPhil.Stomped.Bots.Policy.start(
             policy: policy_path,
             reaction_delay: reaction_delay,
             stateful_step: not Keyword.get(opts, :windowed, false),
             deterministic: Keyword.get(opts, :deterministic, false),
             temperature: Keyword.get(opts, :temperature, 1.0)
           ) do
        {:ok, policy} ->
          Output.success(
            "Agent warm in #{policy.warmup_ms} ms; plays reaction delay " <>
              "#{policy.reaction_delay} (trained #{trained_k})"
          )

          {{ExPhil.Stomped.Bots.Policy,
            seat: policy.seat,
            reaction_delay: policy.reaction_delay,
            trace: Keyword.get(opts, :pad_trace),
            character: character,
            costume: costume}, Keyword.get(opts, :hold_ahead, 1), lead_extra}

        {:error, reason} ->
          Output.error("Could not start the policy agent: #{inspect(reason)}")
          System.halt(1)
      end
  end

timing_log = Keyword.get(opts, :timing_log)

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
     do: "auto (from the measured round trip; grows, never shrinks)",
     else:
       "1 + inputDelayFrames + #{lead_extra} (reaction delay #{arena_input_delay + lead_extra})"
   )},
  {"Hold ahead", hold_ahead},
  {"Timing log", timing_log || "(none — pass --timing-log PATH to keep per-match summaries)"},
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
    timing_log: timing_log,
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
