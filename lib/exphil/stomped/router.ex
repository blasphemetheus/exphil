defmodule ExPhil.Stomped.Router do
  @moduledoc """
  The HTTP surface of a stomped.dev bot: `GET <url>/ping` answers `pong` and
  `<url>` itself accepts the match WebSocket.

  The registered URL is the bot's credential, so everything outside the
  configured secret path is a 404. When a registered URL and the platform key
  are configured, the signed handshake is verified too and a failure is
  refused with 401.

  Options (a map, built by `ExPhil.Stomped.Server`):

    * `:path` — the secret path, e.g. `"/7f3a9c2e"` (required; `""` serves the root)
    * `:match_opts` — options for `ExPhil.Stomped.Match.new/1`
    * `:url` / `:public_key` — both set turns handshake verification on
  """

  @behaviour Plug

  import Plug.Conn

  require Logger

  alias ExPhil.Stomped.{Signature, Socket}

  # A `replay` answer is the whole game so far in one message.
  @max_frame_size 64 * 1024 * 1024
  # Idle limit; the engine's boot between `ready` and the first frame takes seconds.
  @idle_timeout_ms 120_000

  @impl true
  def init(%{path: path, match_opts: _} = opts) when is_binary(path) do
    Map.update!(opts, :path, &String.trim_trailing(&1, "/"))
  end

  @impl true
  def call(conn, %{path: base} = opts) do
    path = String.trim_trailing(conn.request_path, "/")

    cond do
      conn.method == "GET" and path == base <> "/ping" ->
        conn |> put_resp_content_type("text/plain") |> send_resp(200, "pong")

      path != base ->
        send_resp(conn, 404, "not found")

      not websocket_upgrade?(conn) ->
        send_resp(conn, 400, "websocket only")

      true ->
        upgrade(conn, opts)
    end
  end

  defp upgrade(conn, opts) do
    headers = %{
      match: header(conn, "x-stomped-match"),
      timestamp: header(conn, "x-stomped-timestamp"),
      signature: header(conn, "x-stomped-signature")
    }

    case verify(headers, opts) do
      :ok ->
        match_opts = Keyword.put(opts.match_opts, :match_id, headers.match)

        conn
        |> WebSockAdapter.upgrade(Socket, match_opts,
          timeout: @idle_timeout_ms,
          max_frame_size: @max_frame_size
        )
        |> halt()

      {:error, reason} ->
        Logger.warning("[stomped] refused handshake: #{reason}")
        send_resp(conn, 401, "handshake refused")
    end
  end

  defp verify(headers, %{url: url, public_key: key}) when is_binary(url) and is_binary(key) do
    Signature.verify(headers, url, key, System.os_time(:millisecond))
  end

  defp verify(_headers, _opts), do: :ok

  defp websocket_upgrade?(conn) do
    conn
    |> get_req_header("upgrade")
    |> Enum.any?(&(String.downcase(&1) == "websocket"))
  end

  defp header(conn, name) do
    conn |> get_req_header(name) |> List.first()
  end
end
