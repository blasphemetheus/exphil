defmodule ExPhil.Stomped.ServerTest do
  # Plays the arena's side against a real server on a loopback port.
  use ExUnit.Case, async: true

  alias ExPhil.Stomped.{Pad, Server, Signature}
  alias ExPhil.Stomped.Bots.Walker
  alias ExPhil.Test.StompedStream, as: Stream

  @path "/secret-7f3a9c2e"

  defp start_server(opts \\ []) do
    opts = Keyword.merge([port: 0, path: @path, bot: {Walker, character: 2}], opts)
    server = start_supervised!({Server, opts})
    Server.port(server)
  end

  defp get(port, path) do
    Req.get!("http://localhost:#{port}#{path}", retry: false)
  end

  # --- a minimal WebSocket client over Mint -------------------------------

  defp connect(port, path, headers \\ []) do
    {:ok, conn} = Mint.HTTP.connect(:http, "localhost", port, protocols: [:http1])
    {:ok, conn, ref} = Mint.WebSocket.upgrade(:ws, conn, path, headers)
    {conn, responses} = await_done(conn, ref, [])
    status = for({:status, ^ref, status} <- responses, do: status) |> List.first()
    resp_headers = for({:headers, ^ref, h} <- responses, do: h) |> List.flatten()

    if status == 101 do
      {:ok, conn, websocket} = Mint.WebSocket.new(conn, ref, status, resp_headers)
      {:ok, %{conn: conn, ref: ref, websocket: websocket}}
    else
      Mint.HTTP.close(conn)
      {:error, status}
    end
  end

  defp await_done(conn, ref, acc) do
    receive do
      message ->
        {:ok, conn, responses} = Mint.WebSocket.stream(conn, message)
        acc = acc ++ responses

        if Enum.any?(acc, &match?({:done, ^ref}, &1)) do
          {conn, acc}
        else
          await_done(conn, ref, acc)
        end
    after
      2_000 -> flunk("no response to the upgrade request")
    end
  end

  defp push(client, frame) do
    {:ok, websocket, data} = Mint.WebSocket.encode(client.websocket, frame)
    {:ok, conn} = Mint.WebSocket.stream_request_body(client.conn, client.ref, data)
    %{client | conn: conn, websocket: websocket}
  end

  # Collect `count` frames from the server.
  defp pull(client, count, acc \\ [])
  defp pull(client, count, acc) when length(acc) >= count, do: {Enum.take(acc, count), client}

  defp pull(client, count, acc) do
    ref = client.ref

    receive do
      message ->
        {:ok, conn, responses} = Mint.WebSocket.stream(client.conn, message)

        {websocket, frames} =
          Enum.reduce(responses, {client.websocket, []}, fn
            {:data, ^ref, data}, {websocket, frames} ->
              {:ok, websocket, decoded} = Mint.WebSocket.decode(websocket, data)
              {websocket, frames ++ decoded}

            _other, acc ->
              acc
          end)

        pull(%{client | conn: conn, websocket: websocket}, count, acc ++ frames)
    after
      2_000 -> flunk("expected #{count} frames from the server, got #{inspect(acc)}")
    end
  end

  # -------------------------------------------------------------------------

  describe "http" do
    test "GET <url>/ping answers pong" do
      port = start_server()
      assert %{status: 200, body: "pong"} = get(port, @path <> "/ping")
    end

    test "anything outside the secret path is a 404" do
      port = start_server()
      assert %{status: 404} = get(port, "/ping")
      assert %{status: 404} = get(port, "/")
      assert %{status: 404} = get(port, "/other/ping")
    end

    test "the socket URL without an upgrade is a 400" do
      port = start_server()
      assert %{status: 400} = get(port, @path)
    end
  end

  describe "handshake verification" do
    setup do
      {public_key, private_key} = :crypto.generate_key(:eddsa, :ed25519)
      url = "https://bots.example.com" <> @path
      %{public_key: public_key, private_key: private_key, url: url}
    end

    defp signed_headers(ctx, match_id, opts \\ []) do
      timestamp = Integer.to_string(Keyword.get(opts, :at, System.os_time(:millisecond)))
      url = Keyword.get(opts, :url, ctx.url)
      message = timestamp <> "." <> match_id <> "." <> url
      signature = :crypto.sign(:eddsa, :none, message, [ctx.private_key, :ed25519])

      [
        {"x-stomped-match", match_id},
        {"x-stomped-timestamp", timestamp},
        {"x-stomped-signature", Base.encode64(signature)}
      ]
    end

    test "accepts a handshake signed for our URL (the deploy check)", ctx do
      port = start_server(url: ctx.url, public_key: ctx.public_key)
      assert {:ok, _client} = connect(port, @path, signed_headers(ctx, "deploy-check"))
    end

    test "refuses an unsigned upgrade with 401", ctx do
      port = start_server(url: ctx.url, public_key: ctx.public_key)
      assert {:error, 401} = connect(port, @path)
    end

    test "refuses a handshake captured at another bot", ctx do
      port = start_server(url: ctx.url, public_key: ctx.public_key)
      headers = signed_headers(ctx, "Xk3p9QaB", url: "https://other.example.com/abc")
      assert {:error, 401} = connect(port, @path, headers)
    end

    test "refuses a stale timestamp", ctx do
      port = start_server(url: ctx.url, public_key: ctx.public_key)
      headers = signed_headers(ctx, "Xk3p9QaB", at: System.os_time(:millisecond) - 60_000)
      assert {:error, 401} = connect(port, @path, headers)
    end

    test "decodes the published key format" do
      assert {:ok, <<_::binary-size(32)>>} =
               Signature.decode_public_key("BbAAxiRTApJVuhxobkZ9RoYx5CKBmKM58frzsLQDWrc=")

      assert {:error, :bad_public_key} = Signature.decode_public_key("c2hvcnQ=")
    end
  end

  test "plays a whole match over the socket" do
    port = start_server()
    {:ok, client} = connect(port, @path, [{"x-stomped-match", "Xk3p9QaB"}])

    # character → the pick
    client = push(client, {:text, Stream.step("character", %{"port" => 0})})
    {[{:text, pick}], client} = pull(client, 1)
    assert Jason.decode!(pick) == %{"step" => "character", "character" => 2, "costume" => 0}

    # ready → ready
    client = push(client, {:text, Stream.step("ready", %{"port" => 0})})
    {[{:text, ready}], client} = pull(client, 1)
    assert Jason.decode!(ready) == %{"step" => "ready"}

    # game → neutral pads for the two frames already open
    client = push(client, {:text, Stream.step("game", %{"port" => 0, "frame" => -124})})
    {[{:binary, first}, {:binary, second}], client} = pull(client, 2)
    assert first == Pad.neutral(-123)
    assert second == Pad.neutral(-122)

    # the stream: header, then one pad back per frame
    client = push(client, {:binary, Stream.header()})
    client = push(client, {:binary, Stream.frame(-123)})
    {[{:binary, pad}], client} = pull(client, 1)
    assert {:ok, %{frame: -121, stick_x: 188}} = Pad.decode(pad)

    client = push(client, {:binary, Stream.frame(-122, %{0 => -2.0, 1 => 2.0})})
    {[{:binary, pad}], client} = pull(client, 1)
    assert {:ok, %{frame: -120, stick_x: 128}} = Pad.decode(pad)

    # the end: nothing comes back
    client = push(client, {:binary, Stream.game_end()})
    over = Stream.step("over", %{"port" => 0, "winner" => 0, "endReason" => "stocks"})
    client = push(client, {:text, over})
    _client = push(client, :close)
  end
end
