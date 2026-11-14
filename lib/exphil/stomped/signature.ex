defmodule ExPhil.Stomped.Signature do
  @moduledoc """
  Verifies the signed WebSocket handshake of the stomped.dev arena.

  The upgrade request carries `x-stomped-match`, `x-stomped-timestamp` (unix
  milliseconds) and `x-stomped-signature`, an Ed25519 signature over
  `"<timestamp>.<matchId>.<url>"` under the platform key, where `<url>` is the
  bot's registered URL byte for byte. Checking it stops anyone who learns the
  URL from starting games on this server.
  """

  @key_url "https://api.stomped.dev/.well-known/stomped.json"
  # The arena's rule: reject a timestamp more than 30 seconds from our clock.
  @max_skew_ms 30_000

  @type reason :: :missing_headers | :bad_timestamp | :stale_timestamp | :bad_signature

  @doc "Where the platform public key is published."
  def key_url, do: @key_url

  @doc """
  Fetch the platform public key (the raw 32 bytes).

  Meant to be called once at startup; the key is passed to the server config.
  """
  @spec fetch_public_key(String.t()) :: {:ok, <<_::256>>} | {:error, term()}
  def fetch_public_key(url \\ @key_url) do
    case Req.get(url, retry: false, receive_timeout: 5_000) do
      {:ok, %{status: 200, body: %{"publicKey" => encoded}}} -> decode_public_key(encoded)
      {:ok, %{status: status}} -> {:error, {:http_status, status}}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc "Decode the base64 public key from the well-known document."
  @spec decode_public_key(String.t()) :: {:ok, <<_::256>>} | {:error, :bad_public_key}
  def decode_public_key(encoded) when is_binary(encoded) do
    case Base.decode64(encoded) do
      {:ok, <<key::binary-size(32)>>} -> {:ok, key}
      _ -> {:error, :bad_public_key}
    end
  end

  @doc """
  Verify a handshake.

  `headers` holds the three `x-stomped-*` values (any may be `nil`), `url` is
  the registered URL and `now_ms` the local clock in unix milliseconds.
  """
  @spec verify(
          %{match: String.t() | nil, timestamp: String.t() | nil, signature: String.t() | nil},
          String.t(),
          binary(),
          integer()
        ) :: :ok | {:error, reason()}
  def verify(%{match: match, timestamp: timestamp, signature: signature}, url, public_key, now_ms)
      when is_binary(match) and is_binary(timestamp) and is_binary(signature) do
    with {:ok, sent_ms} <- parse_timestamp(timestamp),
         :ok <- check_skew(sent_ms, now_ms),
         {:ok, raw_signature} <- decode_signature(signature) do
      message = timestamp <> "." <> match <> "." <> url

      if :crypto.verify(:eddsa, :none, message, raw_signature, [public_key, :ed25519]) do
        :ok
      else
        {:error, :bad_signature}
      end
    end
  end

  def verify(_headers, _url, _public_key, _now_ms), do: {:error, :missing_headers}

  defp parse_timestamp(timestamp) do
    case Integer.parse(timestamp) do
      {ms, ""} -> {:ok, ms}
      _ -> {:error, :bad_timestamp}
    end
  end

  defp check_skew(sent_ms, now_ms) do
    if abs(now_ms - sent_ms) <= @max_skew_ms, do: :ok, else: {:error, :stale_timestamp}
  end

  defp decode_signature(signature) do
    case Base.decode64(signature) do
      {:ok, <<raw::binary-size(64)>>} -> {:ok, raw}
      _ -> {:error, :bad_signature}
    end
  end
end
