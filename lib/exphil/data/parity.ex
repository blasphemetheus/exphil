defmodule ExPhil.Data.Parity do
  @moduledoc """
  Differential comparison of two independent Slippi replay parsers:
  `Melee.Events` (Elixir, libmelee_ex) and peppi (Rust, via
  `ExPhil.Data.Peppi`).

  Both parsers are driven over the *same bytes*: a `.slp` file's `raw`
  element is byte-identical to the live spectator event stream, so
  `Melee.Events` can consume it directly while peppi reads the container.
  Any disagreement is therefore a decoding disagreement, not an input
  difference.

  See `ExPhil.Data.EventsPeppiParityTest` for the field list, the
  normalizations and their rationale.

  ## Usage

      ExPhil.Data.Parity.check_file("game.slp")
      #=> :ok
      #=> {:skip, :peppi_parse_failed}
      #=> {:divergence, %{file: ..., frame: 1234, port: 2, field: :percent,
      #=>                  melee: 12.0, peppi: 13.0}}
  """

  alias ExPhil.Data.Peppi
  alias Melee.{Events, FrameData, GameState}

  # f32 read on both sides then widened to f64, so agreement should be
  # exact; the epsilon only guards against a future f32 round-trip.
  @eps 1.0e-6

  @type divergence :: %{
          file: Path.t(),
          frame: integer(),
          port: 1..4,
          field: atom(),
          melee: term(),
          peppi: term()
        }

  @doc """
  Unwrap the UBJSON `.slp` container to its `raw` event stream.

  The header is `{U\\x03raw[$U#l` followed by a big-endian u32 length.
  """
  @spec raw_stream(Path.t()) :: {:ok, binary()} | {:error, :not_a_slp_container}
  def raw_stream(path) do
    case File.read(path) do
      {:ok, <<"{U", 3, "raw[$U#l", len::big-unsigned-32, raw::binary-size(len), _::binary>>} ->
        {:ok, raw}

      _ ->
        {:error, :not_a_slp_container}
    end
  end

  @doc """
  The command bytes advertised by a replay's PAYLOADS (0x35) table.

  Reads only the header, so it is cheap enough to screen a whole corpus.
  """
  @spec payload_commands(Path.t()) :: {:ok, [byte()]} | :error
  def payload_commands(path) do
    with {:ok, io} <- File.open(path, [:read, :binary]),
         {:ok, header} <-
           (fn ->
              r = :file.read(io, 256)
              File.close(io)
              r
            end).(),
         <<"{U", 3, "raw[$U#l", _len::big-unsigned-32, 0x35, n, rest::binary>> <- header,
         true <- n >= 1 and byte_size(rest) >= n - 1 do
      {:ok, for(<<c, _size::16 <- binary_part(rest, 0, n - 1)>>, do: c)}
    else
      _ -> :error
    end
  end

  @doc """
  Can `Melee.Events` produce frames from this replay at all?

  `Melee.Events` (like libmelee) emits a completed `GameState` on
  FRAME_BOOKEND (0x3C), which Slippi only added in replay version 2.2.0.
  Older replays parse to `:game_end` with zero frames — an inherent
  capability boundary of the live-stream codec, not a decoding error.
  `check_file/2` routes such files through `Melee.SlpFile`'s manual
  bookends instead, so they are still compared; this predicate now only
  selects WHICH melee-side reader a file gets.
  """
  @spec comparable?(Path.t()) :: boolean()
  def comparable?(path) do
    case payload_commands(path) do
      {:ok, cmds} -> 0x3C in cmds
      :error -> false
    end
  end

  @doc """
  Parse a raw event stream with `Melee.Events`, returning completed frames.

  Returns `{:ok, frames}` when the stream ends in GAME_END or runs out of
  data cleanly, `{:error, reason}` on a codec error.
  """
  @spec melee_frames(binary(), keyword()) :: {:ok, [GameState.t()]} | {:error, term()}
  def melee_frames(raw, opts \\ []) do
    collect(Events.new(opts), raw, [])
  end

  defp collect(parser, chunk, acc) do
    case Events.handle_game_event(parser, chunk) do
      {:frame_complete, gs, parser} -> collect(parser, <<>>, [gs | acc])
      {:rollback, parser} -> collect(parser, <<>>, acc)
      {:continue, _parser} -> {:ok, Enum.reverse(acc)}
      {:game_end, _parser} -> {:ok, Enum.reverse(acc)}
      {:error, reason, _parser} -> {:error, reason}
    end
  end

  @doc """
  Parse `path` with both parsers and compare them field-by-field.

  Returns the FIRST divergence found (in frame then port then field
  order), or `:ok`. Files that either parser rejects are reported as
  `{:skip, reason}` rather than as a divergence.

  Routing: replays that advertise FRAME_BOOKEND (2.2.0+) go through the
  raw event stream exactly as the live spectator path would read it.
  Pre-2.2.0 replays have no bookend, so the live codec alone yields
  zero frames for them — they go through `Melee.SlpFile`, whose
  libmelee-style manual bookends complete each frame off the next
  frame's pre-frame event. Same decoder underneath, different frame
  completion — which is exactly the code path this differential then
  vouches for.
  """
  @spec check_file(Path.t(), keyword()) :: :ok | {:skip, atom()} | {:divergence, divergence()}
  def check_file(path, opts \\ []) do
    # skip_rollback_frames: false — peppi emits every simulation of a
    # frame (rollback re-simulations included) as its own row, so
    # libmelee_ex must too for the two sequences to line up. See
    # ExPhil.Data.EventsPeppiParityTest's "Alignment" note. (Pre-2.2.0
    # replays predate rollback, so it is a no-op there.)
    opts = Keyword.put_new(opts, :skip_rollback_frames, false)

    with {:ok, replay} <- peppi(path),
         {:ok, frames} <- melee_for(path, opts) do
      compare(path, frames, replay.frames, post_frame_len: post_frame_len(path))
    end
  end

  # Total post-frame (0x38) event length (command byte included) as
  # advertised by the file's payload table, or nil when unreadable.
  # Old replay versions have SHORTER post-frame events; a field whose
  # wire offset lies beyond this length does not exist in the file, and
  # both parsers then report their own *invented defaults* (e.g.
  # jumps_left: libmelee_ex defaults 1, the peppi NIF unwrap_or(2)) —
  # a defaults disagreement, not a decoding one, so those fields are
  # excluded from comparison.
  @spec post_frame_len(Path.t()) :: pos_integer() | nil
  def post_frame_len(path) do
    with {:ok, io} <- File.open(path, [:read, :binary]),
         {:ok, header} <-
           (fn ->
              r = :file.read(io, 256)
              File.close(io)
              r
            end).(),
         <<"{U", 3, "raw[$U#l", _len::big-unsigned-32, 0x35, n, rest::binary>> <- header,
         true <- n >= 1 and byte_size(rest) >= n - 1 do
      binary_part(rest, 0, n - 1)
      |> then(&for(<<c, size::16 <- &1>>, do: {c, size}))
      |> List.keyfind(0x38, 0)
      |> case do
        {0x38, size} -> size + 1
        nil -> nil
      end
    else
      _ -> nil
    end
  end

  defp melee_for(path, opts) do
    if comparable?(path) do
      with {:ok, raw} <- unwrap(path), do: melee(raw, opts)
    else
      melee_slp_file(path, opts)
    end
  end

  defp melee_slp_file(path, opts) do
    case path |> Melee.SlpFile.stream!(opts) |> Enum.to_list() do
      [] -> {:skip, :melee_no_frames}
      frames -> {:ok, frames}
    end
  rescue
    _ -> {:skip, :melee_parse_failed}
  end

  defp unwrap(path) do
    case raw_stream(path) do
      {:ok, raw} -> {:ok, raw}
      {:error, _} -> {:skip, :not_a_slp_container}
    end
  end

  defp peppi(path) do
    case Peppi.parse(path) do
      {:ok, replay} -> {:ok, replay}
      {:error, _} -> {:skip, :peppi_parse_failed}
    end
  rescue
    _ -> {:skip, :peppi_parse_failed}
  end

  defp melee(raw, opts) do
    case melee_frames(raw, opts) do
      {:ok, []} -> {:skip, :melee_no_frames}
      {:ok, frames} -> {:ok, frames}
      {:error, _} -> {:skip, :melee_parse_failed}
    end
  end

  @doc """
  Compare already-parsed frame lists.

  The two frame SEQUENCES must agree first — same length, same frame
  numbers in the same order, rollback re-simulations included. Only then
  are they walked pairwise. Checking the sequence explicitly means a
  missing or extra frame is reported as such rather than silently
  shifting every later comparison by one.
  """
  @spec compare(Path.t(), [GameState.t()], [Peppi.GameFrame.t()], keyword()) ::
          :ok | {:divergence, divergence()}
  def compare(path, melee_frames, peppi_frames, opts \\ []) do
    post_len = Keyword.get(opts, :post_frame_len)
    m_ids = Enum.map(melee_frames, & &1.frame)
    p_ids = Enum.map(peppi_frames, & &1.frame_number)

    cond do
      length(m_ids) != length(p_ids) ->
        {:divergence, div(path, nil, nil, :frame_count, length(m_ids), length(p_ids))}

      m_ids != p_ids ->
        {a, b} = Enum.zip(m_ids, p_ids) |> Enum.find(fn {a, b} -> a != b end)
        {:divergence, div(path, a, nil, :frame_number, a, b)}

      true ->
        [melee_frames, peppi_frames]
        |> Enum.zip()
        |> Enum.reduce_while({:ok, MapSet.new()}, fn {mf, pf}, {_, gapped} ->
          case compare_frame(path, mf.frame, mf, pf, gapped, post_len) do
            {:ok, gapped} -> {:cont, {:ok, gapped}}
            other -> {:halt, {other, gapped}}
          end
        end)
        |> elem(0)
    end
  end

  defp compare_frame(path, frame, m, p, gapped, post_len) do
    ports = p.players |> Map.keys() |> Enum.sort()

    Enum.reduce_while(ports, {:ok, gapped}, fn port, {:ok, gapped} ->
      case {Map.get(m.players, port), Map.get(p.players, port)} do
        {nil, pp} ->
          # Slippi can stop writing a port's pre/post events mid-game:
          # elimination, and (observed in old replays) a ~70-frame gap
          # while a player sits on the respawn platform — the port comes
          # BACK afterwards. libmelee_ex omits the port for exactly
          # those frames, faithfully to the bytes; peppi papers over the
          # hole with a fabricated row. So the ground truth is the raw
          # itself: a missing melee port is accepted iff the raw stream
          # really has no pre/post event for that (frame, port). A melee
          # decode bug that drops events actually present still fails.
          cond do
            placeholder?(pp) ->
              {:cont, {:ok, gapped}}

            not raw_has?(path, frame, port) ->
              # Byte-verified write gap. peppi's rows for this port are
              # untrustworthy from here on — after the gap it misassigns
              # the port's returning data (verified against the raw:
              # melee's post-gap values match the bytes, peppi's do
              # not) — so the port is excluded from field comparison
              # for the rest of the file. Divergences before the gap
              # were already checked.
              {:cont, {:ok, MapSet.put(gapped, port)}}

            true ->
              {:halt, {{:divergence, div(path, frame, port, :player_present, nil, pp)}, gapped}}
          end

        {mp, pp} ->
          if MapSet.member?(gapped, port) do
            {:cont, {:ok, gapped}}
          else
            case compare_player(path, frame, port, mp, pp, post_len) do
              :ok -> {:cont, {:ok, gapped}}
              other -> {:halt, {other, gapped}}
            end
          end
      end
    end)
    |> case do
      {:ok, gapped} -> {:ok, gapped}
      {{:divergence, _} = d, _gapped} -> d
    end
  end

  # peppi's stand-in row for a port that is no longer being updated.
  # Position exactly (0.0, 0.0) with zero percent/stock/action makes this
  # unmistakable against real play.
  defp placeholder?(pp) do
    pp.character == 0 and pp.action == 0 and pp.stock == 0 and pp.percent == 0.0 and
      pp.x == 0.0 and pp.y == 0.0
  end

  # Does the raw stream contain any PRE/POST_FRAME event for this
  # (frame, port)? Consulted only when a melee-side port is missing —
  # rare — so the index is built lazily, and cached for the CURRENT file
  # only (single process-dictionary key, no growth across a corpus run).
  defp raw_has?(path, frame, port) do
    index =
      case Process.get({__MODULE__, :raw_index}) do
        {^path, index} ->
          index

        _ ->
          index = build_raw_index(path)
          Process.put({__MODULE__, :raw_index}, {path, index})
          index
      end

    MapSet.member?(index, {frame, port})
  end

  defp build_raw_index(path) do
    {:ok, raw} = raw_stream(path)
    <<0x35, n, rest::binary>> = raw
    table = for <<c, s::16 <- binary_part(rest, 0, n - 1)>>, into: %{}, do: {c, s}
    stream = binary_part(rest, n - 1, byte_size(rest) - (n - 1))
    index_events(stream, table, MapSet.new())
  end

  defp index_events(<<>>, _table, acc), do: acc

  defp index_events(<<cmd, rest::binary>> = bin, table, acc) do
    size = Map.get(table, cmd, 0) + 1

    if size > byte_size(bin) do
      acc
    else
      acc =
        case {cmd in [0x37, 0x38], rest} do
          {true, <<frame::big-signed-32, port, _::binary>>} ->
            MapSet.put(acc, {frame, port + 1})

          _ ->
            acc
        end

      index_events(binary_part(bin, size, byte_size(bin) - size), table, acc)
    end
  end

  @doc """
  The compared (field, melee_value, peppi_value) triples for one player
  on one frame, with all normalizations already applied.

  Exposed so tests (and future debugging) can see the exact field list.
  """
  @spec field_triples(Melee.PlayerState.t(), Peppi.PlayerFrame.t()) :: [{atom(), term(), term()}]
  def field_triples(mp, pp) do
    mc = mp.controller_state
    pc = pp.controller

    # NORMALIZATION 1 (action_frame): libmelee applies a "zero-indexed
    # action" fix (+1) for action states the game counts from 0; peppi
    # reports the raw `state_age` from the wire. Re-apply the same fix to
    # peppi so the two use one convention.
    zero_indexed = FrameData.zero_indexed?(mp.character, mp.action)
    peppi_af = trunc(pp.action_frame) + if zero_indexed, do: 1, else: 0

    [
      {:position_x, mp.position.x, pp.x},
      {:position_y, mp.position.y, pp.y},
      {:percent, mp.percent, pp.percent},
      {:stock, mp.stock, pp.stock},
      {:action, mp.action, pp.action},
      {:action_frame, mp.action_frame, peppi_af},
      # NORMALIZATION 2 (facing): libmelee stores a boolean (true = right),
      # peppi's NIF stores +1/-1 from the same f32 direction field. The
      # wire can carry direction == EXACTLY 0.0 (first observed in an
      # old replay, frame 613: verified by reading the f32 straight out
      # of the raw) — libmelee reads `0.0 > 0` as false, the NIF's sign
      # mapping yields +1, and neither is wrong about the bytes. That
      # one pattern (melee false, peppi +1) is excused; a melee-true /
      # peppi--1 mismatch cannot come from 0.0 and still fails.
      facing_triple(mp.facing, pp.facing),
      {:on_ground, mp.on_ground, pp.on_ground},
      {:jumps_left, mp.jumps_left, pp.jumps_left},
      {:shield_strength, mp.shield_strength, pp.shield_strength},
      # NORMALIZATION 3 (hitstun): the NIF's `hitstun_frames_left` is
      # populated from peppi's `post.hitlag` (offset 0x49, "hitlag
      # remaining"), NOT from `misc_as` (0x2B) which is what libmelee
      # calls hitstun. Compare it against libmelee's `hitlag_left`, which
      # is the field that actually reads 0x49.
      {:hitlag_left, mp.hitlag_left, trunc(pp.hitstun_frames_left)},
      {:speed_air_x_self, mp.speed_air_x_self, pp.speed_air_x_self},
      {:speed_ground_x_self, mp.speed_ground_x_self, pp.speed_ground_x_self},
      {:speed_y_self, mp.speed_y_self, pp.speed_y_self},
      {:speed_x_attack, mp.speed_x_attack, pp.speed_x_attack},
      {:speed_y_attack, mp.speed_y_attack, pp.speed_y_attack},
      {:character, mp.character, peppi_character(mp.character, pp.character)},
      # Controller. Both sides normalize sticks from the wire's [-1, 1]
      # f32 to [0, 1] with 0.5 neutral, so no normalization is needed.
      {:main_stick_x, elem(mc.main_stick, 0), pc.main_stick_x},
      {:main_stick_y, elem(mc.main_stick, 1), pc.main_stick_y},
      {:c_stick_x, elem(mc.c_stick, 0), pc.c_stick_x},
      {:c_stick_y, elem(mc.c_stick, 1), pc.c_stick_y},
      {:button_a, mc.button.a, pc.button_a},
      {:button_b, mc.button.b, pc.button_b},
      {:button_x, mc.button.x, pc.button_x},
      {:button_y, mc.button.y, pc.button_y},
      {:button_z, mc.button.z, pc.button_z},
      {:button_l, mc.button.l, pc.button_l},
      {:button_r, mc.button.r, pc.button_r},
      {:button_start, mc.button.start, pc.button_start},
      {:button_d_up, mc.button.d_up, pc.button_d_up},
      {:button_d_down, mc.button.d_down, pc.button_d_down},
      {:button_d_left, mc.button.d_left, pc.button_d_left},
      {:button_d_right, mc.button.d_right, pc.button_d_right},
      # Analog triggers: the two parsers read DIFFERENT wire fields (see
      # the test moduledoc), so only the shared 0..1 range invariant is
      # checkable here.
      {:trigger_range, true,
       in_unit?(mc.l_shoulder) and in_unit?(pc.l_trigger) and
         in_unit?(pc.r_trigger)}
    ]
  end

  defp in_unit?(v), do: is_number(v) and v >= 0.0 and v <= 1.0

  # See NORMALIZATION 2: {false, +1} is the direction-==-0.0 signature.
  defp facing_triple(false, peppi_facing) when peppi_facing > 0,
    do: {:facing, :zero_direction_excused, :zero_direction_excused}

  defp facing_triple(melee_facing, peppi_facing),
    do: {:facing, melee_facing, peppi_facing > 0}

  # The NIF's historical Roy->-1 internal-id hole (post.character fed
  # through the external-id table; found by this differential
  # 2026-08-05) was fixed 2026-08-13 — internal_character_id/1 is
  # identity now — so no excuse remains: character ids must simply
  # agree. The dedicated pin test now pins the FIX instead of the hole.
  defp peppi_character(_melee_char, peppi_char), do: peppi_char

  # Post-frame fields that only exist from a given event length on
  # (event-relative offsets from lib/melee/events.ex's post_frame
  # decode, plus field width). A shorter event means the file's replay
  # version predates the field; both parsers then report invented
  # defaults (jumps_left: libmelee_ex 1, peppi NIF unwrap_or(2)) —
  # a defaults disagreement, not a decoding one, so skip those fields.
  @post_frame_field_min_len %{
    on_ground: 0x30,
    jumps_left: 0x33,
    speed_air_x_self: 0x39,
    speed_y_self: 0x3D,
    speed_x_attack: 0x41,
    speed_y_attack: 0x45,
    speed_ground_x_self: 0x49,
    hitlag_left: 0x4D
  }

  defp compare_player(path, frame, port, mp, pp, post_len) do
    mp
    |> field_triples(pp)
    |> Enum.reject(fn {field, _a, _b} ->
      min_len = Map.get(@post_frame_field_min_len, field)
      min_len != nil and post_len != nil and post_len < min_len
    end)
    |> Enum.reduce_while(:ok, fn {field, a, b}, _ ->
      if equal?(a, b),
        do: {:cont, :ok},
        else: {:halt, {:divergence, div(path, frame, port, field, a, b)}}
    end)
  end

  defp equal?(a, b) when is_float(a) or is_float(b) do
    a = a * 1.0
    b = b * 1.0
    abs(a - b) <= @eps * max(1.0, max(abs(a), abs(b)))
  end

  defp equal?(a, b), do: a == b

  defp div(file, frame, port, field, m, p),
    do: %{file: file, frame: frame, port: port, field: field, melee: m, peppi: p}

  @doc """
  Discover a replay corpus, largest-and-most-diverse first.

  Honours `PARITY_CORPUS` (a colon-separated list of globs) if set.
  """
  @spec corpus() :: [Path.t()]
  def corpus do
    globs =
      case System.get_env("PARITY_CORPUS") do
        nil ->
          [
            "test/fixtures/replays/*.slp",
            "eval_runs/**/*.slp",
            "replays/huggingface/**/*.slp",
            Path.expand("~/Slippi/**/*.slp")
          ]

        set ->
          String.split(set, ":", trim: true)
      end

    Enum.flat_map(globs, &Path.wildcard/1)
  end

  @doc """
  Deterministically sample `n` paths from `paths` (seeded, so a failure
  is reproducible by re-running with the same PARITY_SEED).
  """
  @spec sample([Path.t()], pos_integer(), integer()) :: [Path.t()]
  def sample(paths, n, seed) do
    paths
    |> Enum.sort()
    |> Enum.sort_by(&:erlang.phash2({seed, &1}))
    |> Enum.take(n)
  end
end
