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
  capability boundary of the live-stream codec, not a decoding error, so
  parity has nothing to compare on them.
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
  """
  @spec check_file(Path.t(), keyword()) :: :ok | {:skip, atom()} | {:divergence, divergence()}
  def check_file(path, opts \\ []) do
    # skip_rollback_frames: false — peppi emits every simulation of a
    # frame (rollback re-simulations included) as its own row, so
    # libmelee_ex must too for the two sequences to line up. See
    # ExPhil.Data.EventsPeppiParityTest's "Alignment" note.
    opts = Keyword.put_new(opts, :skip_rollback_frames, false)

    with :ok <- eligible(path),
         {:ok, raw} <- unwrap(path),
         {:ok, replay} <- peppi(path),
         {:ok, frames} <- melee(raw, opts) do
      compare(path, frames, replay.frames)
    end
  end

  defp eligible(path),
    do: if(comparable?(path), do: :ok, else: {:skip, :pre_2_2_no_frame_bookend})

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
  @spec compare(Path.t(), [GameState.t()], [Peppi.GameFrame.t()]) ::
          :ok | {:divergence, divergence()}
  def compare(path, melee_frames, peppi_frames) do
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
        |> Enum.reduce_while(:ok, fn {mf, pf}, _ ->
          case compare_frame(path, mf.frame, mf, pf) do
            :ok -> {:cont, :ok}
            other -> {:halt, other}
          end
        end)
    end
  end

  defp compare_frame(path, frame, m, p) do
    ports = p.players |> Map.keys() |> Enum.sort()

    Enum.reduce_while(ports, :ok, fn port, _ ->
      case {Map.get(m.players, port), Map.get(p.players, port)} do
        {nil, pp} ->
          # An eliminated player stops receiving pre/post frame updates.
          # libmelee_ex drops the port from that frame; peppi keeps a
          # fixed port list for the whole game and emits a ZEROED
          # placeholder row. Accept that, but only if the row really is
          # the all-zero placeholder — a genuinely dropped port would
          # still be caught here.
          if placeholder?(pp),
            do: {:cont, :ok},
            else: {:halt, {:divergence, div(path, frame, port, :player_present, nil, pp)}}

        {mp, pp} ->
          case compare_player(path, frame, port, mp, pp) do
            :ok -> {:cont, :ok}
            other -> {:halt, other}
          end
      end
    end)
  end

  # peppi's stand-in row for a port that is no longer being updated.
  # Position exactly (0.0, 0.0) with zero percent/stock/action makes this
  # unmistakable against real play.
  defp placeholder?(pp) do
    pp.character == 0 and pp.action == 0 and pp.stock == 0 and pp.percent == 0.0 and
      pp.x == 0.0 and pp.y == 0.0
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
      # peppi's NIF stores +1/-1 from the same f32 direction field.
      {:facing, mp.facing, pp.facing > 0},
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

  # KNOWN BUG in the exphil NIF, not in either parser's decoding (see
  # the test moduledoc): the NIF's
  # `character_id/1` treats peppi's post-frame character as an EXTERNAL
  # (CSS-order) id, but it is the INTERNAL id. The table is accidentally
  # the identity for internal 0x00..0x19, so every character below Roy
  # comes out right; internal 0x1A (Roy) and above fall through to -1.
  # Excused here so the harness reports genuine codec divergences, and
  # pinned by a dedicated test so the hole cannot widen unnoticed.
  @roy_internal 0x1A
  defp peppi_character(melee_char, -1) when melee_char >= @roy_internal, do: melee_char
  defp peppi_character(_melee_char, peppi_char), do: peppi_char

  defp compare_player(path, frame, port, mp, pp) do
    mp
    |> field_triples(pp)
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
