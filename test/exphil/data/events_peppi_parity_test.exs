defmodule ExPhil.Data.EventsPeppiParityTest do
  @moduledoc """
  Differential correctness: `Melee.Events` (Elixir codec, libmelee_ex) vs
  peppi (Rust, `ExPhil.Data.Peppi`) over a corpus of real `.slp` replays.

  Two fully independent implementations decode the *same bytes* — a
  `.slp` file's `raw` element is byte-identical to the live spectator
  event stream, so `Melee.Events` consumes it directly while peppi reads
  the container. Every per-(frame, port) field below must agree. This is
  the strongest available evidence that the Elixir codec's VALUES are
  right at scale; the earlier corpus sweep only proved it does not crash.

  ## Fields compared (per frame, per port)

  Post-frame state: `position_x`, `position_y`, `percent`, `stock`,
  `action`, `action_frame`, `facing`, `on_ground`, `jumps_left`,
  `shield_strength`, `hitlag_left`, `character`, and all five self/attack
  speed components (`speed_air_x_self`, `speed_ground_x_self`,
  `speed_y_self`, `speed_x_attack`, `speed_y_attack`).

  Pre-frame inputs: `main_stick_x/y`, `c_stick_x/y`, and the twelve
  physical buttons (a, b, x, y, z, l, r, start, d_up/down/left/right).

  ## Normalizations, and why each is legitimate

  1. **`action_frame` zero-indexing.** libmelee (and libmelee_ex,
     `Events.fix_frame_indexing/1`) adds 1 to the wire's `state_age` for
     action states the game counts from zero. peppi reports the raw
     value. The test re-applies `Melee.FrameData.zero_indexed?/2` to the
     peppi side. Without it, ~2200 of 3800 player-frames in a single
     1900-frame replay "diverge"; with it, zero do.

  2. **`facing` bool vs ±1.** Both read the same f32 direction field;
     libmelee stores `> 0` as a boolean, the NIF stores `+1`/`-1`. The
     test compares `melee.facing == (peppi.facing > 0)`. (Note the two
     disagree in principle at exactly `direction == 0.0`, which does not
     occur in the corpus.)

  3. **`hitstun_frames_left` is really hitlag.** The NIF populates its
     `hitstun_frames_left` field from peppi's `post.hitlag` — Slippi
     post-frame offset **0x49, "Hitlag Remaining"** — not from `misc_as`
     (0x2B), which is what libmelee calls hitstun. So the comparison is
     against libmelee_ex's `hitlag_left`, which reads the same 0x49.
     Measured: 0 divergences against `hitlag_left`, 614 in one replay
     against `hitstun_frames_left`. This is a *mislabel in the exphil
     NIF*, not a bug in either parser's decoding.

  4. **Sticks need no normalization.** Both sides map the wire's f32
     `[-1, 1]` to `[0, 1]` with 0.5 neutral. Verified byte-exact.

  5. **Analog triggers are NOT directly comparable** (only a 0..1 range
     invariant is asserted). libmelee — and therefore libmelee_ex —
     deliberately reports the *processed* trigger (pre-frame **0x29**)
     in both `l_shoulder` and `r_shoulder`, because the game interprets
     both shoulders together. The exphil NIF reports peppi's
     `triggers_physical` (**0x33**/**0x37**). These are different wire
     fields with a non-invertible relationship: deadzone clips physical
     values to a processed 0, digital presses drive processed to 1.0
     with a low physical reading, and Dolphin bot replays write a
     processed value with both physicals at 0. Both directions of the
     inequality were measured to fail on real data, so no normalization
     exists. Digital `button_l`/`button_r` (0x31) *are* compared and
     agree exactly, so shoulder input is still covered.

  6. **Eliminated ports.** Once a player is out (0 stocks), Slippi stops
     emitting pre/post frame updates for that port. libmelee_ex drops the
     port from the frame; peppi keeps a fixed port list for the whole
     game and emits an all-zero placeholder row (character 0, action 0,
     percent 0.0, position exactly (0.0, 0.0)). Observed in a 4-player
     replay at frame 8085. The harness accepts a missing melee port only
     when peppi's row is exactly that placeholder, so a genuinely dropped
     port would still fail.

  7. **`character` — a REAL BUG the sweep found, in the exphil NIF.**
     Both parsers read the post-frame character byte, which is the
     **internal** character id. The NIF's `character_id/1`
     (`native/exphil_peppi/src/lib.rs`) documents its input as peppi's
     "external (CSS-order)" id and maps accordingly — but peppi's
     `post.character` is internal. The table happens to be the identity
     over `0x00..0x19`, so every character below Roy still comes out with
     the right *number*; internal **0x1A (Roy) and above fall through to
     `-1`**. Real consequence for exphil: Roy replays feed
     `character: -1` into the embeddings. (`character_name/1` is wrong
     for essentially every character — it labels internal ids with
     external names, e.g. internal 1 = Fox is reported "Donkey Kong".)
     **FIXED 2026-08-13** (`internal_character_id/1` in the NIF is
     pinned as identity; the rebuilt .so ships it). The harness no
     longer excuses anything — character ids must simply agree — and
     the dedicated test below now pins the FIX instead of the hole.

  8. **`invulnerable` is excluded.** libmelee reads the hurtbox
     collision state (0x34); the NIF derives it from ad-hoc
     `state_flags` bit tests. Genuinely different semantics, not a
     representation difference — comparing them would be meaningless.

  ## Alignment and rollback (the one substantive normalization)

  The two frame SEQUENCES are compared first — same length, same frame
  numbers, same order — and only then walked pairwise, so a missing or
  extra frame is reported as `:frame_count`/`:frame_number` instead of
  shifting every later comparison by one.

  Netplay `.slp` files DO contain rollback re-simulations, and peppi
  emits each simulation as its own row. `Melee.Events` defaults to
  `skip_rollback_frames: true` (it keeps the FIRST simulation and drops
  later ones, which is right for a live agent). Diffing those two
  directly is an apples-to-oranges comparison: on a real netplay replay
  it produced 8920 melee frames vs 9119 peppi rows, and — because
  last-wins dedup silently re-pairs them — ~370 "divergences" concentrated
  on the remote player's inputs and positions, exactly the values a
  rollback revises. The harness therefore parses with
  **`skip_rollback_frames: false`**, at which point the two emit the
  identical 9119-frame sequence (8920 distinct frame numbers) with **zero**
  field divergences. Local/console replays have no rollbacks and are
  unaffected.

  ## Corpus eligibility (a real finding)

  `Melee.Events`, like libmelee, completes a frame on **FRAME_BOOKEND
  (0x3C)**, an event Slippi only added in replay version **2.2.0**. Older
  replays parse to a clean `:game_end` with **zero frames** — silently
  empty, not an error. Measured over the local corpora: 9092 of 9995
  huggingface replays (91%) are pre-2.2.0 and yield nothing; all 683
  eval_runs and all 19 fixtures are 2.2.0+. This is an inherent boundary
  of a live-spectator codec (the live stream is always modern), not a
  parser bug. Since 2026-08-14 those files are **no longer skipped**:
  `Parity.check_file/2` routes them through `Melee.SlpFile`'s manual
  bookends, and a dedicated test below samples exactly that population.
  The main test still uses `Parity.comparable?/1` to sample the modern
  population, where anything with a bookend that yields no frames
  counts as a hard failure, not a skip.

  What the extension itself found (none visible without a differential):

    * `Melee.SlpFile` dropped the FINAL frame of every GAME_END-ending
      manual-bookend replay (no successor pre-frame ever completes it) —
      every old corpus file came up exactly one frame short. Fixed in
      libmelee_ex `83bcc6a`.
    * An early-rollback-era pre-2.2.0 replay can carry the SAME frame
      twice (frame 6176 duplicated, observed once in 9,092 files).
      `Melee.SlpFile` merged the re-simulation into the first
      simulation; it now treats the repeat as a frame boundary with
      normal rollback semantics (libmelee_ex `f278bfa`).
    * Fields absent at a file's replay version (jumps at 0x32, speeds,
      hitlag, on_ground) are reported as *invented defaults* by both
      parsers — and the defaults differ (libmelee_ex `jumps_left: 1`,
      the peppi NIF `unwrap_or(2)`). The harness gates field
      comparisons on the file's advertised post-frame payload length
      (`Parity.post_frame_len/1`).
    * The wire's facing direction can be EXACTLY 0.0 (one frame in the
      corpus, byte-verified) — libmelee reads false, the NIF's sign
      map reads +1, neither wrong about the bytes. Normalization 2
      excuses precisely that pattern.
    * Old Slippi can STOP WRITING a port's events mid-game (~70 frames
      while the player sits on the respawn platform, then resume).
      libmelee_ex omits the port for exactly the event-less frames —
      byte-faithful — while **peppi fabricates rows over the hole and
      then misassigns the port's returning data** (verified against
      the raw: melee's post-gap values match the bytes, peppi's do
      not). The harness accepts a missing melee port iff the raw
      really has no events for it, and excludes that port from field
      comparison for the remainder of the file.

  ## Running

      # default: 25 replays, tagged :slow
      mix test test/exphil/data/events_peppi_parity_test.exs --include slow

      # meaningful scale
      PARITY_SAMPLE=500 mix test test/exphil/data/events_peppi_parity_test.exs \\
        --include slow

  `PARITY_SEED` makes the sample reproducible; `PARITY_CORPUS` overrides
  the corpus globs (colon-separated).

  ## Result of the scale run (2026-08-05)

  Run over the **entire eligible corpus**: 2625 of 11717 discovered
  replays (fixtures + eval_runs + huggingface + `~/Slippi`) advertise a
  frame bookend; every one of them was parsed by both implementations and
  compared field-by-field — **19.1M frames / 39.5M player-frames / ~1.34
  BILLION field comparisons**. **Zero divergences**, 358s wall.

  The only skips were files peppi itself refused to parse (10 of the
  first 400 checked — truncated bot eval recordings from `eval_runs/`).
  `Melee.Events` read all of those, so on this corpus the Elixir codec is
  strictly the more tolerant of the two, and is byte-for-byte in
  agreement everywhere both can read.

  ## Result of the pre-2.2.0 scale run (2026-08-16)

  The entire pre-2.2.0 population of the corpus — **9,092 replays**,
  every one routed through `Melee.SlpFile`'s manual bookends — was
  compared field-by-field against peppi: **9,087 OK, 0 divergences**,
  45 min wall. The only 5 skips were Rust panics inside peppi's own
  deserializer (`peppi-2.1.2 de.rs` assertion failures); libmelee_ex
  read all five. Getting to zero took the five findings listed under
  "Corpus eligibility" — two real libmelee_ex fixes (final-frame flush,
  re-simulation boundary) and three harness normalizations grounded in
  byte-level verification, including one case where peppi fabricates
  and misassigns data over a Slippi write-gap and libmelee_ex is
  demonstrably the more faithful parser.
  """

  use ExUnit.Case, async: false

  alias ExPhil.Data.Parity

  @moduletag :slow
  # A few thousand replays takes a while; the default sample is small.
  @moduletag timeout: :infinity

  @sample String.to_integer(System.get_env("PARITY_SAMPLE") || "25")
  @seed String.to_integer(System.get_env("PARITY_SEED") || "20260805")

  describe "Melee.Events vs peppi" do
    test "agree field-by-field across a sampled replay corpus" do
      paths = Parity.corpus()

      if paths == [] do
        flunk("no replay corpus found; set PARITY_CORPUS")
      end

      # Screen out pre-2.2.0 replays up front (see @moduledoc "Corpus
      # eligibility") so the sample size means what it says.
      sample =
        paths
        |> Parity.sample(length(paths), @seed)
        |> Stream.filter(&Parity.comparable?/1)
        |> Enum.take(@sample)

      {checked, skipped, divergence} =
        Enum.reduce_while(sample, {0, [], nil}, fn path, {ok, skips, _} ->
          case Parity.check_file(path) do
            :ok -> {:cont, {ok + 1, skips, nil}}
            {:skip, reason} -> {:cont, {ok, [{path, reason} | skips], nil}}
            {:divergence, d} -> {:halt, {ok, skips, d}}
          end
        end)

      if divergence do
        flunk("""
        Melee.Events and peppi disagree.

          file:  #{divergence.file}
          frame: #{inspect(divergence.frame)}
          port:  #{inspect(divergence.port)}
          field: #{divergence.field}
          melee: #{inspect(divergence.melee)}
          peppi: #{inspect(divergence.peppi)}

        (sample=#{@sample} seed=#{@seed}; re-run with the same PARITY_SEED
        to reproduce, or PARITY_CORPUS=#{divergence.file} to isolate.)
        """)
      end

      # Some replays are genuinely unreadable by peppi (truncated bot eval
      # recordings). That is allowed, but must stay a small minority —
      # otherwise the test would pass vacuously.
      assert checked > 0, "every sampled replay was skipped: #{inspect(skipped)}"

      assert length(skipped) <= div(length(sample), 4),
             "too many skipped replays (#{length(skipped)}/#{length(sample)}): " <>
               inspect(Enum.take(skipped, 10))
    end

    test "agree on PRE-2.2.0 replays via Melee.SlpFile's manual bookends" do
      # These files have no FRAME_BOOKEND, so the live codec yields
      # nothing for them; check_file routes them through Melee.SlpFile,
      # which completes each frame off the next frame's pre-frame event
      # (and flushes the final frame at GAME_END — a real one-frame-short
      # bug this very differential found when first extended, fixed in
      # libmelee_ex 83bcc6a). Fields the file's replay version predates
      # (jumps, speeds, hitlag, on_ground — see
      # Parity.post_frame_len/1) are excluded: both parsers invent
      # different defaults for absent fields, which is a defaults
      # disagreement, not a decoding one.
      paths = Parity.corpus()

      if paths == [] do
        flunk("no replay corpus found; set PARITY_CORPUS")
      end

      sample =
        paths
        |> Parity.sample(length(paths), @seed)
        |> Stream.reject(&Parity.comparable?/1)
        |> Enum.take(@sample)

      if sample == [] do
        IO.puts("\n[parity] no pre-2.2.0 replays in the corpus; nothing to compare")
      else
        {checked, skipped, divergence} =
          Enum.reduce_while(sample, {0, [], nil}, fn path, {ok, skips, _} ->
            case Parity.check_file(path) do
              :ok -> {:cont, {ok + 1, skips, nil}}
              {:skip, reason} -> {:cont, {ok, [{path, reason} | skips], nil}}
              {:divergence, d} -> {:halt, {ok, skips, d}}
            end
          end)

        if divergence do
          flunk("""
          Melee.SlpFile (manual bookends) and peppi disagree on a pre-2.2.0 replay.

            file:  #{divergence.file}
            frame: #{inspect(divergence.frame)}
            port:  #{inspect(divergence.port)}
            field: #{divergence.field}
            melee: #{inspect(divergence.melee)}
            peppi: #{inspect(divergence.peppi)}

          (sample=#{@sample} seed=#{@seed}; re-run with the same PARITY_SEED
          to reproduce, or PARITY_CORPUS=#{divergence.file} to isolate.)
          """)
        end

        assert checked > 0, "every sampled old replay was skipped: #{inspect(skipped)}"

        assert length(skipped) <= div(length(sample), 4),
               "too many skipped old replays (#{length(skipped)}/#{length(sample)}): " <>
                 inspect(Enum.take(skipped, 10))
      end
    end
  end

  describe "the harness itself" do
    @fixture "test/fixtures/replays/fox_multishine.slp"

    test "both parsers actually see the same frames" do
      {:ok, raw} = Parity.raw_stream(@fixture)
      {:ok, melee} = Parity.melee_frames(raw)
      {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)

      assert length(melee) == length(replay.frames)
      assert hd(melee).frame == hd(replay.frames).frame_number
      assert List.last(melee).frame == List.last(replay.frames).frame_number
      # Not a vacuous corpus: a real game's worth of frames.
      assert length(melee) > 1000
    end

    test "compares a non-trivial number of fields per player-frame" do
      {:ok, raw} = Parity.raw_stream(@fixture)
      {:ok, [gs | _]} = Parity.melee_frames(raw)
      {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)
      pf = hd(replay.frames)

      port = gs.players |> Map.keys() |> hd()
      triples = Parity.field_triples(gs.players[port], pf.players[port])

      assert length(triples) >= 34
      assert Enum.all?(triples, fn {name, _, _} -> is_atom(name) end)
    end

    @roy_replay "replays/huggingface/20_50_14 [INFP] Peach + Roy (YS).slp"

    test "pins the FIXED NIF character-id mapping (Roy -> 0x1A, no more -1)" do
      # History: the NIF used to feed post.character (an INTERNAL id)
      # through the external-id table — identity below 0x1A, but Roy and
      # above fell to -1 (found by this differential 2026-08-05). Fixed
      # 2026-08-13: internal_character_id/1 is pinned as identity. This
      # test used to pin the HOLE so it couldn't widen unnoticed; it now
      # pins the fix so it cannot regress.
      if File.exists?(@roy_replay) do
        {:ok, raw} = Parity.raw_stream(@roy_replay)
        {:ok, [_, _, gs | _]} = Parity.melee_frames(raw, skip_rollback_frames: false)
        {:ok, replay} = ExPhil.Data.Peppi.parse(@roy_replay)
        pf = Enum.at(replay.frames, 2)

        roy_port =
          Enum.find(Map.keys(gs.players), fn port -> gs.players[port].character == 0x1A end)

        assert roy_port, "fixture no longer contains a Roy"

        # Both sides now report the internal id, in agreement.
        assert gs.players[roy_port].character == 0x1A
        assert pf.players[roy_port].character == 0x1A
      end
    end

    test "detects an injected divergence" do
      {:ok, raw} = Parity.raw_stream(@fixture)
      {:ok, melee} = Parity.melee_frames(raw)
      {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)

      # Corrupt one player's percent on one frame; the comparison must
      # name that exact frame/port/field rather than merely counting.
      [target | rest] = Enum.drop(replay.frames, 10)
      port = target.players |> Map.keys() |> hd()

      bad_player = %{target.players[port] | percent: target.players[port].percent + 5.0}
      bad = %{target | players: Map.put(target.players, port, bad_player)}
      frames = Enum.take(replay.frames, 10) ++ [bad | rest]

      assert {:divergence, d} = Parity.compare(@fixture, melee, frames)
      assert d.field == :percent
      assert d.frame == target.frame_number
      assert d.port == port
      assert_in_delta d.peppi - d.melee, 5.0, 1.0e-6
    end
  end
end
