#!/usr/bin/env bash
# NEW-ERA rounds 13+ (newera8): first HEADLESS-PARALLEL probe loop.
#
# Succeeds overnight_newera7_20260716.sh. Pool = 48 games (newera7's 36 +
# the 12 r11/r12 probe replays). New in this template:
#   - probe_fanout: 2 fan-outs of 3 headless ExiAI games (plain vs
#     --jump-debounce A/B preserved), private per-arm replay dirs =
#     attribution (no ~/Slippi mtime scans), EXI inputs carry analog
#     (GOTCHA #66 RESOLVED 2026-07-17 — shield gates cross-build
#     comparable again)
#   - liveness detector: an arm with no .slp after LIVENESS_SECS is killed
#     (the 2026-07-17 lesson: a 1s crash hid inside 900s timeouts twice);
#     ALL arms early-failing = ENVIRONMENTAL WIPEOUT, exit 3 — explicitly
#     NOT a 0-conversion policy verdict
#   - #67 airtight: ONE mix compile at t=0, fan-outs run --no-compile;
#     nothing may relink the shared exla NIF while probe BEAMs live
#   - per-phase TIMING lines + rounds/day estimate (#20)
#   - report card is x/10 from r15 on (gate 10 armed approaches; x/9
#     r13-r14 with gate 9 shield breaks; x/8 thru r12)
#   NEWERA8_CONVERSION_WEIGHT=  --conversion-weight (r15 pool sampling)
#   NEWERA8_PROBE_REG= / NEWERA8_PROBE_REG_EVERY=  --probe-reg[-every] (r15)
#
# Env knobs:
#   NEWERA8_ROUNDS=2       rounds to run (r13..)
#   NEWERA8_BACKBONE=      bake-off: backbone for dagger_drill --backbone;
#                          checkpoint names gain _<backbone> when set
#   NEWERA8_TAG=           screen-run isolation (e.g. screen_min_gru):
#                          checkpoint NAME_SUFFIX becomes _<tag> (replacing
#                          the backbone suffix), log + probe dirs gain the
#                          tag — screen runs never collide with the r-line
#   NEWERA8_MAX_EPOCHS=100 dagger_drill --max-epochs (30 = screening budget)
#   NEWERA8_DRILL_FLAGS=   extra dagger_drill flags, appended VERBATIM
#                          (word-split; e.g. "--hidden-size 512")
#   NEWERA8_SCENARIOS=1    run scenario_suite per round (eval phase)
#   NEWERA8_SMOKE=1        guards + one 2-arm fan-out vs r12, then exit
#   PROBE_MODE=headless|windowed   windowed = newera7's sequential probe
#                          (calibration; needs launcher CLOSED, #62)
#   PROBE_TIMEOUT / LIVENESS_SECS / PROBE_MEMFRAC / NEWERA8_DROPOUT
set -uo pipefail

if [ -z "${NEWERA8_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export NEWERA8_INHIBITED=1
  exec systemd-inhibit --what=sleep --who="newera8" \
    --why="new-era DAgger loop in progress" "$0" "$@"
fi
cd "$(dirname "$0")/.."

TAG=${NEWERA8_TAG:-}
LOG=logs/overnight_newera8${TAG:+_$TAG}_$(date +%Y%m%d).log
PROBE_BASE=probes/newera8${TAG:+/$TAG}
TRACE=scripts/trace_tech_chase.exs
ROUNDS=${NEWERA8_ROUNDS:-2}
FIRST_ROUND=13
BACKBONE=${NEWERA8_BACKBONE:-}
MAX_EPOCHS=${NEWERA8_MAX_EPOCHS:-100}
DRILL_FLAGS=${NEWERA8_DRILL_FLAGS:-}
# TAG wins the checkpoint suffix (screen tags already name the backbone);
# without a TAG the r-line's backbone-suffix behavior is unchanged.
if [ -n "$TAG" ]; then
  NAME_SUFFIX=_$TAG
else
  NAME_SUFFIX=${BACKBONE:+_$BACKBONE}
fi
PROBE_MODE=${PROBE_MODE:-headless}
# MAINLINE from r15 on (#66-RESOLUTION: ExiAI drops analog trigger pipes;
# mainline carries them fully). mamba_full/r13-r14 cards were scored on
# ExiAI — cross-engine card comparisons are invalid; the 3-game same-policy
# A/B (post-run queue) anchors the translation. Override to the old ExiAI
# wrapper via NEWERA8_DOLPHIN for back-compat probes. Mainline nuance:
# one benign xcb window per instance; respawn timing differs slightly.
HEADLESS_DOLPHIN=${NEWERA8_DOLPHIN:-$HOME/.local/share/slippi/mainline/dolphin-emu-mainline}
NETPLAY_DOLPHIN=$HOME/.local/share/slippi/netplay
ISO=$HOME/isos/melee.iso
PROBE_TIMEOUT=${PROBE_TIMEOUT:-720}
# --stateful-step for probes (default on): at 60Hz pacing with 3 arms
# sharing the GPU, windowed-style inference runs ~30/s = acting every
# OTHER frame; the O(1) step path (0.21ms) restores every-frame acting.
# Set PROBE_STATEFUL=0 for non-GRU/LSTM backbones (bake-off).
PROBE_STATEFUL=${PROBE_STATEFUL:-1}
LIVENESS_SECS=${LIVENESS_SECS:-180}
MEMFRAC=${PROBE_MEMFRAC:-0.15}
DROPOUT=${NEWERA8_DROPOUT:-0.4}
BASE_PORT=51500

mkdir -p logs "$PROBE_BASE"

# ---------------------------------------------------------------- guards
# #67's hazard is beams sharing OUR exla NIF — i.e. beams running out of
# the nx-consumer repos. Unrelated apps (2026-07-21: Bradley's Phoenix
# server in ~/git/shine) must not block training; match by /proc cwd.
exla_beams() {
  local p
  for p in $(pgrep -x beam.smp); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in
      "$HOME/git/exphil"* | "$HOME/git/edifice"* | "$HOME/git/nx"*) echo "$p" ;;
    esac
  done
}
if [ -n "$(exla_beams)" ]; then
  echo "[newera8] EXLA-sharing BEAM ALREADY LIVE ($(exla_beams | tr '\n' ' ')) — refusing to start (#67)" | tee -a "$LOG"
  exit 1
fi
assert_no_beam() {
  [ -n "$(exla_beams)" ] && {
    echo "[newera8] EXLA-sharing BEAM alive before $1 — refusing (#67)" | tee -a "$LOG"
    exit 1
  }
}
for tool in make cargo; do
  command -v "$tool" >/dev/null || {
    echo "[newera8] $tool not on PATH — run me from 'devenv shell' (#60)" | tee -a "$LOG"
    exit 1
  }
done

if [ "$PROBE_MODE" = headless ]; then
  [ -x "$HEADLESS_DOLPHIN" ] || {
    echo "[newera8] headless wrapper missing: $HEADLESS_DOLPHIN (GOTCHAS #64)" | tee -a "$LOG"
    exit 1
  }
  # #64: pinned nix store paths in the wrapper can be GC'd by a system
  # update. Check the store ROOTS only (/nix/store/<hash>-<name>) — the
  # wrapper lines carry shell fragments after the path.
  for sp in $(grep -oE '/nix/store/[^/ :"]+' "$HEADLESS_DOLPHIN" | sort -u); do
    [ -e "$sp" ] || {
      echo "[newera8] wrapper store path GC'd: $sp — re-run the nix build from GOTCHAS #64" | tee -a "$LOG"
      exit 1
    }
  done
  case "$HEADLESS_DOLPHIN" in
    *exi-ai*)
      echo "[newera8] probes: headless ExiAI, PIPE inputs — analog triggers dropped (#66:" | tee -a "$LOG"
      echo "[newera8] policy analog-shoulder head is a no-op; shield gates not comparable" | tee -a "$LOG"
      echo "[newera8] with r1-r12 windowed history; EXI mode stays OPT-IN until its" | tee -a "$LOG"
      echo "[newera8] analog-release latch is fixed — see #66 addendum)" | tee -a "$LOG"
      ;;
    *)
      echo "[newera8] probes: headless MAINLINE (#66-RESOLUTION) — analog pipes intact;" | tee -a "$LOG"
      echo "[newera8] cards NOT comparable with ExiAI-scored rounds (thru mamba_full)" | tee -a "$LOG"
      ;;
  esac
else
  echo "[newera8] probes: WINDOWED calibration mode — keep the Slippi launcher CLOSED (#62)" | tee -a "$LOG"
fi

# Last compile of the run (#67): fan-outs use --no-compile below.
echo "[newera8] precompiling (the ONLY mix compile of this run)..." | tee -a "$LOG"
mix compile 2>&1 | tail -2 >>"$LOG"

# ------------------------------------------------------------------ pool
# Explicit paths only, never size-globbed (#58). 36 from newera7 + 12
# r11/r12 probes. newera8's own probe replays append by private-dir path.
POOL=(
  "$HOME/Slippi/Game_20260714T115716.slp"
  "$HOME/Slippi/Game_20260714T142354.slp"
  "$HOME/Slippi/Game_20260714T170156.slp"
  "$HOME/Slippi/Game_20260714T195416.slp"
  "$HOME/Slippi/Game_20260714T200239.slp"
  "$HOME/Slippi/Game_20260714T201103.slp"
  "$HOME/Slippi/Game_20260714T213744.slp"
  "$HOME/Slippi/Game_20260714T214601.slp"
  "$HOME/Slippi/Game_20260714T215417.slp"
  "$HOME/Slippi/Game_20260714T234927.slp"
  "$HOME/Slippi/Game_20260714T235750.slp"
  "$HOME/Slippi/Game_20260715T000607.slp"
  "$HOME/Slippi/Game_20260715T023011.slp"
  "$HOME/Slippi/Game_20260715T023834.slp"
  "$HOME/Slippi/Game_20260715T024658.slp"
  "$HOME/Slippi/Game_20260715T050120.slp"
  "$HOME/Slippi/Game_20260715T050944.slp"
  "$HOME/Slippi/Game_20260715T051807.slp"
  "$HOME/Slippi/Game_20260715T074615.slp"
  "$HOME/Slippi/Game_20260715T075438.slp"
  "$HOME/Slippi/Game_20260715T080302.slp"
  "$HOME/Slippi/Game_20260715T115334.slp"
  "$HOME/Slippi/Game_20260715T120157.slp"
  "$HOME/Slippi/Game_20260715T121021.slp"
  "$HOME/Slippi/Game_20260715T194348.slp"
  "$HOME/Slippi/Game_20260716T020330.slp"
  "$HOME/Slippi/Game_20260716T021147.slp"
  "$HOME/Slippi/Game_20260716T022012.slp"
  "$HOME/Slippi/Game_20260716T022829.slp"
  "$HOME/Slippi/Game_20260716T023646.slp"
  "$HOME/Slippi/Game_20260716T024504.slp"
  "$HOME/Slippi/Game_20260716T065841.slp"
  "$HOME/Slippi/Game_20260716T070659.slp"
  "$HOME/Slippi/Game_20260716T071523.slp"
  "$HOME/Slippi/Game_20260716T071931.slp"
  "$HOME/Slippi/Game_20260716T072614.slp"
  "$HOME/Slippi/Game_20260717T034738.slp"
  "$HOME/Slippi/Game_20260717T035556.slp"
  "$HOME/Slippi/Game_20260717T040414.slp"
  "$HOME/Slippi/Game_20260717T041239.slp"
  "$HOME/Slippi/Game_20260717T042103.slp"
  "$HOME/Slippi/Game_20260717T042921.slp"
  "$HOME/Slippi/Game_20260717T082059.slp"
  "$HOME/Slippi/Game_20260717T082923.slp"
  "$HOME/Slippi/Game_20260717T083741.slp"
  "$HOME/Slippi/Game_20260717T084349.slp"
  "$HOME/Slippi/Game_20260717T085207.slp"
  "$HOME/Slippi/Game_20260717T090032.slp"
  # r13 probe replays (private-dir paths): r14 trained on the 48 above +
  # these 6 = the 54-game pool. Screen runs (TAG set, fresh r13-with-tag
  # checkpoints) train on the same diet r14 saw. When the r-line resumes
  # at r15, append r14's 6 probe replays here too.
  "probes/newera8/r13/plain/p1/Game_20260718T002318.slp"
  "probes/newera8/r13/plain/p2/Game_20260718T002330.slp"
  "probes/newera8/r13/plain/p3/Game_20260718T002343.slp"
  "probes/newera8/r13/debounce/p1/Game_20260718T003155.slp"
  "probes/newera8/r13/debounce/p2/Game_20260718T003208.slp"
  "probes/newera8/r13/debounce/p3/Game_20260718T003220.slp"
)

# NEWERA8_EXTRA_ROLLOUTS: comma-separated extra replays appended to the
# pool (e.g. human-vs-bot demo games — the richest DAgger rollouts we get)
if [ -n "${NEWERA8_EXTRA_ROLLOUTS:-}" ]; then
  IFS=',' read -ra _EXTRA <<< "$NEWERA8_EXTRA_ROLLOUTS"
  for _r in "${_EXTRA[@]}"; do
    [ -f "$_r" ] || { echo "[newera8] FATAL: extra rollout missing: $_r"; exit 1; }
    POOL+=("$_r")
  done
  echo "[newera8] pool += ${#_EXTRA[@]} extra rollouts (NEWERA8_EXTRA_ROLLOUTS)"
fi

# --------------------------------------------------------------- timing
T_TRAIN=0 T_PROBE=0 T_EVAL=0 ROUNDS_DONE=0
phase_begin() { PHASE_T0=$SECONDS; }
phase_end() {
  local d=$((SECONDS - PHASE_T0))
  echo "[newera8] TIMING round=$2 phase=$1 secs=$d build=$PROBE_MODE" | tee -a "$LOG"
  case $1 in
    train) T_TRAIN=$((T_TRAIN + d)) ;;
    probe) T_PROBE=$((T_PROBE + d)) ;;
    eval) T_EVAL=$((T_EVAL + d)) ;;
  esac
}

sweep_strays() {
  # #58/#63: exact-PID kills, never bare pkill -f
  for pid in $(pgrep -f '/tmp/[l]ibmelee_' 2>/dev/null); do
    [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
  done
}

# --------------------------------------------------------- probe_fanout
# probe_fanout TAG POLICY N OUT_DIR [extra play flags...]
# Feedback globals: FANOUT_REPLAYS (private-dir .slp paths), FANOUT_OK,
# FANOUT_EARLY. Return: 0 => >=1 usable replay; 1 => partial failures;
# 2 => ENVIRONMENTAL WIPEOUT (all arms early-failed, nothing scoreable).
# Scoring (trace/report-card) is the CALLER's job — those are mix runs and
# must wait until every probe BEAM is reaped (#67).
probe_fanout() {
  local tag=$1 policy=$2 n=$3 out=$4
  shift 4
  local extra=("$@")
  FANOUT_REPLAYS=()
  FANOUT_OK=0
  FANOUT_EARLY=0

  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    mv "$out" "$out.old.$(date +%H%M%S)"
    echo "[newera8] $tag: stale probe dir moved aside" | tee -a "$LOG"
  fi
  mkdir -p "$out"
  sweep_strays
  sleep 1

  local -a pids start early done_flag
  local i
  for i in $(seq 1 "$n"); do
    mkdir -p "$out/p$i"
    echo "[newera8] $tag p$i launching (port $((BASE_PORT + i)))..." | tee -a "$LOG"
    # Three memory guards (2026-07-17: arms 1-2 ballooned to the 0.75 dev
    # default and starved arm 3 to 50MB — EXLA_MEMORY_FRACTION was dead
    # config until the runtime.exs client fix; XLA_PYTHON_CLIENT_* are the
    # PJRT-level backstops, shell-set per GOTCHA #37):
    EXLA_MEMORY_FRACTION=$MEMFRAC \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC \
      timeout "$PROBE_TIMEOUT" \
      mix run --no-compile --no-deps-check scripts/play_dolphin_async.exs \
      --policy "$policy" \
      --dolphin "$PROBE_DOLPHIN" --iso "$ISO" \
      --character mewtwo --stage final_destination \
      --dummy tech_random --dummy-cpu-level 0 \
      --press-threshold 0.45 --release-threshold 0.3 \
      --no-audio \
      --deterministic --on-game-end stop \
      $HEADLESS_FLAG ${STATEFUL_FLAG:-} --slippi-port $((BASE_PORT + i)) \
      --replay-dir "$out/p$i" \
      "${extra[@]}" >"$out/p$i.log" 2>&1 &
    pids[i]=$!
    start[i]=$SECONDS
    early[i]=0
    done_flag[i]=0
    # 12s stagger: at 5s, arm 3's EXLA init raced arms 1-2's JIT compile
    # and died in the NIF ("unknown exception thrown within NIF" at
    # from_buffers, both arms' p3, 2026-07-17 validation)
    sleep 12
  done

  # Active wait + liveness: judge arms by OUTPUT APPEARING, not exit codes
  while :; do
    local alive=0
    for i in $(seq 1 "$n"); do
      [ "${done_flag[i]}" = 1 ] && continue
      if kill -0 "${pids[i]}" 2>/dev/null; then
        alive=1
        if [ $((SECONDS - start[i])) -gt "$LIVENESS_SECS" ] &&
          ! compgen -G "$out/p$i/*.slp" >/dev/null; then
          echo "[newera8] $tag p$i: no replay after ${LIVENESS_SECS}s — killing (early-fail)" | tee -a "$LOG"
          kill "${pids[i]}" 2>/dev/null
          early[i]=1
        fi
      else
        done_flag[i]=1
      fi
    done
    [ "$alive" = 0 ] && break
    sleep 10
  done

  for i in $(seq 1 "$n"); do
    wait "${pids[i]}" 2>/dev/null
    local slp
    # LARGEST, not newest: at real-time pacing the post-game window is
    # long enough for auto-menu to boot a second game before stop lands —
    # its stub .slp is newer than the completed game (2026-07-17).
    slp=$(ls -S "$out/p$i"/*.slp 2>/dev/null | head -1)
    if [ -n "$slp" ]; then
      local size
      size=$(stat -c %s "$slp")
      if [ "$size" -lt 512000 ]; then
        echo "[newera8] $tag p$i: garbage-size replay ($size bytes, #58) — discarded" | tee -a "$LOG"
      else
        # teardown segfaults after clean games are expected (#64) —
        # replay presence IS success
        FANOUT_REPLAYS+=("$slp")
        FANOUT_OK=$((FANOUT_OK + 1))
        echo "[newera8] $tag p$i replay: $slp" | tee -a "$LOG"
        continue
      fi
    fi
    if [ "${early[i]}" = 1 ]; then
      FANOUT_EARLY=$((FANOUT_EARLY + 1))
    else
      echo "[newera8] $tag p$i: no usable replay after full runtime — see $out/p$i.log" | tee -a "$LOG"
    fi
  done

  sweep_strays

  if [ "$FANOUT_OK" -gt 0 ]; then
    [ "$FANOUT_OK" -eq "$n" ] && return 0 || return 1
  fi

  # Zero usable replays — environmental regardless of HOW the arms died
  # (liveness-killed idlers OR instant crashes; the 2026-07-17 incident
  # was a 1s crash idling behind a 900s timeout).
  echo "[newera8] $tag: ENVIRONMENTAL WIPEOUT — all $n arms produced no replay" | tee -a "$LOG"
  echo "[newera8] (early-killed: $FANOUT_EARLY, crashed/other: $((n - FANOUT_EARLY)))." | tee -a "$LOG"
  echo "[newera8] NOT a 0-conversion policy result. Suspects: checkpoint" | tee -a "$LOG"
  echo "[newera8] deserialization (#68), wrapper store paths (#64), EXLA init." | tee -a "$LOG"
  echo "[newera8] First log: $out/p1.log" | tee -a "$LOG"
  return 2
}

if [ "$PROBE_MODE" = headless ]; then
  PROBE_DOLPHIN=$HEADLESS_DOLPHIN
  HEADLESS_FLAG=--headless
else
  PROBE_DOLPHIN=$NETPLAY_DOLPHIN
  HEADLESS_FLAG=
fi
STATEFUL_FLAG=
[ "$PROBE_STATEFUL" = 1 ] && STATEFUL_FLAG=--stateful-step

# ---------------------------------------------------------------- smoke
if [ -n "${NEWERA8_SMOKE:-}" ]; then
  SMOKE_POLICY=${SMOKE_POLICY:-checkpoints/mewtwo_combo_newera_r12_policy.bin}
  echo "[newera8] SMOKE: 2-arm fan-out vs $SMOKE_POLICY" | tee -a "$LOG"
  phase_begin
  probe_fanout smoke "$SMOKE_POLICY" 2 "$PROBE_BASE/smoke"
  rc=$?
  phase_end probe smoke
  if [ "$rc" = 2 ]; then exit 3; fi
  assert_no_beam smoke-scoring
  mix run --no-compile "$TRACE" "${FANOUT_REPLAYS[@]}" 2>/dev/null | tee -a "$LOG" | grep -oE 'conversions=[0-9]+'
  mix run --no-compile scripts/report_card.exs "${FANOUT_REPLAYS[@]}" 2>/dev/null | tee -a "$LOG" | grep 'SCORE:'
  echo "[newera8] SMOKE done (rc=$rc)" | tee -a "$LOG"
  exit 0
fi

# ------------------------------------------------------------ round loop
BEST_AVG10=-1
BEST_ROUND=""

for R in $(seq "$FIRST_ROUND" $((FIRST_ROUND + ROUNDS - 1))); do
  POLICY=checkpoints/mewtwo_combo_newera${NAME_SUFFIX}_r${R}_policy.bin
  ROLLOUTS=$(IFS=,; echo "${POOL[*]}")
  echo "[newera8] === round $R: ${#POOL[@]} replays, dropout $DROPOUT, backbone ${BACKBONE:-gru(default)} -> $POLICY ===" | tee -a "$LOG"

  phase_begin
  if [ -f "$POLICY" ]; then
    echo "[newera8] round $R: checkpoint exists — skipping training (resume mode)" | tee -a "$LOG"
  else
    RESUME_FLAG=
    if [ -f "$POLICY.trainer.ckpt" ]; then
      echo "[newera8] round $R: trainer snapshot found — resuming interrupted training" | tee -a "$LOG"
      RESUME_FLAG=--resume
    fi
    assert_no_beam "round $R training"
    # $DRILL_FLAGS deliberately unquoted: NEWERA8_DRILL_FLAGS is appended
    # verbatim (word-split) for bake-off shape knobs like --hidden-size 512
    mix run scripts/dagger_drill.exs \
      --expert mewtwo_combo \
      --max-epochs "$MAX_EPOCHS" \
      --prev-action-dropout "$DROPOUT" \
      --transition-weight 2.0 \
      ${BACKBONE:+--backbone "$BACKBONE"} \
      ${NEWERA8_CONVERSION_WEIGHT:+--conversion-weight "$NEWERA8_CONVERSION_WEIGHT"} \
      ${NEWERA8_PROBE_REG:+--probe-reg "$NEWERA8_PROBE_REG"} \
      ${NEWERA8_PROBE_REG_EVERY:+--probe-reg-every "$NEWERA8_PROBE_REG_EVERY"} \
      ${NEWERA8_PROBE_EVAL:+--probe-eval-every "$NEWERA8_PROBE_EVAL"} \
      ${NEWERA8_MAMBA_CHUNK:+--mamba-chunk-size "$NEWERA8_MAMBA_CHUNK"} \
      ${NEWERA8_MAMBA_MATMUL:+--mamba-matmul-scan} \
      $DRILL_FLAGS \
      $RESUME_FLAG \
      --rollouts "$ROLLOUTS" \
      --out "$POLICY" >>"$LOG" 2>&1
  fi
  if [ ! -f "$POLICY" ]; then
    echo "[newera8] round $R produced no checkpoint — stopping" | tee -a "$LOG"
    exit 1
  fi
  phase_end train "$R"

  phase_begin
  ROUND_CONV=0 ROUND_SCORE=0 ROUND_CARDS=0
  ROUND_REPLAYS=()

  probe_fanout "r${R}_plain" "$POLICY" 3 "$PROBE_BASE/r$R/plain"
  rc1=$?
  if [ "$rc1" = 2 ]; then
    echo "[newera8] aborting: environment dead (plain arm wiped)" | tee -a "$LOG"
    exit 3
  fi
  ROUND_REPLAYS+=("${FANOUT_REPLAYS[@]}")

  probe_fanout "r${R}_debounce" "$POLICY" 3 "$PROBE_BASE/r$R/debounce" --jump-debounce 10
  rc2=$?
  if [ "$rc2" = 2 ]; then
    echo "[newera8] WARNING: debounce arm wiped while plain arm lived —" | tee -a "$LOG"
    echo "[newera8] suspect the --jump-debounce path; continuing on plain data" | tee -a "$LOG"
  else
    ROUND_REPLAYS+=("${FANOUT_REPLAYS[@]}")
  fi

  # Scoring: all probe BEAMs are reaped; single sequential mix runs (#67)
  assert_no_beam "round $R scoring"
  if [ ${#ROUND_REPLAYS[@]} -gt 0 ]; then
    TRACE_OUT=$(mix run --no-compile "$TRACE" "${ROUND_REPLAYS[@]}" 2>/dev/null | tee -a "$LOG")
    # pure-bash sum: bc is not on the devenv PATH (found when validation 6
    # showed 9 traced conversions but ROUND_CONV=0)
    while read -r c; do
      ROUND_CONV=$((ROUND_CONV + c))
    done < <(grep -oE 'conversions=[0-9]+' <<<"$TRACE_OUT" | cut -d= -f2)
    CARD_OUT=$(mix run --no-compile scripts/report_card.exs "${ROUND_REPLAYS[@]}" 2>/dev/null | tee -a "$LOG")
    while read -r s; do
      ROUND_SCORE=$((ROUND_SCORE + s))
      ROUND_CARDS=$((ROUND_CARDS + 1))
    done < <(grep -oE 'SCORE: [0-9]+' <<<"$CARD_OUT" | grep -oE '[0-9]+$')
    POOL+=("${ROUND_REPLAYS[@]}")
  fi
  phase_end probe "$R"

  if [ -n "${NEWERA8_SCENARIOS:-}" ]; then
    phase_begin
    assert_no_beam "round $R scenarios"
    EXLA_MEMORY_FRACTION=0.15 timeout 1800 \
      mix run --no-compile scripts/scenario_suite.exs \
      --policy "$POLICY" --out "logs/newera8_r${R}_scenarios.json" >>"$LOG" 2>&1 || true
    echo "[newera8] round $R scenarios -> logs/newera8_r${R}_scenarios.json" | tee -a "$LOG"
    phase_end eval "$R"
  fi

  if [ "$ROUND_CARDS" -gt 0 ]; then
    AVG10=$((ROUND_SCORE * 10 / ROUND_CARDS))
    echo "[newera8] round $R report-card avg: $((AVG10 / 10)).$((AVG10 % 10))/10 over $ROUND_CARDS games build=$PROBE_MODE" | tee -a "$LOG"
    if [ "$AVG10" -gt "$BEST_AVG10" ]; then
      BEST_AVG10=$AVG10
      BEST_ROUND=$R
    fi
  else
    echo "[newera8] round $R: no scoreable games (mixed failures) — treating as" | tee -a "$LOG"
    echo "[newera8] ENVIRONMENTAL, not a policy verdict. Stopping." | tee -a "$LOG"
    exit 3
  fi

  ROUNDS_DONE=$((ROUNDS_DONE + 1))

  if [ "$ROUND_CONV" -eq 0 ]; then
    echo "[newera8] round $R: 0 conversions across $ROUND_CARDS real games — POLICY gate, stopping" | tee -a "$LOG"
    break
  fi
  echo "[newera8] round $R total conversions: $ROUND_CONV build=$PROBE_MODE" | tee -a "$LOG"
done

# ------------------------------------------------------------- post-loop
if [ -n "$BEST_ROUND" ]; then
  ln -sfn "mewtwo_combo_newera${NAME_SUFFIX}_r${BEST_ROUND}_policy.bin" \
    "checkpoints/mewtwo_combo_newera${NAME_SUFFIX}_best_policy.bin"
  echo "[newera8] best round by report card: r$BEST_ROUND ($((BEST_AVG10 / 10)).$((BEST_AVG10 % 10))/10 avg, build=$PROBE_MODE) -> checkpoints/mewtwo_combo_newera${NAME_SUFFIX}_best_policy.bin" | tee -a "$LOG"
fi

assert_no_beam interp
echo "[newera8] running interp_p3_case3..." | tee -a "$LOG"
mix run --no-compile scripts/interp_p3_case3.exs >>"$LOG" 2>&1 || true
sleep 3
sweep_strays

TOTAL=$((T_TRAIN + T_PROBE + T_EVAL))
echo "[newera8] TIMING TOTAL train=${T_TRAIN}s probe=${T_PROBE}s eval=${T_EVAL}s rounds=$ROUNDS_DONE" | tee -a "$LOG"
if [ "$TOTAL" -gt 0 ] && [ "$ROUNDS_DONE" -gt 0 ]; then
  RPD10=$((864000 * ROUNDS_DONE / TOTAL))
  echo "[newera8] estimated rounds/day: $((RPD10 / 10)).$((RPD10 % 10))" | tee -a "$LOG"
fi
echo "[newera8] done" | tee -a "$LOG"
