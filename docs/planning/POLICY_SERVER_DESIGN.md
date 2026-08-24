# Resident policy server — design (JIT_WARMUP option 3)

**STATUS 2026-08-24: BUILT + SMOKE-VALIDATED (M1-M3).** Session C
checked out the preloaded agent at **warmup 0ms** and played a full
clean game through it (2,457 frames, staleness 1 — remote inference
holds 60fps). Three design corrections from the smokes, all committed:
(1) per-beam JIT identity does NOT amortize across Agent instances
(fresh Axon.build closures = new cache identity; measured 19.2s for a
second Agent) — the amortizer is WARM-AGENT POOLING (release parks,
checkout reuses via Agent.reconfigure, structural-vs-tunable opts
split); (2) distribution must be on FROM VM BOOT (mid-run Node.start
strands pre-rename pids — the first session died in EXLA's cache);
(3) session cleanup RELEASES server-owned agents, never stops them.
Launch: see scripts/policy_server.exs header + --policy-server in
play_dolphin_async. Remaining: eval-harness adoption (M3's fleet
half), checkpoint hot-swap ergonomics (M4).

2026-08-24. The last structural JIT lever: one long-lived beam holds
the JIT'd policy and serves inference to every session; games and
dolphins come and go, the compile happens once per server boot.

## Why this shape wins

- **JIT cost amortizes perfectly in-beam.** EXLA's compiled-executable
  cache is per-beam (persistent_term keyed by fn+shapes), so the
  SECOND Agent instance in the same beam warms in ~0s. A server beam
  can hold one Agent per session — stateful streams stay isolated —
  while every Agent after the first is free.
- **Zero API change.** AsyncRunner talks to the Agent through exactly
  two GenServer calls (`reset_buffer/1`,
  `get_controller_with_confidence/3`) plus boot-time
  `warmup`/`warmed_up?`. A remote `{name, node}` ref drops in.
  Payload = the game_state struct (~KBs of terms); local-loopback
  dist call RTT is ~100-200µs against a 16.7ms frame budget and
  1-9ms inference.
- **Session boot collapses to menu time.** No warmup at all in the
  session (the server pre-warmed): local launch ≈ 3-4s, netplay ≈
  post-JIT flow (~2.5s + connect). Beats even the stateful path's
  1.5s, for the WINDOWED deploy configuration — no behavior questions.
- **Eval fleets are the second customer.** Gate sweeps / deciders /
  promotion rungs run dozens-to-hundreds of session starts at 20s
  JIT each today. Against the server: zero per-start compile, and
  N arms share the GPU through one client (no fraction juggling).

## Architecture

```
policy-server beam (named node, e.g. exphil_policy@127.0.0.1)
  ExPhil.PolicyServer (GenServer, globally named)
    checkout(policy_path, agent_opts) -> {:ok, agent_ref}
      - starts (or reuses a pooled) Agent under a DynamicSupervisor
      - first Agent for a (checkpoint, config) pays the JIT; later
        ones warm instantly (shared in-beam executable cache)
    release(agent_ref)          - stops/reclaims the session's Agent
    preload(policy_path, opts)  - warm a checkpoint ahead of sessions
    status()                    - loaded checkpoints, live agents,
                                  warmup states

session beam (play_dolphin_async / eval harnesses)
  --policy-server flag:
    Node.start(:"session_<pid>@127.0.0.1"), Node.connect(server)
    agent = PolicyServer.checkout(policy, opts)   # remote ref
    AsyncRunner unchanged — calls the remote ref
```

## Hazards and their answers

1. **The second-EXLA-client law** ("a second EXLA client SIGBUS/
   SIGSEGVs any live exphil beam"): the session beam must NEVER
   initialize a CUDA client. In server mode the session runs no
   Nx/EXLA work at all (embedding+inference live server-side).
   Belt-and-braces: `--policy-server` sets the session env CPU-only
   (`EXLA_CPU_ONLY=1`) so an accidental Nx call cannot create a GPU
   client. The play script's delay-id guard etc. run server-side at
   checkout (the server loads the metadata).
2. **The no-mix law now covers the server.** A running policy server
   is a live exphil beam: `mix` anywhere in the shared-EXLA repos can
   SIGBUS it. Same standing rule as training runs; `status()` +
   `pgrep -f policy_server` join the pre-mix checklist.
3. **Staleness across code changes**: the server runs OLD code after
   a recompile. Mitigation: the server prints its git rev at boot;
   sessions log a rev-mismatch warning at checkout (config carries
   the rev). Restarting the server costs one 20s warmup.
4. **Crash isolation**: a session crash must not take the server
   down (checkout monitors the caller and auto-releases); a server
   crash fails sessions loudly (AsyncRunner already treats a dead
   agent as fatal). No transparent failover — sessions rerun.
5. **Distribution security**: nodes bind loopback only
   (`inet_dist_use_interface {127,0,0,1}`), random cookie written to
   a user-only file that sessions read.

## Delivery plan

- M1: `ExPhil.PolicyServer` + `scripts/policy_server.exs` (boot,
  preload champion, log rev + warmup time, loop). Unit: checkout
  bookkeeping with a stub agent module.
- M2: `--policy-server` in play_dolphin_async (node start, connect,
  checkout, CPU-only env guard). Smoke: local FD game with the
  session's warmup line absent and the server's agent serving.
- M3: second-session instant-warmup demonstration (the amortization
  claim, measured) + eval-harness adoption (gate sweeps point at the
  server).
- M4 (later): checkpoint hot-swap via preload+checkout-new, pool
  reuse policies, remote status in `analyze_menu_time`.

## Non-goals (v1)

Cross-machine serving; transparent failover; batching multiple
sessions into one inference call (each Agent stays a stateful stream;
GPU sharing via CUDA is enough at 2-6 arms).
