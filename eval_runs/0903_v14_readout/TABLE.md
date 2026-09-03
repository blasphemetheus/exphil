=== T=1.0 health bar 2026-09-03T04:56:13-05:00
T10 rc=0
=== T=0.5 control 2026-09-03T05:06:32-05:00
T05 rc=0
=== durations
T10: n=4 mean=115.5s reaching_cap=2/4
T05: n=4 mean=105.8s reaching_cap=3/4
=== loop_report
==> exla
Using libexla.so from /home/blewf/.cache/xla/exla/elixir-1.18.4-erts-16.4.0.4-xla-0.10.0-exla-0.13.1-2gwjr5st6sq2h5oph3af6rzta4/libexla.so
EXLA_CPU_ONLY is not set, checking for nvcc availability
CUDA is available.
make: '/home/blewf/git/exphil/_build/dev/lib/exla/priv/libexla.so' is up to date.
==> exphil
      warning: function blind_target_css/1 is unused
      │
 1676 │   defp blind_target_css(state) do
      │        ~
      │
      └─ lib/exphil_bridge/melee_port.ex:1676:8: ExPhil.Bridge.MeleePort (module)

     warning: clauses with the same name and arity (number of arguments) should be grouped together, "def handle_call/3" was previously defined (lib/exphil/agents/agent.ex:526)
     │
 809 │   def handle_call({:reconfigure, opts}, _from, state) do
     │       ~
     │
     └─ lib/exphil/agents/agent.ex:809:7


05:16:13.224 [info] [ExPhil] Application started

05:16:13.226 [debug] [ExPhil] Supervision tree: [{ExPhil.Training.AsyncCheckpoint, #PID<0.285.0>, :worker, [ExPhil.Training.AsyncCheckpoint]}, {ExPhil.Telemetry.Collector, #PID<0.284.0>, :worker, [ExPhil.Telemetry.Collector]}, {ExPhil.Agents.Supervisor, #PID<0.283.0>, :supervisor, [ExPhil.Agents.Supervisor]}, {ExPhil.Bridge.Supervisor, #PID<0.282.0>, :supervisor, [ExPhil.Bridge.Supervisor]}, {ExPhil.Registry, #PID<0.280.0>, :supervisor, [Registry]}]

╔════════════════════════════════════════════════════════════╗
║                    Loop / Taunt Report                     ║
╚════════════════════════════════════════════════════════════╝

[1mConfiguration:[0m
  [2m  Replays:[0m 8
  [2m  Bot port:[0m 1
  [2m  Groups:[0m 2

Scoring [[32m████[0m[2m░░░░░░░░░░░░░░░░░░░░░░░░░░[0m] 13% (1/8)[KScoring [[32m████████[0m[2m░░░░░░░░░░░░░░░░░░░░░░[0m] 25% (2/8)[KScoring [[32m███████████[0m[2m░░░░░░░░░░░░░░░░░░░[0m] 38% (3/8)[KScoring [[32m███████████████[0m[2m░░░░░░░░░░░░░░░[0m] 50% (4/8)[KScoring [[32m███████████████████[0m[2m░░░░░░░░░░░[0m] 63% (5/8)[KScoring [[32m███████████████████████[0m[2m░░░░░░░[0m] 75% (6/8)[KScoring [[32m██████████████████████████[0m[2m░░░░[0m] 88% (7/8)[KScoring [[32m██████████████████████████████[0m[2m[0m] 100% (8/8)[K

| arm | scored/played | taunts/min | d_up press/min | frozen-input frac | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|---|---|
| T05 | 4/4 | 2.03~1.97 [0.98-3.93] | 101.16~104.71 [93.69-112.39] | 0.00~0.01 [0.00-0.01] | 0.23~0.24 [0.15-0.29] | 0.49~0.49 [0.00-1.47] | 2.25~4 [0-5] |
| T10 | 4/4 | 2.05~2.46 [0.62-2.65] | 410.05~407.45 [399.45-427.14] | 0.00~0.00 [0.00-0.00] | 0.31~0.35 [0.20-0.41] | 0.16~0.00 [0.00-0.62] | 0.75~0 [0-3] |

[05:16:13] [33m⚠️  mean~median [min-max]. Standing law: differences under 2x are UNRESOLVED, and a range that spans the other arm's mean is no difference at all. A mean far from its median = one degenerate game is dragging the group.[0m

| arm | game | min | taunts/min | d_up/min | frozen | held | loops/min | max reps |
|---|---|---|---|---|---|---|---|---|
| T05 | r1.slp | 2.04 | 0.98 | 93.69 | 0.01 | 0.15 | 0.49 | 4 |
| T05 | r2.slp | 2.04 | 3.93 | 112.39 | 0.01 | 0.23 | 0.00 | 0 |
| T05 | r3.slp | 2.03 | 1.97 | 93.86 | 0.00 | 0.24 | 1.47 | 5 |
| T05 | r4.slp | 0.80 | 1.25 | 104.71 | 0.00 | 0.29 | 0.00 | 0 |
| T10 | r1.slp | 2.03 | 2.46 | 427.14 | 0.00 | 0.27 | 0.00 | 0 |
| T10 | r2.slp | 2.04 | 2.46 | 399.45 | 0.00 | 0.41 | 0.00 | 0 |
| T10 | r3.slp | 1.61 | 0.62 | 407.45 | 0.00 | 0.35 | 0.62 | 3 |
| T10 | r4.slp | 1.88 | 2.65 | 406.13 | 0.00 | 0.20 | 0.00 | 0 |


Most repeated action cycles (action-state ids):

| action cycle | episodes |
|---|---|
| `GRAB_WAIT>GRAB_PUMMEL` | 2 |
| `DAMAGE_NEUTRAL_2>LANDING` | 1 |
| `SHIELD_START>SPOTDODGE` | 1 |
| `GRAB_PUMMEL>GRAB_WAIT` | 1 |

[05:16:13] [32m✓ Wrote eval_runs/0903_v14_readout/loops/report.md and report.json[0m
=== drill measure (60 eps, v1.4) 2026-09-03T05:16:13-05:00
==> exla
Using libexla.so from /home/blewf/.cache/xla/exla/elixir-1.18.4-erts-16.4.0.4-xla-0.10.0-exla-0.13.1-2gwjr5st6sq2h5oph3af6rzta4/libexla.so
EXLA_CPU_ONLY is not set, checking for nvcc availability
CUDA is available.
make: '/home/blewf/git/exphil/_build/dev/lib/exla/priv/libexla.so' is up to date.
==> exphil
      warning: function blind_target_css/1 is unused
      │
 1676 │   defp blind_target_css(state) do
      │        ~
      │
      └─ lib/exphil_bridge/melee_port.ex:1676:8: ExPhil.Bridge.MeleePort (module)

     warning: clauses with the same name and arity (number of arguments) should be grouped together, "def handle_call/3" was previously defined (lib/exphil/agents/agent.ex:526)
     │
 809 │   def handle_call({:reconfigure, opts}, _from, state) do
     │       ~
     │
     └─ lib/exphil/agents/agent.ex:809:7


05:16:14.196 [info] [ExPhil] Application started

05:16:14.198 [debug] [ExPhil] Supervision tree: [{ExPhil.Training.AsyncCheckpoint, #PID<0.285.0>, :worker, [ExPhil.Training.AsyncCheckpoint]}, {ExPhil.Telemetry.Collector, #PID<0.284.0>, :worker, [ExPhil.Telemetry.Collector]}, {ExPhil.Agents.Supervisor, #PID<0.283.0>, :supervisor, [ExPhil.Agents.Supervisor]}, {ExPhil.Bridge.Supervisor, #PID<0.282.0>, :supervisor, [ExPhil.Bridge.Supervisor]}, {ExPhil.Registry, #PID<0.280.0>, :supervisor, [Registry]}]

╔════════════════════════════════════════════════════════════╗
║             Hit-confirm drill — uthrow_low_mid             ║
╚════════════════════════════════════════════════════════════╝

[1mConfiguration:[0m
  [2m  Policy:[0m "checkpoints/fox_gen_v1.4_long_20260903_072152_best_policy.bin"
  [2m  Drill:[0m "drills/uthrow_low_mid.json"
  [2m  Episodes:[0m 60
  [2m  Window:[0m 240
  [2m  Bands:[0m "0-19%, 20-39%"
  [2m  Dummy:[0m "fox"
  [2m  Warm context:[0m true
  [2m  Out:[0m "/home/blewf/git/exphil/eval_runs/0903_v14_readout/drill"


05:16:14.683 [warning] [Checkpoint] checkpoints/fox_gen_v1.4_long_20260903_072152_best_policy.bin metadata contains atoms not interned in this VM (likely an external spec arch); falling back to unrestricted binary_to_term. Only load checkpoints from trusted sources.

05:16:14.684 [info] [Checkpoint] Loaded 11.3 MB from checkpoints/fox_gen_v1.4_long_20260903_072152_best_policy.bin

05:16:14.684 [warning] [Agent] Autoregressive controller head ACTIVE (sequential per-frame sampling)

05:16:14.684 [info] [Agent] Loading temporal policy TRUNK (backbone: gru, window: 60, AR head)

05:16:14.922 [info] [Agent] Starting JIT warmup...

05:16:14.928 [info] [Agent] warmup stage embed: 6ms

05:16:34.929 [info] [Agent] warmup stage sample1 (predict+heads compile): 19998ms

05:16:34.936 [info] [Agent] warmup stage sample2 (prev-buttons variant): 7ms

05:16:34.936 [info] [Agent] warmup stage confidence: 0ms

05:16:34.936 [info] [Agent] JIT warmup complete (20014ms)
[05:16:34] [32m✓ Agent warmed up (20014ms)[0m
[05:16:34]   agent: use_prev_action=nil temporal=true

05:16:35.499 [info] [MeleePort] Frame 0: menu_state=255 | P1:? P2:?

05:16:35.541 [info] [MeleePort] Frame 0: menu_state=5 | P1:? P2:?

05:16:35.543 [info] [MeleePort] MEM1 located (emulator 894060, base 0x2300000000) — selection words now read fresh per frame

05:16:35.652 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:16:35.730 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:16:35.860 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:16:36.088 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:16:36.909 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:16:37.710 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:16:38.520 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:16:39.293 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:16:40.261 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:16:41.338 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:16:42.257 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:21%/4stk
[05:16:42]   ep 1 [0-19%]: hits=3 dmg=21.5  (ref 3.9/27.0)  [game ep 1, handoff f98]

05:16:43.088 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:21%/4stk

05:16:43.937 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:21%/4stk

05:16:44.835 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:28%/4stk

05:16:45.773 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:28%/4stk

05:16:46.697 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:28%/4stk

05:16:47.567 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:28%/4stk
[05:16:47]   ep 2 [20-39%]: hits=2 dmg=7.0  (ref 3.7/20.5)  [game ep 2, handoff f432]

05:16:48.401 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:28%/4stk

05:16:49.357 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:35%/4stk

05:16:50.252 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:35%/4stk

05:16:51.237 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:39%/4stk

05:16:52.200 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:39%/4stk
[05:16:52]   ep 3 [20-39%]: hits=2 dmg=11.0  (ref 3.7/20.5)  [game ep 3, handoff f763]

05:16:53.099 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:39%/4stk

05:16:54.072 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:39%/4stk

05:16:55.006 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:49%/4stk

05:16:55.963 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:51%/4stk

05:16:56.954 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:51%/4stk

05:16:57.942 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:51%/4stk
[05:16:58]   ep 4 [20-39%]: hits=2 dmg=11.8  (ref 3.7/20.5)  [game ep 4, handoff f1087]

05:16:58.166 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:51%/4stk

05:16:58.498 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:51%/3stk

05:16:59.364 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:17:00.230 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:17:01.063 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:17:01.944 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:2%/3stk

05:17:02.832 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:17:03.690 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:17:04.539 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:7%/3stk
[05:17:05]   ep 5 [0-19%]: hits=2 dmg=7.0  (ref 3.9/27.0)  [game ep 5, handoff f1673]

05:17:05.432 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:17:06.350 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:9%/3stk

05:17:07.258 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:17:08.110 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:17:08.990 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:13%/3stk
[05:17:09]   ep 6 [0-19%]: hits=2 dmg=6.0  (ref 3.9/27.0)  [game ep 6, handoff f1972]

05:17:09.833 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:17:10.689 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:17:11.659 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:19%/3stk

05:17:12.564 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:17:13.416 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:17:14.315 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:24%/3stk
[05:17:14]   ep 7 [0-19%]: hits=1 dmg=10.5  (ref 3.9/27.0)  [game ep 7, handoff f2305]

05:17:15.162 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:17:16.047 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:29%/3stk

05:17:16.948 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:29%/3stk

05:17:17.792 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:29%/3stk

05:17:18.646 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:34%/3stk
[05:17:18]   ep 8 [20-39%]: hits=2 dmg=10.4  (ref 3.7/20.5)  [game ep 8, handoff f2592]

05:17:19.496 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:34%/3stk

05:17:20.347 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:34%/3stk

05:17:21.217 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:40%/3stk

05:17:22.069 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:40%/3stk

05:17:22.904 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:40%/3stk

05:17:23.753 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:40%/3stk
[05:17:24]   ep 9 [20-39%]: hits=2 dmg=5.9  (ref 3.7/20.5)  [game ep 9, handoff f2966]

05:17:24.576 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:40%/3stk

05:17:25.457 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:45%/3stk

05:17:26.370 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:45%/3stk

05:17:27.318 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:50%/3stk

05:17:28.236 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:50%/3stk
[05:17:28]   ep 10 [20-39%]: hits=2 dmg=10.0  (ref 3.7/20.5)  [game ep 10, handoff f3253]

05:17:28.524 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:50%/3stk

05:17:29.041 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:50%/2stk

05:17:30.001 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:17:30.890 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:17:31.812 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:17:32.677 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:17:33.520 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:17:34.355 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:17:35.216 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:6%/2stk
[05:17:35]   ep 11 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 11, handoff f3819]

05:17:36.053 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:17:36.917 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:11%/2stk

05:17:37.771 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:11%/2stk

05:17:38.619 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:11%/2stk

05:17:39.481 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:11%/2stk
[05:17:40]   ep 12 [0-19%]: hits=2 dmg=5.5  (ref 3.9/27.0)  [game ep 12, handoff f4117]

05:17:40.439 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:11%/2stk

05:17:41.329 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:11%/2stk

05:17:42.249 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:17%/2stk

05:17:43.199 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:18%/2stk

05:17:44.208 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:18%/2stk
[05:17:45]   ep 13 [0-19%]: hits=2 dmg=6.3  (ref 3.9/27.0)  [game ep 13, handoff f4439]

05:17:45.238 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:18%/2stk

05:17:46.265 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:18%/2stk

05:17:47.328 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:35%/2stk

05:17:48.276 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:40%/2stk

05:17:49.260 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:54%/2stk
[05:17:50]   ep 14 [0-19%]: hits=2 dmg=36.4 STOCK  (ref 3.9/27.0)  [game ep 14, handoff f4736]

05:17:50.155 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:54%/1stk

05:17:51.017 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:0%/1stk

05:17:51.847 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:0%/1stk

05:17:52.699 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:0%/1stk

05:17:53.577 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:0%/1stk

05:17:54.498 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:7%/1stk

05:17:55.580 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:12%/1stk

05:17:56.563 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:12%/1stk
[05:17:57]   ep 15 [0-19%]: hits=2 dmg=12.5  (ref 3.9/27.0)  [game ep 15, handoff f5220]

05:17:57.410 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/3stk P2:12%/1stk

05:17:58.279 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/3stk P2:12%/1stk

05:17:59.179 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/3stk P2:19%/1stk

05:18:00.039 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/3stk P2:19%/1stk

05:18:00.980 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/3stk P2:19%/1stk

05:18:01.929 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/3stk P2:19%/1stk
[05:18:02]   ep 16 [0-19%]: hits=2 dmg=6.9  (ref 3.9/27.0)  [game ep 16, handoff f5528]

05:18:02.856 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/3stk P2:21%/1stk

05:18:03.844 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:04.990 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:06.265 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/3stk P2:31%/1stk
[05:18:07]   ep 17 [0-19%]: hits=2 dmg=12.0  (ref 3.9/27.0)  [game ep 17, handoff f5815]

05:18:07.159 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:08.002 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:08.845 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:09.685 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:10.529 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:11.372 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:12.218 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:13.083 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:13.911 [info] [MeleePort] Frame 6540: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:14.752 [info] [MeleePort] Frame 6600: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:15.587 [info] [MeleePort] Frame 6660: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:16.433 [info] [MeleePort] Frame 6720: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:17.277 [info] [MeleePort] Frame 6780: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:18.139 [info] [MeleePort] Frame 6840: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:18.985 [info] [MeleePort] Frame 6900: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:19.819 [info] [MeleePort] Frame 6960: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:20.671 [info] [MeleePort] Frame 7020: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:21.510 [info] [MeleePort] Frame 7080: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:22.344 [info] [MeleePort] Frame 7140: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:23.247 [info] [MeleePort] Frame 7200: menu_state=2 | P1:0%/3stk P2:31%/1stk
[05:18:24]     [reset] position timeout

05:18:24.076 [info] [MeleePort] Frame 7260: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:24.210 [info] [MeleePort] Frame 7320: menu_state=2 | P1:0%/3stk P2:31%/1stk

05:18:24.490 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:18:24.505 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:18:24.630 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[05:18:24]   -- game boundary (game 2 starting)

05:18:24.816 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:18:25.696 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:18:26.604 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:18:27.438 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:18:28.339 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:18:29.191 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:18:30.089 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:18:31.242 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:12%/4stk
[05:18:31]   ep 18 [0-19%]: hits=2 dmg=12.5  (ref 3.9/27.0)  [game ep 18, handoff f98]

05:18:32.219 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:12%/4stk

05:18:33.125 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:12%/4stk

05:18:34.019 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:18:34.872 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:18:35.801 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:18:36.989 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:19%/4stk
[05:18:37]   ep 19 [0-19%]: hits=2 dmg=13.9  (ref 3.9/27.0)  [game ep 19, handoff f450]

05:18:37.978 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:26%/4stk

05:18:38.899 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:26%/4stk

05:18:39.905 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:33%/4stk

05:18:40.937 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:34%/4stk

05:18:41.851 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:46%/4stk

05:18:42.860 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:46%/4stk
[05:18:43]   ep 20 [20-39%]: hits=3 dmg=19.5  (ref 3.7/20.5)  [game ep 20, handoff f823]

05:18:43.692 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:46%/4stk

05:18:43.820 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:46%/4stk

05:18:44.394 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:46%/3stk

05:18:45.418 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:18:46.450 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:18:47.329 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:18:48.219 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:18:49.102 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:11%/3stk

05:18:49.957 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:11%/3stk

05:18:50.802 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:11%/3stk
[05:18:51]   ep 21 [0-19%]: hits=2 dmg=11.3  (ref 3.9/27.0)  [game ep 21, handoff f1419]

05:18:51.815 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:11%/3stk

05:18:52.847 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:11%/3stk

05:18:53.853 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:18%/3stk

05:18:54.825 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:18%/3stk

05:18:55.693 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:18:56.556 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:24%/3stk
[05:18:57]   ep 22 [0-19%]: hits=2 dmg=13.1  (ref 3.9/27.0)  [game ep 22, handoff f1779]

05:18:57.657 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:18:58.528 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:18:59.382 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:24%/3stk

05:19:00.233 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:26%/3stk

05:19:01.108 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:35%/3stk

05:19:01.989 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:35%/3stk

05:19:02.880 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:35%/3stk
[05:19:03]   ep 23 [20-39%]: hits=2 dmg=11.0  (ref 3.7/20.5)  [game ep 23, handoff f2207]
[05:19:03]     [reset] percent out of band

05:19:03.859 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:42%/3stk

05:19:03.986 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:42%/3stk

05:19:04.114 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:42%/3stk

05:19:04.648 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:42%/2stk

05:19:05.494 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:19:06.369 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:19:07.241 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:19:08.105 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:19:08.963 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:19:09.826 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:19:10.725 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:22%/2stk
[05:19:11]   ep 24 [0-19%]: hits=3 dmg=24.5  (ref 3.9/27.0)  [game ep 24, handoff f2857]

05:19:11.626 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:25%/2stk

05:19:12.476 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:25%/2stk

05:19:13.315 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:25%/2stk

05:19:14.269 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:19:15.210 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:19:16.110 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:19:17.052 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:36%/2stk
[05:19:17]   ep 25 [20-39%]: hits=2 dmg=11.6  (ref 3.7/20.5)  [game ep 25, handoff f3246]

05:19:17.927 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:36%/2stk

05:19:18.825 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:47%/2stk

05:19:19.707 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:47%/2stk

05:19:20.635 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:47%/2stk

05:19:21.583 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:55%/2stk
[05:19:21]   ep 26 [20-39%]: hits=1 dmg=19.0  (ref 3.7/20.5)  [game ep 26, handoff f3544]

05:19:21.767 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:55%/2stk

05:19:21.897 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:55%/2stk

05:19:22.687 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:55%/1stk

05:19:23.592 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:19:24.472 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:19:25.404 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:19:26.279 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:7%/1stk

05:19:27.147 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:7%/1stk

05:19:28.008 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:7%/1stk

05:19:28.854 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:7%/1stk
[05:19:29]   ep 27 [0-19%]: hits=2 dmg=6.8  (ref 3.9/27.0)  [game ep 27, handoff f4161]

05:19:29.678 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:7%/1stk

05:19:30.522 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:7%/1stk

05:19:31.422 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:13%/1stk

05:19:32.257 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:27%/1stk

05:19:33.101 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:27%/1stk

05:19:33.946 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:27%/1stk
[05:19:34]   ep 28 [0-19%]: hits=2 dmg=33.0  (ref 3.9/27.0)  [game ep 28, handoff f4525]

05:19:34.791 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:40%/1stk

05:19:35.638 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:40%/1stk

05:19:36.547 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:46%/1stk

05:19:37.421 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:46%/1stk

05:19:38.266 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:51%/1stk

05:19:39.124 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:51%/1stk
[05:19:39]   ep 29 [20-39%]: hits=2 dmg=11.3  (ref 3.7/20.5)  [game ep 29, handoff f4886]

05:19:39.581 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:51%/1stk

05:19:39.888 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:19:39.901 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:19:40.023 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[05:19:40]   -- game boundary (game 3 starting)

05:19:40.209 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:19:41.048 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:19:41.944 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:19:42.808 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:19:43.682 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:44.527 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:45.374 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:46.217 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:7%/4stk
[05:19:46]   ep 30 [0-19%]: hits=2 dmg=7.5  (ref 3.9/27.0)  [game ep 30, handoff f98]

05:19:47.110 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:48.021 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:48.870 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:49.709 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:19:50.573 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:14%/4stk

05:19:51.442 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:14%/4stk

05:19:52.308 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:14%/4stk

05:19:53.183 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:14%/4stk
[05:19:53]   ep 31 [0-19%]: hits=2 dmg=6.8  (ref 3.9/27.0)  [game ep 31, handoff f554]

05:19:54.056 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:14%/4stk

05:19:54.928 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:20%/4stk

05:19:55.785 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:20%/4stk

05:19:56.618 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:20%/4stk

05:19:57.447 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:30%/4stk
[05:19:58]   ep 32 [0-19%]: hits=2 dmg=16.2  (ref 3.9/27.0)  [game ep 32, handoff f876]

05:19:58.359 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:19:59.221 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:20:00.077 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:20:00.968 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:20:01.948 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:36%/4stk

05:20:03.016 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:41%/4stk

05:20:04.066 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:41%/4stk

05:20:04.928 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:41%/4stk
[05:20:05]   ep 33 [20-39%]: hits=2 dmg=11.0  (ref 3.7/20.5)  [game ep 33, handoff f1333]

05:20:05.273 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:41%/4stk

05:20:05.749 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:41%/3stk

05:20:06.598 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:20:07.446 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:20:08.299 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:20:09.242 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:20:10.139 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:20:10.990 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:20:11.844 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:6%/3stk
[05:20:12]   ep 34 [0-19%]: hits=2 dmg=10.5  (ref 3.9/27.0)  [game ep 34, handoff f1901]

05:20:12.671 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:10%/3stk

05:20:13.552 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:16%/3stk

05:20:14.479 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:21%/3stk

05:20:15.331 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:21%/3stk

05:20:16.201 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:21%/3stk
[05:20:16]   ep 35 [0-19%]: hits=2 dmg=10.2  (ref 3.9/27.0)  [game ep 35, handoff f2189]

05:20:17.034 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:21%/3stk

05:20:17.906 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:27%/3stk

05:20:18.768 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:36%/3stk

05:20:19.701 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:36%/3stk

05:20:20.585 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:36%/3stk
[05:20:21]   ep 36 [20-39%]: hits=2 dmg=15.1  (ref 3.7/20.5)  [game ep 36, handoff f2496]

05:20:21.438 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:36%/3stk

05:20:22.325 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:40%/3stk

05:20:23.240 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:46%/3stk

05:20:24.198 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:46%/3stk

05:20:25.061 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:46%/3stk
[05:20:25]   ep 37 [20-39%]: hits=2 dmg=10.5  (ref 3.7/20.5)  [game ep 37, handoff f2804]

05:20:25.717 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:46%/3stk

05:20:25.848 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:46%/3stk

05:20:26.475 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:46%/2stk

05:20:27.306 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:20:28.146 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:20:28.999 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:20:29.870 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:20:30.718 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:20:31.549 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:20:32.385 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:7%/2stk
[05:20:32]   ep 38 [0-19%]: hits=2 dmg=7.1  (ref 3.9/27.0)  [game ep 38, handoff f3386]

05:20:33.235 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:20:34.111 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:13%/2stk

05:20:34.975 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:13%/2stk

05:20:35.891 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:13%/2stk

05:20:36.791 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:13%/2stk
[05:20:37]   ep 39 [0-19%]: hits=2 dmg=13.1  (ref 3.9/27.0)  [game ep 39, handoff f3698]

05:20:37.638 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:20%/2stk

05:20:38.479 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:20%/2stk

05:20:39.319 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:20%/2stk

05:20:40.163 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:20%/2stk

05:20:41.059 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:20%/2stk

05:20:41.994 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:26%/2stk

05:20:42.863 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:38%/2stk

05:20:43.710 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:45%/2stk

05:20:44.552 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:45%/2stk
[05:20:45]   ep 40 [20-39%]: hits=2 dmg=25.0  (ref 3.7/20.5)  [game ep 40, handoff f4240]

05:20:45.162 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:45%/2stk

05:20:45.290 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:45%/2stk

05:20:45.937 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:45%/1stk

05:20:46.785 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:20:47.632 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:20:48.484 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:20:49.370 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:6%/1stk

05:20:50.221 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:6%/1stk

05:20:51.061 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:6%/1stk

05:20:51.902 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:6%/1stk
[05:20:52]   ep 41 [0-19%]: hits=2 dmg=6.1  (ref 3.9/27.0)  [game ep 41, handoff f4826]

05:20:52.776 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:6%/1stk

05:20:53.686 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:12%/1stk

05:20:54.528 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:12%/1stk

05:20:55.392 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:12%/1stk

05:20:56.245 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:12%/1stk
[05:20:56]   ep 42 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 42, handoff f5117]

05:20:57.170 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:12%/1stk

05:20:58.195 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:20:59.461 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:00.743 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:01.961 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:23%/1stk
[05:21:02]   ep 43 [0-19%]: hits=2 dmg=11.2  (ref 3.9/27.0)  [game ep 43, handoff f5404]

05:21:02.827 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:03.679 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:04.522 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:05.446 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:06.308 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:07.183 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:08.046 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:08.900 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:09.737 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:10.601 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:11.452 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:12.290 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:13.160 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:14.005 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:14.838 [info] [MeleePort] Frame 6540: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:15.693 [info] [MeleePort] Frame 6600: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:16.568 [info] [MeleePort] Frame 6660: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:17.422 [info] [MeleePort] Frame 6720: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:18.323 [info] [MeleePort] Frame 6780: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:19.169 [info] [MeleePort] Frame 6840: menu_state=2 | P1:0%/4stk P2:23%/1stk
[05:21:19]     [reset] position timeout

05:21:19.382 [info] [MeleePort] Frame 6900: menu_state=2 | P1:0%/4stk P2:23%/1stk

05:21:19.675 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:21:19.688 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:21:19.812 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[05:21:19]   -- game boundary (game 4 starting)

05:21:19.995 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:21:20.854 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:21:21.750 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:21:22.630 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:21:23.521 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:21:24.369 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:21:25.216 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:21:26.066 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:7%/4stk
[05:21:26]   ep 44 [0-19%]: hits=2 dmg=7.5  (ref 3.9/27.0)  [game ep 44, handoff f98]

05:21:26.897 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:21:27.793 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:21:28.653 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:13%/4stk

05:21:29.511 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:21:30.359 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:21:31.277 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:19%/4stk
[05:21:31]   ep 45 [0-19%]: hits=1 dmg=11.8  (ref 3.9/27.0)  [game ep 45, handoff f464]

05:21:32.108 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:21:32.959 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:21:33.800 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:19%/4stk

05:21:34.708 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:21:35.570 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:21:36.498 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:30%/4stk
[05:21:37]   ep 46 [0-19%]: hits=2 dmg=11.0  (ref 3.9/27.0)  [game ep 46, handoff f838]

05:21:37.656 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:21:38.541 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:30%/4stk

05:21:39.469 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:36%/4stk

05:21:40.425 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:45%/4stk

05:21:41.331 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:45%/4stk

05:21:42.171 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:45%/4stk
[05:21:42]   ep 47 [20-39%]: hits=2 dmg=15.1  (ref 3.7/20.5)  [game ep 47, handoff f1142]

05:21:42.329 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:45%/4stk

05:21:42.712 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:45%/3stk

05:21:43.597 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:21:44.439 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:21:45.315 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:21:46.175 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:2%/3stk

05:21:47.053 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:6%/3stk

05:21:47.890 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:21:48.739 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:7%/3stk
[05:21:49]   ep 48 [0-19%]: hits=3 dmg=22.0  (ref 3.9/27.0)  [game ep 48, handoff f1730]

05:21:49.580 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:22%/3stk

05:21:50.431 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:22%/3stk

05:21:51.438 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:28%/3stk

05:21:52.406 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:29%/3stk

05:21:53.299 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:29%/3stk

05:21:54.277 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:36%/3stk
[05:21:54]   ep 49 [20-39%]: hits=3 dmg=14.4  (ref 3.7/20.5)  [game ep 49, handoff f2066]

05:21:55.294 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:36%/3stk

05:21:56.262 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:43%/3stk

05:21:57.177 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:43%/3stk

05:21:58.114 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:43%/3stk

05:21:59.079 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:43%/3stk
[05:21:59]   ep 50 [20-39%]: hits=2 dmg=6.6  (ref 3.7/20.5)  [game ep 50, handoff f2379]

05:21:59.728 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:43%/3stk

05:21:59.856 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:43%/3stk

05:22:00.317 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:43%/2stk

05:22:01.278 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:22:02.185 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:22:03.075 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:22:03.940 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:22:04.844 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:6%/2stk

05:22:05.731 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:22:06.586 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:7%/2stk
[05:22:07]   ep 51 [0-19%]: hits=2 dmg=7.1  (ref 3.9/27.0)  [game ep 51, handoff f2983]

05:22:07.430 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:22:08.316 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:7%/2stk

05:22:09.253 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:9%/2stk

05:22:10.232 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:22:11.194 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:22:12.122 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:31%/2stk
[05:22:13]   ep 52 [0-19%]: hits=1 dmg=24.1  (ref 3.9/27.0)  [game ep 52, handoff f3354]

05:22:13.098 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:22:14.057 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:31%/2stk

05:22:15.025 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:37%/2stk

05:22:15.920 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:37%/2stk

05:22:16.824 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:37%/2stk

05:22:17.662 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:37%/2stk
[05:22:18]   ep 53 [20-39%]: hits=2 dmg=6.0  (ref 3.7/20.5)  [game ep 53, handoff f3689]

05:22:18.525 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:37%/2stk

05:22:19.373 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:37%/2stk

05:22:20.256 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:43%/2stk

05:22:21.105 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:43%/2stk

05:22:21.956 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:43%/2stk
[05:22:22]   ep 54 [20-39%]: hits=2 dmg=5.7  (ref 3.7/20.5)  [game ep 54, handoff f4017]

05:22:22.800 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:43%/2stk

05:22:22.931 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:44%/2stk

05:22:23.154 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:44%/1stk

05:22:23.989 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:22:24.834 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:22:25.675 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:22:26.638 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:22:27.503 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:8%/1stk

05:22:28.429 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:9%/1stk

05:22:29.316 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:30.251 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:14%/1stk
[05:22:30]   ep 55 [0-19%]: hits=3 dmg=13.7  (ref 3.9/27.0)  [game ep 55, handoff f4622]

05:22:31.173 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:32.036 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:32.903 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:33.748 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:34.662 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:35.628 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:36.590 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:37.606 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:38.612 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:39.559 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:40.552 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:41.514 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:42.356 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:43.204 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:44.183 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:45.142 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:45.996 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:46.838 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:47.737 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:48.630 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:14%/1stk
[05:22:48]     [reset] position timeout

05:22:48.817 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:14%/1stk

05:22:49.111 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:22:49.124 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:22:49.249 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[05:22:49]   -- game boundary (game 5 starting)

05:22:49.397 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:22:50.265 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:22:51.114 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:22:52.017 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

05:22:52.984 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

05:22:53.861 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:8%/4stk

05:22:54.775 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:8%/4stk

05:22:55.633 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:8%/4stk
[05:22:56]   ep 56 [0-19%]: hits=2 dmg=8.5  (ref 3.9/27.0)  [game ep 56, handoff f98]

05:22:56.480 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:8%/4stk

05:22:57.366 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:15%/4stk

05:22:58.249 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:21%/4stk

05:22:59.120 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:26%/4stk

05:22:59.975 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:34%/4stk
[05:23:00]   ep 57 [0-19%]: hits=3 dmg=27.3  (ref 3.9/27.0)  [game ep 57, handoff f394]

05:23:00.812 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:38%/4stk

05:23:01.707 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:45%/4stk

05:23:02.563 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:45%/4stk

05:23:03.461 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:50%/4stk

05:23:04.335 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:50%/4stk
[05:23:04]   ep 58 [20-39%]: hits=2 dmg=18.1  (ref 3.7/20.5)  [game ep 58, handoff f696]

05:23:04.992 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:56%/4stk

05:23:05.165 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:56%/3stk

05:23:06.020 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:23:06.890 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:23:07.740 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:23:08.673 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:0%/3stk

05:23:09.564 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:23:10.421 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:23:11.274 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:23:12.130 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:7%/3stk
[05:23:12]   ep 59 [0-19%]: hits=2 dmg=7.0  (ref 3.9/27.0)  [game ep 59, handoff f1266]

05:23:12.962 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:23:13.847 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:7%/3stk

05:23:14.765 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:23:15.729 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:23:16.735 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:23:17.719 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:13%/3stk
[05:23:17]   ep 60 [0-19%]: hits=2 dmg=6.5  (ref 3.9/27.0)  [game ep 60, handoff f1624]

05:23:17.897 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:13%/3stk

05:23:18.029 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:13%/2stk

05:23:18.158 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.284 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.412 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.540 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.678 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.803 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:18.931 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:0%/2stk

05:23:19.059 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.181 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.312 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.434 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.557 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.688 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.814 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:19.948 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:0%/1stk

05:23:20.201 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

05:23:20.215 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

05:23:20.341 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[05:23:20]   -- game boundary (game 6 starting)
[05:23:20] [32m✓ console: 60 episodes across 5 games, 405.6s[0m
[05:23:20] 
[05:23:20] == smoke summary (live counter; authoritative = drill_score.exs on the bank)
[05:23:20]    0-19%: n=39  hits 2.1  >=3 13%  dmg 12.5   (expert n=39: 3.9 / 87% / 27.0)
[05:23:20]    20-39%: n=21  hits 2.0  >=3 10%  dmg 12.2   (expert n=24: 3.7 / 83% / 20.5)
[05:23:20] [32m✓ bank -> /home/blewf/git/exphil/eval_runs/0903_v14_readout/drill (/home/blewf/git/exphil/eval_runs/0903_v14_readout/drill/episodes.jsonl)[0m
[05:23:20] [33m⚠️  Killed 1 orphaned Dolphin(s)[0m
==> exla
Using libexla.so from /home/blewf/.cache/xla/exla/elixir-1.18.4-erts-16.4.0.4-xla-0.10.0-exla-0.13.1-2gwjr5st6sq2h5oph3af6rzta4/libexla.so
EXLA_CPU_ONLY is not set, checking for nvcc availability
CUDA is available.
make: '/home/blewf/git/exphil/_build/dev/lib/exla/priv/libexla.so' is up to date.
==> exphil
      warning: function blind_target_css/1 is unused
      │
 1676 │   defp blind_target_css(state) do
      │        ~
      │
      └─ lib/exphil_bridge/melee_port.ex:1676:8: ExPhil.Bridge.MeleePort (module)

     warning: clauses with the same name and arity (number of arguments) should be grouped together, "def handle_call/3" was previously defined (lib/exphil/agents/agent.ex:526)
     │
 809 │   def handle_call({:reconfigure, opts}, _from, state) do
     │       ~
     │
     └─ lib/exphil/agents/agent.ex:809:7


05:23:21.262 [info] [ExPhil] Application started

05:23:21.264 [debug] [ExPhil] Supervision tree: [{ExPhil.Training.AsyncCheckpoint, #PID<0.285.0>, :worker, [ExPhil.Training.AsyncCheckpoint]}, {ExPhil.Telemetry.Collector, #PID<0.284.0>, :worker, [ExPhil.Telemetry.Collector]}, {ExPhil.Agents.Supervisor, #PID<0.283.0>, :supervisor, [ExPhil.Agents.Supervisor]}, {ExPhil.Bridge.Supervisor, #PID<0.282.0>, :supervisor, [ExPhil.Bridge.Supervisor]}, {ExPhil.Registry, #PID<0.280.0>, :supervisor, [Registry]}]

╔════════════════════════════════════════════════════════════╗
║                     Drill bank scorer                      ║
╚════════════════════════════════════════════════════════════╝

[05:23:21]   60 episodes across 5 replays

# Drill bank score — eval_runs/0903_v14_readout/drill (uthrow_low_mid)

60 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 30/60.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot 0-19% | 39 | 1.9 | 18 | 12.5 | 1 |
| expert 0-19% | 39 | 3.9 | 87 | 27.0 | 0 |
| bot 20-39% | 21 | 2.1 | 29 | 12.2 | 0 |
| expert 20-39% | 24 | 3.7 | 83 | 20.5 | 0 |


0-19% hits histogram: 1:14  2:18  3:4  4:2  6:1


20-39% hits histogram: 1:5  2:10  3:4  4:2


[05:23:21] [32m✓ wrote eval_runs/0903_v14_readout/drill/RESULTS.md[0m
V14 READOUT DONE 2026-09-03T05:23:21-05:00
