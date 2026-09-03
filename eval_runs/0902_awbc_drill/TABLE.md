=== AWBC drill arm start 2026-09-02T18:22:31-05:00 resume=checkpoints/fox_gen_v1.3_AR_20260901_194518_epoch3.axon epochs=2 mix=drills/drill1_hitconfirm.frames x20
=== train end 2026-09-02T22:55:46-05:00 rc=0
policy: checkpoints/fox_gen_v1.3_AWBCdrill_20260902_232232_policy.bin
=== drill re-measure start 2026-09-02T22:55:46-05:00
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


22:55:47.288 [info] [ExPhil] Application started

22:55:47.290 [debug] [ExPhil] Supervision tree: [{ExPhil.Training.AsyncCheckpoint, #PID<0.285.0>, :worker, [ExPhil.Training.AsyncCheckpoint]}, {ExPhil.Telemetry.Collector, #PID<0.284.0>, :worker, [ExPhil.Telemetry.Collector]}, {ExPhil.Agents.Supervisor, #PID<0.283.0>, :supervisor, [ExPhil.Agents.Supervisor]}, {ExPhil.Bridge.Supervisor, #PID<0.282.0>, :supervisor, [ExPhil.Bridge.Supervisor]}, {ExPhil.Registry, #PID<0.280.0>, :supervisor, [Registry]}]

╔════════════════════════════════════════════════════════════╗
║             Hit-confirm drill — uthrow_low_mid             ║
╚════════════════════════════════════════════════════════════╝

[1mConfiguration:[0m
  [2m  Policy:[0m "checkpoints/fox_gen_v1.3_AWBCdrill_20260902_232232_policy.bin"
  [2m  Drill:[0m "drills/uthrow_low_mid.json"
  [2m  Episodes:[0m 150
  [2m  Window:[0m 240
  [2m  Bands:[0m "0-19%, 20-39%"
  [2m  Dummy:[0m "fox"
  [2m  Warm context:[0m true
  [2m  Out:[0m "/home/blewf/git/exphil/eval_runs/0902_awbc_drill/drill_remeasure"


22:55:47.865 [warning] [Checkpoint] checkpoints/fox_gen_v1.3_AWBCdrill_20260902_232232_policy.bin metadata contains atoms not interned in this VM (likely an external spec arch); falling back to unrestricted binary_to_term. Only load checkpoints from trusted sources.

22:55:47.867 [info] [Checkpoint] Loaded 11.3 MB from checkpoints/fox_gen_v1.3_AWBCdrill_20260902_232232_policy.bin

22:55:47.867 [warning] [Agent] Autoregressive controller head ACTIVE (sequential per-frame sampling)

22:55:47.867 [info] [Agent] Loading temporal policy TRUNK (backbone: gru, window: 60, AR head)

22:55:48.146 [info] [Agent] Starting JIT warmup...

22:55:48.154 [info] [Agent] warmup stage embed: 8ms

22:56:09.550 [info] [Agent] warmup stage sample1 (predict+heads compile): 21392ms

22:56:09.557 [info] [Agent] warmup stage sample2 (prev-buttons variant): 7ms

22:56:09.557 [info] [Agent] warmup stage confidence: 0ms

22:56:09.557 [info] [Agent] JIT warmup complete (21411ms)
[22:56:09] [32m✓ Agent warmed up (21411ms)[0m
[22:56:09]   agent: use_prev_action=nil temporal=true

22:56:10.109 [info] [MeleePort] Frame 0: menu_state=255 | P1:? P2:?

22:56:10.159 [info] [MeleePort] Frame 0: menu_state=5 | P1:? P2:?

22:56:10.161 [info] [MeleePort] MEM1 located (emulator 196123, base 0x2300000000) — selection words now read fresh per frame

22:56:10.281 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

22:56:10.347 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

22:56:10.477 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk

22:56:10.723 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:56:11.682 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:56:12.680 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:56:13.571 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:56:14.498 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

22:56:15.457 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:16.337 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:17.422 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:20%/4stk
[22:56:18]   ep 1 [0-19%]: hits=2 dmg=20.4  (ref 3.9/27.0)  [game ep 1, handoff f98]

22:56:18.658 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:19.802 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:20.948 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:22.028 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:56:23.149 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:24.351 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:25.833 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:27.027 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:27%/4stk
[22:56:27]   ep 2 [20-39%]: hits=2 dmg=6.9  (ref 3.7/20.5)  [game ep 2, handoff f570]

22:56:28.073 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:29.100 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:30.101 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:27%/4stk

22:56:31.311 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:34%/4stk

22:56:32.447 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:34%/4stk

22:56:33.509 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:34%/4stk

22:56:34.694 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:34%/4stk
[22:56:35]   ep 3 [20-39%]: hits=2 dmg=6.3  (ref 3.7/20.5)  [game ep 3, handoff f993]

22:56:35.732 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:34%/4stk

22:56:36.826 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:34%/4stk

22:56:38.009 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:39.178 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:40.323 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:39%/4stk
[22:56:41]   ep 4 [20-39%]: hits=2 dmg=5.8  (ref 3.7/20.5)  [game ep 4, handoff f1320]

22:56:41.365 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:42.470 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:43.589 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:44.623 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:56:45.827 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:45%/4stk

22:56:46.860 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:57%/4stk

22:56:47.900 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:57%/4stk

22:56:49.268 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:57%/4stk
[22:56:49]   ep 5 [20-39%]: hits=2 dmg=17.7  (ref 3.7/20.5)  [game ep 5, handoff f1760]

22:56:49.886 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:57%/4stk

22:56:50.105 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:57%/3stk

22:56:51.226 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:56:52.405 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:56:53.545 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:56:54.735 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:56:55.969 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:11%/3stk

22:56:57.067 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:56:58.316 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:56:59.657 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:17%/3stk
[22:56:59]   ep 6 [0-19%]: hits=2 dmg=17.1  (ref 3.9/27.0)  [game ep 6, handoff f2344]

22:57:00.904 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:57:02.053 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:57:03.058 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:57:04.092 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:17%/3stk

22:57:05.273 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:23%/3stk

22:57:06.384 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:23%/3stk

22:57:07.477 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:23%/3stk
[22:57:08]   ep 7 [0-19%]: hits=2 dmg=14.7  (ref 3.9/27.0)  [game ep 7, handoff f2816]

22:57:08.676 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:32%/3stk

22:57:09.793 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:32%/3stk

22:57:10.998 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:38%/3stk

22:57:12.033 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:57:13.048 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:57:14.117 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:47%/3stk
[22:57:14]   ep 8 [20-39%]: hits=2 dmg=14.9  (ref 3.7/20.5)  [game ep 8, handoff f3121]

22:57:14.263 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:57:14.655 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:47%/2stk

22:57:15.678 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:57:16.718 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:57:17.775 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:57:18.872 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:2%/2stk

22:57:19.943 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:9%/2stk

22:57:20.977 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:16%/2stk

22:57:22.094 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:16%/2stk
[22:57:23]   ep 9 [0-19%]: hits=2 dmg=21.1  (ref 3.9/27.0)  [game ep 9, handoff f3711]

22:57:23.182 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:21%/2stk

22:57:24.346 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:21%/2stk

22:57:25.452 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:28%/2stk

22:57:26.524 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:28%/2stk

22:57:27.603 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:35%/2stk

22:57:28.609 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:35%/2stk
[22:57:28]   ep 10 [20-39%]: hits=2 dmg=13.7  (ref 3.7/20.5)  [game ep 10, handoff f4028]

22:57:29.638 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:35%/2stk

22:57:30.676 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:35%/2stk

22:57:31.729 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:41%/2stk

22:57:32.801 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:41%/2stk

22:57:33.843 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:41%/2stk
[22:57:34]   ep 11 [20-39%]: hits=2 dmg=6.5  (ref 3.7/20.5)  [game ep 11, handoff f4378]

22:57:34.843 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:41%/2stk

22:57:34.971 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:41%/2stk

22:57:35.253 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:41%/1stk

22:57:36.276 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:0%/1stk

22:57:37.327 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:0%/1stk

22:57:38.337 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:0%/1stk

22:57:39.445 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:0%/1stk

22:57:40.515 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:6%/1stk

22:57:41.539 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:6%/1stk

22:57:42.886 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:6%/1stk

22:57:44.444 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:6%/1stk
[22:57:44]   ep 12 [0-19%]: hits=2 dmg=6.1  (ref 3.9/27.0)  [game ep 12, handoff f4981]

22:57:45.715 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:8%/1stk

22:57:46.914 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:48.041 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:49.156 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:12%/1stk
[22:57:50]   ep 13 [0-19%]: hits=2 dmg=5.7  (ref 3.9/27.0)  [game ep 13, handoff f5268]

22:57:50.245 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:51.307 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:52.381 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:53.514 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:12%/1stk

22:57:54.654 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:17%/1stk

22:57:55.752 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:21%/1stk

22:57:56.944 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:28%/1stk

22:57:58.094 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:28%/1stk
[22:57:58]   ep 14 [0-19%]: hits=2 dmg=16.5  (ref 3.9/27.0)  [game ep 14, handoff f5707]

22:57:59.148 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:28%/1stk

22:58:00.231 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:28%/1stk

22:58:01.236 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:28%/1stk

22:58:02.302 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/4stk P2:34%/1stk

22:58:03.291 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/4stk P2:34%/1stk

22:58:04.380 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/4stk P2:34%/1stk

22:58:05.523 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/4stk P2:34%/1stk
[22:58:06]   ep 15 [20-39%]: hits=2 dmg=11.2  (ref 3.7/20.5)  [game ep 15, handoff f6143]
[22:58:06]     [reset] percent out of band

22:58:06.138 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/4stk P2:42%/1stk

22:58:06.267 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/4stk P2:42%/1stk

22:58:06.532 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

22:58:06.545 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

22:58:06.672 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[22:58:06]   -- game boundary (game 2 starting)

22:58:06.863 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:58:07.830 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:58:08.835 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:58:09.876 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:58:11.020 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

22:58:12.221 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:58:13.322 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:58:14.355 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:20%/4stk
[22:58:15]   ep 16 [0-19%]: hits=2 dmg=20.4  (ref 3.9/27.0)  [game ep 16, handoff f98]

22:58:15.477 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:58:16.567 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:58:17.623 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:20%/4stk

22:58:18.752 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:58:20.082 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:58:21.728 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:39%/4stk
[22:58:23]   ep 17 [20-39%]: hits=1 dmg=18.9  (ref 3.7/20.5)  [game ep 17, handoff f480]

22:58:23.409 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:58:24.675 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:58:25.879 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:46%/4stk

22:58:27.072 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:46%/4stk

22:58:28.194 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:54%/4stk
[22:58:29]   ep 18 [20-39%]: hits=2 dmg=16.9  (ref 3.7/20.5)  [game ep 18, handoff f777]

22:58:29.253 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:56%/4stk

22:58:29.382 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:56%/4stk

22:58:29.905 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:56%/3stk

22:58:30.922 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:58:31.940 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:58:32.981 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:0%/3stk

22:58:34.102 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:5%/3stk

22:58:35.140 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:24%/3stk

22:58:36.158 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:27%/3stk

22:58:37.146 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:35%/3stk
[22:58:37]   ep 19 [0-19%]: hits=3 dmg=34.7  (ref 3.9/27.0)  [game ep 19, handoff f1364]

22:58:38.139 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:35%/3stk

22:58:39.142 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:37%/3stk

22:58:40.297 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:58:41.441 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:58:42.586 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:47%/3stk
[22:58:43]   ep 20 [20-39%]: hits=2 dmg=12.3  (ref 3.7/20.5)  [game ep 20, handoff f1672]

22:58:43.630 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:58:43.765 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:47%/3stk

22:58:44.784 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:47%/2stk

22:58:45.921 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:58:47.013 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:58:48.038 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/2stk

22:58:49.105 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:9%/2stk

22:58:50.154 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:16%/2stk

22:58:51.315 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:16%/2stk

22:58:52.401 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:18%/2stk
[22:58:52]   ep 21 [0-19%]: hits=2 dmg=18.3  (ref 3.9/27.0)  [game ep 21, handoff f2235]

22:58:53.427 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/3stk P2:18%/2stk

22:58:54.455 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/3stk P2:18%/2stk

22:58:55.463 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/3stk P2:18%/2stk

22:58:56.467 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/3stk P2:18%/2stk

22:58:57.517 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:58:58.517 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:58:59.536 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/3stk P2:38%/2stk
[22:59:00]   ep 22 [0-19%]: hits=1 dmg=19.5  (ref 3.9/27.0)  [game ep 22, handoff f2697]

22:59:00.571 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:59:01.661 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:59:02.732 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:59:03.775 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:38%/2stk

22:59:04.935 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:45%/2stk

22:59:06.262 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:45%/2stk

22:59:07.909 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:45%/2stk

22:59:09.489 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:45%/2stk
[22:59:09]   ep 23 [20-39%]: hits=2 dmg=6.9  (ref 3.7/20.5)  [game ep 23, handoff f3127]

22:59:09.773 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:45%/2stk

22:59:09.902 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:45%/2stk

22:59:10.590 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:45%/1stk

22:59:11.745 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:0%/1stk

22:59:12.815 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:0%/1stk

22:59:13.977 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:0%/1stk

22:59:15.133 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:6%/1stk

22:59:16.257 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:12%/1stk

22:59:17.355 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:15%/1stk

22:59:18.440 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:15%/1stk
[22:59:18]   ep 24 [0-19%]: hits=1 dmg=14.5  (ref 3.9/27.0)  [game ep 24, handoff f3752]

22:59:19.439 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:15%/1stk

22:59:20.459 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:16%/1stk

22:59:21.606 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:21%/1stk

22:59:22.737 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:30%/1stk

22:59:23.894 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:32%/1stk
[22:59:24]   ep 25 [0-19%]: hits=2 dmg=17.6  (ref 3.9/27.0)  [game ep 25, handoff f4067]

22:59:25.011 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:32%/1stk

22:59:26.132 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:32%/1stk

22:59:27.245 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:32%/1stk

22:59:28.436 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:39%/1stk

22:59:29.463 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:40%/1stk

22:59:30.532 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:40%/1stk

22:59:31.573 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:40%/1stk
[22:59:31]   ep 26 [20-39%]: hits=2 dmg=7.4  (ref 3.7/20.5)  [game ep 26, handoff f4459]

22:59:32.572 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:40%/1stk

22:59:33.709 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:46%/1stk

22:59:34.789 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:46%/1stk

22:59:35.907 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:46%/1stk

22:59:37.062 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:53%/1stk
[22:59:37]   ep 27 [20-39%]: hits=2 dmg=13.3  (ref 3.7/20.5)  [game ep 27, handoff f4746]

22:59:37.306 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:53%/1stk

22:59:37.445 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:53%/1stk

22:59:37.682 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

22:59:37.696 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

22:59:37.823 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[22:59:37]   -- game boundary (game 3 starting)

22:59:38.022 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:59:39.141 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:59:40.245 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:59:41.382 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

22:59:42.491 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

22:59:43.539 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

22:59:44.582 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

22:59:45.581 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:26%/4stk
[22:59:46]   ep 28 [0-19%]: hits=2 dmg=25.7  (ref 3.9/27.0)  [game ep 28, handoff f98]

22:59:46.651 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:26%/4stk

22:59:47.739 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:26%/4stk

22:59:48.855 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:26%/4stk

22:59:49.973 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:33%/4stk

22:59:50.988 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:33%/4stk

22:59:52.015 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:33%/4stk

22:59:53.100 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:33%/4stk
[22:59:53]   ep 29 [20-39%]: hits=2 dmg=6.9  (ref 3.7/20.5)  [game ep 29, handoff f488]

22:59:54.252 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:33%/4stk

22:59:55.340 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:39%/4stk

22:59:56.417 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:40%/4stk

22:59:57.522 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:40%/4stk

22:59:58.642 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:40%/4stk
[22:59:59]   ep 30 [20-39%]: hits=2 dmg=13.0  (ref 3.7/20.5)  [game ep 30, handoff f817]

22:59:59.461 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:48%/4stk

22:59:59.592 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:48%/4stk

23:00:00.350 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:48%/3stk

23:00:01.515 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:00:02.708 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:00:03.909 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:00:05.098 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:7%/3stk

23:00:06.241 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:12%/3stk

23:00:07.330 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:15%/3stk

23:00:08.452 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:25%/3stk
[23:00:09]   ep 31 [0-19%]: hits=6 dmg=26.7  (ref 3.9/27.0)  [game ep 31, handoff f1411]

23:00:09.509 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:28%/3stk

23:00:10.546 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:00:11.551 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:00:12.639 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:47%/3stk

23:00:13.745 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:47%/3stk
[23:00:14]   ep 32 [20-39%]: hits=2 dmg=18.5  (ref 3.7/20.5)  [game ep 32, handoff f1703]

23:00:14.257 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/3stk P2:47%/3stk

23:00:14.415 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/3stk P2:47%/3stk

23:00:15.382 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/3stk P2:47%/2stk

23:00:16.400 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:00:17.552 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:00:18.557 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:00:19.573 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/3stk P2:7%/2stk

23:00:20.708 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/3stk P2:12%/2stk

23:00:22.229 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/3stk P2:12%/2stk

23:00:23.803 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/3stk P2:12%/2stk
[23:00:24]   ep 33 [0-19%]: hits=2 dmg=12.5  (ref 3.9/27.0)  [game ep 33, handoff f2297]

23:00:25.092 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/3stk P2:12%/2stk

23:00:26.252 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:00:27.378 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:00:28.441 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/3stk P2:45%/2stk

23:00:29.743 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/3stk P2:45%/2stk
[23:00:30]   ep 34 [0-19%]: hits=2 dmg=32.5  (ref 3.9/27.0)  [game ep 34, handoff f2620]

23:00:30.684 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/3stk P2:45%/2stk

23:00:30.813 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:45%/2stk

23:00:31.390 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:45%/1stk

23:00:32.392 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:00:33.496 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:00:34.523 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:00:35.630 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:6%/1stk

23:00:36.732 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:12%/1stk

23:00:37.781 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:38.783 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:15%/1stk
[23:00:39]   ep 35 [0-19%]: hits=1 dmg=14.7  (ref 3.9/27.0)  [game ep 35, handoff f3221]

23:00:39.783 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:40.754 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:41.754 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:42.814 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:43.828 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:44.814 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:45.914 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:46.998 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:48.083 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:49.256 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:50.355 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:51.492 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:52.552 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:53.606 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:54.651 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:55.717 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:56.786 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:57.926 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:00:59.110 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:01:00.212 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:15%/1stk
[23:01:01]     [reset] position timeout

23:01:01.062 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:01:01.197 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:01:01.473 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:01:01.488 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:01:01.617 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:01:01]   -- game boundary (game 4 starting)

23:01:01.782 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:01:02.920 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:01:04.102 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:01:05.335 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:01:06.585 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:01:07.676 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:01:08.747 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:37%/4stk

23:01:09.809 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:37%/4stk
[23:01:10]   ep 36 [0-19%]: hits=1 dmg=36.8  (ref 3.9/27.0)  [game ep 36, handoff f98]

23:01:10.823 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:37%/4stk

23:01:11.860 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:37%/4stk

23:01:12.974 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:44%/4stk

23:01:14.123 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:44%/4stk

23:01:15.272 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:44%/4stk

23:01:16.416 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:44%/4stk
[23:01:17]   ep 37 [20-39%]: hits=2 dmg=7.0  (ref 3.7/20.5)  [game ep 37, handoff f455]

23:01:17.121 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:44%/4stk

23:01:17.283 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:44%/4stk

23:01:17.772 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:44%/3stk

23:01:18.970 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:01:20.091 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:01:21.243 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:01:22.356 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:2%/3stk

23:01:23.551 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:12%/3stk

23:01:24.600 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:12%/3stk

23:01:25.642 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:12%/3stk
[23:01:26]   ep 38 [0-19%]: hits=2 dmg=22.7  (ref 3.9/27.0)  [game ep 38, handoff f1069]

23:01:26.712 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:23%/3stk

23:01:27.764 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:23%/3stk

23:01:28.854 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:01:29.957 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:01:31.042 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:01:32.079 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:29%/3stk
[23:01:32]   ep 39 [20-39%]: hits=2 dmg=5.8  (ref 3.7/20.5)  [game ep 39, handoff f1413]

23:01:33.086 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:01:34.099 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:01:35.182 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:34%/3stk

23:01:36.184 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:34%/3stk

23:01:37.250 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:34%/3stk

23:01:38.327 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:34%/3stk
[23:01:38]   ep 40 [20-39%]: hits=2 dmg=5.4  (ref 3.7/20.5)  [game ep 40, handoff f1752]

23:01:39.452 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:40%/3stk

23:01:40.635 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:46%/3stk

23:01:41.780 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:55%/3stk

23:01:42.925 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:55%/3stk

23:01:44.096 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:55%/3stk
[23:01:45]   ep 41 [20-39%]: hits=2 dmg=15.0  (ref 3.7/20.5)  [game ep 41, handoff f2079]

23:01:45.077 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:55%/3stk

23:01:45.207 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:55%/3stk

23:01:46.034 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:55%/2stk

23:01:47.034 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:01:48.043 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:01:49.044 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:01:50.124 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:01:51.160 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:01:52.176 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:01:53.237 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/3stk P2:19%/2stk
[23:01:53]   ep 42 [0-19%]: hits=2 dmg=18.9  (ref 3.9/27.0)  [game ep 42, handoff f2661]

23:01:54.303 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:01:55.391 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:01:56.511 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:01:58.076 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:01:59.577 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:26%/2stk
[23:02:01]   ep 43 [0-19%]: hits=2 dmg=7.5  (ref 3.9/27.0)  [game ep 43, handoff f2996]

23:02:01.091 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:02.386 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:03.511 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:04.549 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:05.621 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:06.696 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:07.861 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:09.075 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:10.221 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:11.420 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:12.534 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:13.643 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:14.746 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:15.854 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:17.056 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:18.179 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:19.256 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:20.427 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:21.612 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:22.649 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:26%/2stk
[23:02:23]     [reset] position timeout

23:02:23.773 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:23.902 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:02:24.334 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:26%/1stk

23:02:25.393 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:02:26.542 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:02:27.742 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:02:28.940 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:2%/1stk

23:02:30.204 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:7%/1stk

23:02:31.432 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:7%/1stk

23:02:32.663 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:7%/1stk
[23:02:33]   ep 44 [0-19%]: hits=2 dmg=6.8  (ref 3.9/27.0)  [game ep 44, handoff f4789]

23:02:33.797 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:7%/1stk

23:02:34.969 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:13%/1stk

23:02:36.142 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:31%/1stk

23:02:37.350 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:40%/1stk

23:02:38.495 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:40%/1stk
[23:02:39]   ep 45 [0-19%]: hits=1 dmg=33.4  (ref 3.9/27.0)  [game ep 45, handoff f5076]

23:02:39.206 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:40%/1stk

23:02:39.336 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:40%/1stk

23:02:39.560 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:02:39.576 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:02:39.708 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:02:39]   -- game boundary (game 5 starting)

23:02:39.901 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:02:41.042 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:02:42.162 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:02:43.300 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:02:44.475 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:02:45.542 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:02:46.739 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:02:47.808 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:20%/4stk
[23:02:48]   ep 46 [0-19%]: hits=2 dmg=20.4  (ref 3.9/27.0)  [game ep 46, handoff f98]

23:02:48.893 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:02:49.973 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:02:51.054 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:02:52.187 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:02:53.245 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:02:54.376 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:02:55.503 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:27%/4stk
[23:02:55]   ep 47 [20-39%]: hits=2 dmg=6.9  (ref 3.7/20.5)  [game ep 47, handoff f505]

23:02:56.513 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:02:57.705 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:02:58.973 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:03:00.090 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:03:01.223 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:03:02.351 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:34%/4stk
[23:03:02]   ep 48 [20-39%]: hits=2 dmg=6.3  (ref 3.7/20.5)  [game ep 48, handoff f845]

23:03:03.512 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:03:04.709 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:03:05.922 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:35%/4stk

23:03:07.022 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:08.205 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:09.326 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:39%/4stk
[23:03:10]   ep 49 [20-39%]: hits=2 dmg=5.8  (ref 3.7/20.5)  [game ep 49, handoff f1249]

23:03:10.404 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:11.424 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:12.536 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:13.696 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:39%/4stk

23:03:14.901 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:03:16.022 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:03:17.111 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:03:18.136 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:49%/4stk
[23:03:18]   ep 50 [20-39%]: hits=2 dmg=13.4  (ref 3.7/20.5)  [game ep 50, handoff f1691]

23:03:18.434 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:53%/4stk

23:03:18.565 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:53%/4stk

23:03:19.400 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:53%/3stk

23:03:20.455 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:03:21.573 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:03:22.723 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:03:23.900 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:03:24.998 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:03:26.095 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:03:27.155 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:6%/3stk
[23:03:27]   ep 51 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 51, handoff f2299]

23:03:28.258 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:03:29.313 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:03:30.386 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:11%/3stk

23:03:31.406 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:11%/3stk

23:03:32.593 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:11%/3stk

23:03:33.809 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:11%/3stk
[23:03:33]   ep 52 [0-19%]: hits=2 dmg=5.6  (ref 3.9/27.0)  [game ep 52, handoff f2645]

23:03:35.004 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:11%/3stk

23:03:36.203 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:11%/3stk

23:03:37.370 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:11%/3stk

23:03:38.474 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:17%/3stk

23:03:39.713 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:19%/3stk

23:03:40.892 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:19%/3stk

23:03:42.098 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:19%/3stk
[23:03:43]   ep 53 [0-19%]: hits=2 dmg=14.5  (ref 3.9/27.0)  [game ep 53, handoff f3104]

23:03:43.291 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:26%/3stk

23:03:44.471 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:26%/3stk

23:03:45.613 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:33%/3stk

23:03:46.742 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:45%/3stk

23:03:47.850 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:45%/3stk

23:03:49.062 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:45%/3stk
[23:03:49]   ep 54 [20-39%]: hits=1 dmg=18.9  (ref 3.7/20.5)  [game ep 54, handoff f3460]

23:03:49.806 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:45%/3stk

23:03:50.012 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:45%/2stk

23:03:51.072 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:03:52.070 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:03:53.162 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:03:54.274 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:03:55.432 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:10%/2stk

23:03:56.623 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:03:57.904 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:03:59.449 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:19%/2stk
[23:03:59]   ep 55 [0-19%]: hits=2 dmg=19.4  (ref 3.9/27.0)  [game ep 55, handoff f4025]

23:04:00.796 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:04:01.908 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:04:03.056 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:04:04.252 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:04:05.389 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:04:06.574 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:04:07.714 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:32%/2stk
[23:04:08]   ep 56 [0-19%]: hits=2 dmg=12.9  (ref 3.9/27.0)  [game ep 56, handoff f4473]

23:04:08.842 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:04:10.137 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:04:11.282 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:32%/2stk

23:04:12.490 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:04:13.657 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:43%/2stk

23:04:14.862 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:46%/2stk

23:04:16.273 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:60%/2stk
[23:04:16]   ep 57 [20-39%]: hits=2 dmg=30.9  (ref 3.7/20.5)  [game ep 57, handoff f4878]

23:04:16.815 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:65%/2stk

23:04:17.083 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:65%/1stk

23:04:18.159 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:04:19.301 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:04:20.479 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:04:21.562 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:04:22.803 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/3stk P2:11%/1stk

23:04:23.844 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/3stk P2:31%/1stk

23:04:24.846 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:25.910 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/2stk P2:31%/1stk
[23:04:25]   ep 58 [0-19%]: hits=3 dmg=31.3  (ref 3.9/27.0)  [game ep 58, handoff f5462]

23:04:27.015 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:28.195 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:29.263 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:30.401 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:31.574 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:32.604 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:33.666 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:34.673 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:35.782 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:36.933 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:38.075 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:39.268 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:40.459 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:41.624 [info] [MeleePort] Frame 6540: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:42.719 [info] [MeleePort] Frame 6600: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:43.765 [info] [MeleePort] Frame 6660: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:44.890 [info] [MeleePort] Frame 6720: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:46.107 [info] [MeleePort] Frame 6780: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:47.270 [info] [MeleePort] Frame 6840: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:48.400 [info] [MeleePort] Frame 6900: menu_state=2 | P1:0%/2stk P2:31%/1stk
[23:04:48]     [reset] position timeout

23:04:48.603 [info] [MeleePort] Frame 6960: menu_state=2 | P1:0%/2stk P2:31%/1stk

23:04:48.904 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:04:48.920 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:04:49.046 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:04:49]   -- game boundary (game 6 starting)

23:04:49.243 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:04:50.444 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:04:51.682 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:04:52.807 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:04:53.892 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:04:54.963 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:13%/4stk

23:04:56.028 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:16%/4stk

23:04:57.272 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:24%/4stk
[23:04:58]   ep 59 [0-19%]: hits=1 dmg=30.2  (ref 3.9/27.0)  [game ep 59, handoff f98]

23:04:58.463 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:32%/4stk

23:04:59.728 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:05:00.933 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:05:02.134 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:63%/4stk

23:05:03.299 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:63%/4stk
[23:05:03]   ep 60 [20-39%]: hits=4 dmg=31.3  (ref 3.7/20.5)  [game ep 60, handoff f391]

23:05:03.948 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:63%/4stk

23:05:04.076 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:63%/4stk

23:05:04.829 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:63%/3stk

23:05:05.982 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:05:07.146 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:05:08.243 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:05:09.449 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:7%/3stk

23:05:10.684 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:05:11.818 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:05:12.874 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/3stk P2:31%/3stk
[23:05:13]   ep 61 [0-19%]: hits=2 dmg=30.6  (ref 3.9/27.0)  [game ep 61, handoff f989]

23:05:13.948 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:15.092 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:16.137 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:17.164 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:18.173 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:19.297 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:20.433 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:21.543 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:22.662 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:23.823 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:24.949 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:25.943 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:26.957 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:28.085 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:29.211 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:30.418 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:31.623 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:32.846 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:34.041 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:35.172 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/3stk P2:31%/3stk
[23:05:35]     [reset] position timeout

23:05:35.828 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:35.956 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/3stk P2:31%/3stk

23:05:36.782 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/3stk P2:31%/2stk

23:05:37.954 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:05:39.139 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:05:40.307 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:05:41.501 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/3stk P2:7%/2stk

23:05:42.707 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:05:43.859 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:05:45.010 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:19%/2stk
[23:05:45]   ep 62 [0-19%]: hits=1 dmg=19.5  (ref 3.9/27.0)  [game ep 62, handoff f2786]

23:05:46.059 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:05:47.132 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:05:48.263 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:05:49.494 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:31%/2stk

23:05:50.723 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:31%/2stk

23:05:51.967 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:31%/2stk
[23:05:52]   ep 63 [0-19%]: hits=2 dmg=11.9  (ref 3.9/27.0)  [game ep 63, handoff f3158]

23:05:53.518 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:31%/2stk

23:05:54.757 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:31%/2stk

23:05:55.943 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:05:57.052 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:05:58.250 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:05:59.441 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:38%/2stk
[23:05:59]   ep 64 [20-39%]: hits=2 dmg=6.5  (ref 3.7/20.5)  [game ep 64, handoff f3483]

23:06:00.574 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:06:01.716 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:06:02.810 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:38%/2stk

23:06:04.034 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:44%/2stk

23:06:05.353 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:44%/2stk

23:06:06.502 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:44%/2stk

23:06:07.654 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:44%/2stk
[23:06:07]   ep 65 [20-39%]: hits=2 dmg=6.0  (ref 3.7/20.5)  [game ep 65, handoff f3906]

23:06:07.889 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:44%/2stk

23:06:08.021 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:44%/2stk

23:06:08.891 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:44%/1stk

23:06:10.051 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:06:11.191 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:06:12.290 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:06:13.515 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:6%/1stk

23:06:14.612 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:21%/1stk

23:06:15.704 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:21%/1stk

23:06:16.780 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:21%/1stk
[23:06:17]   ep 66 [0-19%]: hits=2 dmg=20.8  (ref 3.9/27.0)  [game ep 66, handoff f4525]

23:06:17.784 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:21%/1stk

23:06:18.859 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:21%/1stk

23:06:19.932 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:21%/1stk

23:06:21.019 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:26%/1stk

23:06:22.067 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:26%/1stk

23:06:23.219 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:26%/1stk
[23:06:24]   ep 67 [20-39%]: hits=2 dmg=5.6  (ref 3.7/20.5)  [game ep 67, handoff f4918]

23:06:24.573 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:26%/1stk

23:06:25.907 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:26%/1stk

23:06:27.049 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:32%/1stk

23:06:28.178 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:33%/1stk

23:06:29.367 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:33%/1stk

23:06:30.445 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/3stk P2:33%/1stk
[23:06:30]   ep 68 [20-39%]: hits=2 dmg=6.3  (ref 3.7/20.5)  [game ep 68, handoff f5226]

23:06:31.555 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/3stk P2:34%/1stk

23:06:32.773 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/3stk P2:51%/1stk

23:06:33.912 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/3stk P2:51%/1stk

23:06:34.973 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/3stk P2:51%/1stk
[23:06:35]   ep 69 [20-39%]: hits=1 dmg=32.7  (ref 3.7/20.5)  [game ep 69, handoff f5514]

23:06:35.968 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/3stk P2:65%/1stk

23:06:36.104 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/2stk P2:65%/1stk

23:06:36.233 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/2stk P2:65%/1stk

23:06:36.451 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:06:36.466 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:06:36.596 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:06:36]   -- game boundary (game 7 starting)

23:06:36.765 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:06:37.921 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:06:39.123 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:06:40.225 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:06:41.310 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:06:42.423 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:06:43.549 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:06:44.951 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:7%/4stk
[23:06:45]   ep 70 [0-19%]: hits=2 dmg=12.5  (ref 3.9/27.0)  [game ep 70, handoff f98]

23:06:45.986 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:12%/4stk

23:06:47.072 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:06:48.158 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:06:49.143 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:06:50.233 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:19%/4stk
[23:06:51]   ep 71 [0-19%]: hits=2 dmg=6.9  (ref 3.9/27.0)  [game ep 71, handoff f402]

23:06:51.777 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:06:52.960 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:06:54.132 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:06:55.308 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:06:56.394 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:06:57.393 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:27%/4stk
[23:06:57]   ep 72 [0-19%]: hits=2 dmg=7.3  (ref 3.9/27.0)  [game ep 72, handoff f737]

23:06:58.497 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:06:59.682 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:07:00.852 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:07:01.853 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:34%/4stk
[23:07:02]   ep 73 [20-39%]: hits=2 dmg=7.0  (ref 3.7/20.5)  [game ep 73, handoff f1016]

23:07:02.832 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:34%/4stk

23:07:04.012 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:07:05.101 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:07:06.144 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:07:07.194 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:40%/4stk
[23:07:07]   ep 74 [20-39%]: hits=2 dmg=5.9  (ref 3.7/20.5)  [game ep 74, handoff f1303]

23:07:08.283 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:07:09.432 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:40%/4stk

23:07:10.547 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:07:11.595 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:07:12.735 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:07:14.311 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:45%/4stk
[23:07:14]   ep 75 [20-39%]: hits=2 dmg=5.6  (ref 3.7/20.5)  [game ep 75, handoff f1632]

23:07:14.708 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:07:14.840 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:07:15.629 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:45%/3stk

23:07:16.782 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:07:17.932 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:07:19.133 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:07:20.410 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:5%/3stk

23:07:21.645 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:07:22.867 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:07:24.381 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:31%/3stk
[23:07:25]   ep 76 [0-19%]: hits=2 dmg=31.0  (ref 3.9/27.0)  [game ep 76, handoff f2245]

23:07:25.970 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:07:27.043 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:07:28.183 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:07:29.447 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:37%/3stk

23:07:30.710 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:49%/3stk

23:07:31.884 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:49%/3stk

23:07:32.940 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:49%/3stk
[23:07:33]   ep 77 [20-39%]: hits=2 dmg=17.5  (ref 3.7/20.5)  [game ep 77, handoff f2649]

23:07:33.202 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:49%/3stk

23:07:33.606 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:49%/2stk

23:07:34.680 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:07:35.739 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:07:36.845 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:07:38.007 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:2%/2stk

23:07:39.156 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:9%/2stk

23:07:40.335 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:14%/2stk

23:07:41.570 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:14%/2stk
[23:07:42]   ep 78 [0-19%]: hits=1 dmg=14.0  (ref 3.9/27.0)  [game ep 78, handoff f3232]

23:07:42.746 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:14%/2stk

23:07:43.808 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:44.994 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:46.168 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:47.321 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:48.409 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:49.450 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:50.604 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:51.763 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:53.034 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:54.158 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:55.304 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:56.489 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:57.660 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:07:58.821 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:00.079 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:01.335 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:02.533 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:03.758 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:04.986 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:14%/2stk
[23:08:06]     [reset] position timeout

23:08:06.116 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:06.249 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:14%/2stk

23:08:06.751 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:14%/1stk

23:08:07.953 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:08:09.136 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:08:10.304 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:08:11.411 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:2%/1stk

23:08:12.481 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:7%/1stk

23:08:13.627 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:8%/1stk

23:08:14.800 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:8%/1stk
[23:08:15]   ep 79 [0-19%]: hits=2 dmg=8.5  (ref 3.9/27.0)  [game ep 79, handoff f5029]

23:08:15.914 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:8%/1stk

23:08:17.004 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:10%/1stk

23:08:18.070 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:08:19.104 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:08:20.161 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/3stk P2:15%/1stk
[23:08:21]   ep 80 [0-19%]: hits=2 dmg=6.9  (ref 3.9/27.0)  [game ep 80, handoff f5333]

23:08:21.364 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/3stk P2:15%/1stk

23:08:22.590 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/3stk P2:17%/1stk

23:08:23.755 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:24.906 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:26.378 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/3stk P2:27%/1stk
[23:08:27]   ep 81 [0-19%]: hits=2 dmg=12.1  (ref 3.9/27.0)  [game ep 81, handoff f5631]

23:08:27.902 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:29.039 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:30.055 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:31.132 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:32.233 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:33.447 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:34.676 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:35.801 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:36.886 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:37.903 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:38.918 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:39.981 [info] [MeleePort] Frame 6540: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:41.018 [info] [MeleePort] Frame 6600: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:42.098 [info] [MeleePort] Frame 6660: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:43.140 [info] [MeleePort] Frame 6720: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:44.297 [info] [MeleePort] Frame 6780: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:45.490 [info] [MeleePort] Frame 6840: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:46.650 [info] [MeleePort] Frame 6900: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:47.677 [info] [MeleePort] Frame 6960: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:48.781 [info] [MeleePort] Frame 7020: menu_state=2 | P1:0%/3stk P2:27%/1stk
[23:08:49]     [reset] position timeout

23:08:49.824 [info] [MeleePort] Frame 7080: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:49.961 [info] [MeleePort] Frame 7140: menu_state=2 | P1:0%/3stk P2:27%/1stk

23:08:50.242 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:08:50.257 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:08:50.380 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:08:50]   -- game boundary (game 8 starting)

23:08:50.568 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:08:51.650 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:08:52.795 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:08:53.993 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:08:55.141 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:08:56.155 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:08:57.207 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:08:58.309 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:38%/4stk
[23:08:59]   ep 82 [0-19%]: hits=2 dmg=38.5  (ref 3.9/27.0)  [game ep 82, handoff f98]

23:08:59.381 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:09:00.378 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:09:01.494 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:09:02.533 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:09:03.560 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:38%/4stk

23:09:04.759 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:45%/4stk

23:09:05.937 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:67%/4stk

23:09:07.143 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:67%/4stk

23:09:08.536 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:67%/4stk
[23:09:08]   ep 83 [20-39%]: hits=4 dmg=29.0  (ref 3.7/20.5)  [game ep 83, handoff f601]

23:09:08.714 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:67%/4stk

23:09:09.045 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:67%/3stk

23:09:10.263 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:09:11.433 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:09:12.626 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:09:13.748 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:09:14.953 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:10%/3stk

23:09:16.092 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:09:17.253 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:28%/3stk
[23:09:18]   ep 84 [0-19%]: hits=2 dmg=27.8  (ref 3.9/27.0)  [game ep 84, handoff f1198]

23:09:18.325 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:28%/3stk

23:09:19.462 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:34%/3stk

23:09:20.631 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:34%/3stk

23:09:21.866 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:09:23.107 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:47%/3stk
[23:09:23]   ep 85 [20-39%]: hits=7 dmg=20.5  (ref 3.7/20.5)  [game ep 85, handoff f1455]

23:09:23.513 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:48%/3stk

23:09:23.827 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:48%/2stk

23:09:24.904 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:09:26.026 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:09:27.166 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:09:28.265 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:09:29.373 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:14%/2stk

23:09:30.496 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:33%/2stk

23:09:31.606 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:33%/2stk
[23:09:32]   ep 86 [0-19%]: hits=2 dmg=32.6  (ref 3.9/27.0)  [game ep 86, handoff f2039]

23:09:32.651 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:33%/2stk

23:09:33.739 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:33%/2stk

23:09:34.926 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:33%/2stk

23:09:36.099 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:39%/2stk

23:09:37.200 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:44%/2stk

23:09:38.264 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:44%/2stk

23:09:39.333 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:44%/2stk
[23:09:40]   ep 87 [20-39%]: hits=2 dmg=11.7  (ref 3.7/20.5)  [game ep 87, handoff f2443]

23:09:40.119 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:44%/2stk

23:09:40.279 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:44%/1stk

23:09:41.333 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:09:42.435 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:09:43.625 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:09:44.845 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:09:46.058 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:21%/1stk

23:09:47.199 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:21%/1stk

23:09:48.511 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:27%/1stk

23:09:49.946 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:40%/1stk
[23:09:50]   ep 88 [0-19%]: hits=1 dmg=41.2  (ref 3.9/27.0)  [game ep 88, handoff f3007]

23:09:50.221 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:45%/1stk

23:09:50.499 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:09:50.514 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:09:50.638 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:09:50]   -- game boundary (game 9 starting)

23:09:50.831 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:09:51.938 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:09:53.058 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:09:54.134 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:09:55.245 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:09:56.332 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:09:57.585 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:09:59.205 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:7%/4stk
[23:10:00]   ep 89 [0-19%]: hits=2 dmg=7.5  (ref 3.9/27.0)  [game ep 89, handoff f98]

23:10:00.841 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:10:01.965 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:10:03.133 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:14%/4stk

23:10:04.214 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:28%/4stk

23:10:05.253 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:28%/4stk
[23:10:06]   ep 90 [0-19%]: hits=2 dmg=20.8  (ref 3.9/27.0)  [game ep 90, handoff f419]

23:10:06.348 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:28%/4stk

23:10:07.517 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:28%/4stk

23:10:08.661 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:35%/4stk

23:10:09.753 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:35%/4stk

23:10:11.034 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:35%/4stk

23:10:12.556 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:35%/4stk
[23:10:12]   ep 91 [20-39%]: hits=2 dmg=6.5  (ref 3.7/20.5)  [game ep 91, handoff f722]

23:10:13.824 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:35%/4stk

23:10:14.915 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:10:16.003 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:41%/4stk

23:10:17.073 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:41%/4stk

23:10:18.199 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:41%/4stk
[23:10:19]   ep 92 [20-39%]: hits=2 dmg=6.0  (ref 3.7/20.5)  [game ep 92, handoff f1070]

23:10:19.187 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:41%/4stk

23:10:19.316 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:41%/4stk

23:10:19.532 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:41%/3stk

23:10:20.659 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:10:21.674 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:10:22.702 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:10:23.758 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:10:24.889 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:11%/3stk

23:10:25.926 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:10:27.073 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:10:28.109 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:19%/3stk
[23:10:28]   ep 93 [0-19%]: hits=2 dmg=18.5  (ref 3.9/27.0)  [game ep 93, handoff f1684]

23:10:29.223 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:10:30.249 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:10:31.351 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:19%/3stk

23:10:32.547 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:24%/3stk

23:10:33.610 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:10:34.703 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:10:35.869 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:25%/3stk
[23:10:36]   ep 94 [0-19%]: hits=2 dmg=6.7  (ref 3.9/27.0)  [game ep 94, handoff f2123]

23:10:37.018 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:10:38.143 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:10:39.343 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:27%/3stk

23:10:40.473 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:10:41.593 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:40%/3stk

23:10:42.735 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:42%/3stk
[23:10:43]   ep 95 [20-39%]: hits=2 dmg=16.3  (ref 3.7/20.5)  [game ep 95, handoff f2507]

23:10:43.890 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:10:44.033 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:10:44.804 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:42%/2stk

23:10:45.920 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:10:47.017 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:10:48.213 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:10:49.484 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:10:50.682 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:11%/2stk

23:10:51.859 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:11%/2stk

23:10:52.933 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:11%/2stk
[23:10:53]   ep 96 [0-19%]: hits=2 dmg=10.9  (ref 3.9/27.0)  [game ep 96, handoff f3092]

23:10:53.968 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:11%/2stk

23:10:55.030 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:17%/2stk

23:10:56.091 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:29%/2stk

23:10:57.205 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:29%/2stk

23:10:58.261 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:29%/2stk
[23:10:58]   ep 97 [0-19%]: hits=2 dmg=18.1  (ref 3.9/27.0)  [game ep 97, handoff f3392]

23:10:59.397 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:29%/2stk

23:11:00.580 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:29%/2stk

23:11:01.733 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:29%/2stk

23:11:03.035 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:35%/2stk

23:11:04.236 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:39%/2stk

23:11:05.321 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:39%/2stk

23:11:06.373 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:39%/2stk
[23:11:06]   ep 98 [20-39%]: hits=2 dmg=10.1  (ref 3.7/20.5)  [game ep 98, handoff f3805]

23:11:07.561 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:39%/2stk

23:11:08.869 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:45%/2stk

23:11:10.043 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:45%/2stk

23:11:11.195 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:45%/2stk

23:11:12.333 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:45%/2stk
[23:11:12]   ep 99 [20-39%]: hits=2 dmg=6.1  (ref 3.7/20.5)  [game ep 99, handoff f4096]

23:11:12.711 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:45%/2stk

23:11:12.839 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:45%/2stk

23:11:13.467 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:45%/1stk

23:11:14.511 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:11:15.621 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:11:16.663 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:11:17.705 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:11:18.765 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:11%/1stk

23:11:19.913 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:21.090 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:14%/1stk
[23:11:21]   ep 100 [0-19%]: hits=1 dmg=14.0  (ref 3.9/27.0)  [game ep 100, handoff f4721]

23:11:22.221 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:23.328 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:24.465 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:25.683 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:26.904 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:28.076 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:29.166 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:30.223 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:31.352 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:32.405 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:33.444 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:34.503 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:35.550 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:36.712 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:37.926 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:39.136 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:40.343 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:41.593 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:42.762 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:43.905 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:14%/1stk
[23:11:44]     [reset] position timeout

23:11:44.778 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:44.907 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/4stk P2:14%/1stk

23:11:45.156 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:11:45.170 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:11:45.301 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:11:45]   -- game boundary (game 10 starting)

23:11:45.495 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:11:46.693 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:11:47.959 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:11:49.227 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:11:50.455 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:11:51.755 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:11:53.468 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:11:55.276 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:19%/4stk
[23:11:56]   ep 101 [0-19%]: hits=1 dmg=19.5  (ref 3.9/27.0)  [game ep 101, handoff f98]

23:11:57.030 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:11:58.245 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:11:59.756 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:01.495 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:03.058 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:29%/4stk
[23:12:03]   ep 102 [0-19%]: hits=2 dmg=9.9  (ref 3.9/27.0)  [game ep 102, handoff f394]

23:12:04.677 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:05.851 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:06.931 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:08.122 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:09.339 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:10.444 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:11.482 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:12.514 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:13.653 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:14.665 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:15.781 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:16.886 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:17.986 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:19.036 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:20.071 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:21.103 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:22.144 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:23.179 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:24.238 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:25.248 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:29%/4stk
[23:12:25]     [reset] position timeout

23:12:25.913 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:26.042 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:29%/4stk

23:12:26.776 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:29%/3stk

23:12:27.812 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:12:28.843 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:12:29.861 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:12:31.067 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:12:32.165 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:12:33.325 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:12:34.494 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:6%/3stk
[23:12:35]   ep 103 [0-19%]: hits=2 dmg=6.5  (ref 3.9/27.0)  [game ep 103, handoff f2189]

23:12:35.670 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:12:36.874 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:8%/3stk

23:12:38.033 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:17%/3stk

23:12:39.228 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:17%/3stk

23:12:40.848 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:17%/3stk
[23:12:42]   ep 104 [0-19%]: hits=2 dmg=11.0  (ref 3.9/27.0)  [game ep 104, handoff f2511]

23:12:42.360 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:17%/3stk

23:12:43.611 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:17%/3stk

23:12:44.777 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:23%/3stk

23:12:45.992 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:23%/3stk

23:12:47.164 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:23%/3stk

23:12:48.323 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:47%/3stk
[23:12:48]   ep 105 [0-19%]: hits=3 dmg=29.7  (ref 3.9/27.0)  [game ep 105, handoff f2831]

23:12:48.641 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:47%/3stk

23:12:48.776 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:47%/3stk

23:12:49.117 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:47%/2stk

23:12:50.306 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:12:51.510 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:12:52.628 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:12:53.829 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:12:55.090 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:12:56.348 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:12:57.428 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:24%/2stk
[23:12:58]   ep 106 [0-19%]: hits=1 dmg=23.9  (ref 3.9/27.0)  [game ep 106, handoff f3477]

23:12:58.534 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:12:59.685 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:25%/2stk

23:13:00.909 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:30%/2stk

23:13:02.059 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:37%/2stk

23:13:03.174 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:37%/2stk
[23:13:04]   ep 107 [20-39%]: hits=2 dmg=13.0  (ref 3.7/20.5)  [game ep 107, handoff f3770]

23:13:04.264 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:37%/2stk

23:13:05.394 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:37%/2stk

23:13:06.470 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:37%/2stk

23:13:07.583 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:43%/2stk

23:13:08.742 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:43%/2stk

23:13:09.850 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:43%/2stk

23:13:10.979 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:54%/2stk
[23:13:11]   ep 108 [20-39%]: hits=2 dmg=16.8  (ref 3.7/20.5)  [game ep 108, handoff f4157]

23:13:11.397 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:54%/2stk

23:13:11.699 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:54%/1stk

23:13:12.736 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:13:13.799 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:13:14.821 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:13:15.802 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:13:16.838 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:13:17.962 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:13:19.050 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:6%/1stk
[23:13:20]   ep 109 [0-19%]: hits=2 dmg=6.1  (ref 3.9/27.0)  [game ep 109, handoff f4738]

23:13:20.160 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:13:21.336 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:13:22.465 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:12%/1stk

23:13:23.529 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:12%/1stk

23:13:24.692 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:12%/1stk

23:13:25.853 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:21%/1stk
[23:13:26]   ep 110 [0-19%]: hits=8 dmg=19.1  (ref 3.9/27.0)  [game ep 110, handoff f5083]

23:13:27.006 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:26%/1stk

23:13:28.210 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:29.345 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:30.403 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:31.634 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:33%/1stk
[23:13:32]   ep 111 [20-39%]: hits=2 dmg=6.8  (ref 3.7/20.5)  [game ep 111, handoff f5370]

23:13:33.198 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:34.381 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:35.481 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:36.594 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:33%/1stk

23:13:37.673 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:39%/1stk

23:13:38.878 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:39%/1stk

23:13:40.392 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:39%/1stk

23:13:41.918 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:39%/1stk
[23:13:42]   ep 112 [20-39%]: hits=2 dmg=6.3  (ref 3.7/20.5)  [game ep 112, handoff f5840]

23:13:43.443 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:39%/1stk

23:13:44.686 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/4stk P2:45%/1stk

23:13:45.891 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/4stk P2:45%/1stk

23:13:47.063 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/4stk P2:45%/1stk

23:13:48.189 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/4stk P2:60%/1stk
[23:13:48]   ep 113 [20-39%]: hits=2 dmg=21.0  (ref 3.7/20.5)  [game ep 113, handoff f6146]

23:13:48.775 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/4stk P2:60%/1stk

23:13:48.910 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/4stk P2:60%/1stk

23:13:49.151 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:13:49.164 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:13:49.292 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:13:49]   -- game boundary (game 11 starting)

23:13:49.492 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:13:50.628 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:13:51.850 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:13:53.017 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:13:54.212 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:13:55.284 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:13:56.292 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:13:57.394 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:20%/4stk
[23:13:58]   ep 114 [0-19%]: hits=2 dmg=20.4  (ref 3.9/27.0)  [game ep 114, handoff f98]

23:13:58.501 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:13:59.606 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:14:00.678 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:14:01.852 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:14:02.847 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:14:03.888 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:27%/4stk

23:14:04.901 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:27%/4stk
[23:14:04]   ep 115 [20-39%]: hits=2 dmg=6.9  (ref 3.7/20.5)  [game ep 115, handoff f484]

23:14:05.996 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/3stk P2:27%/4stk

23:14:07.043 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/3stk P2:27%/4stk

23:14:08.076 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/3stk P2:27%/4stk

23:14:09.158 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/3stk P2:35%/4stk

23:14:10.227 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/3stk P2:47%/4stk

23:14:11.230 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/3stk P2:47%/4stk

23:14:12.297 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/3stk P2:47%/4stk
[23:14:12]   ep 116 [20-39%]: hits=2 dmg=19.5  (ref 3.7/20.5)  [game ep 116, handoff f926]

23:14:12.837 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/3stk P2:47%/4stk

23:14:13.072 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/3stk P2:47%/3stk

23:14:14.122 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/3stk P2:0%/3stk

23:14:15.242 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/3stk P2:0%/3stk

23:14:16.347 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/3stk P2:0%/3stk

23:14:17.382 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/3stk P2:0%/3stk

23:14:18.468 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/3stk P2:7%/3stk

23:14:19.594 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/3stk P2:7%/3stk

23:14:20.680 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/3stk P2:7%/3stk

23:14:21.814 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/3stk P2:7%/3stk
[23:14:21]   ep 117 [0-19%]: hits=2 dmg=6.9  (ref 3.9/27.0)  [game ep 117, handoff f1502]

23:14:22.974 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/3stk P2:7%/3stk

23:14:24.226 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/3stk P2:10%/3stk

23:14:25.407 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/3stk P2:26%/3stk

23:14:26.525 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/3stk P2:26%/3stk

23:14:27.698 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/3stk P2:26%/3stk
[23:14:28]   ep 118 [0-19%]: hits=1 dmg=28.3  (ref 3.9/27.0)  [game ep 118, handoff f1847]

23:14:28.809 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/3stk P2:35%/3stk

23:14:29.981 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/3stk P2:41%/3stk

23:14:31.123 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/3stk P2:41%/3stk

23:14:32.292 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/3stk P2:41%/3stk

23:14:33.458 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/3stk P2:41%/3stk
[23:14:34]   ep 119 [20-39%]: hits=2 dmg=6.3  (ref 3.7/20.5)  [game ep 119, handoff f2132]

23:14:34.168 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/3stk P2:48%/3stk

23:14:34.296 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/3stk P2:48%/3stk

23:14:34.429 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/3stk P2:48%/3stk

23:14:34.976 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/3stk P2:48%/2stk

23:14:35.897 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:14:36.964 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:14:37.941 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/3stk P2:0%/2stk

23:14:38.989 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/3stk P2:6%/2stk

23:14:39.935 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/3stk P2:12%/2stk

23:14:40.878 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/3stk P2:19%/2stk

23:14:41.838 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/3stk P2:26%/2stk
[23:14:42]   ep 120 [0-19%]: hits=1 dmg=26.0  (ref 3.9/27.0)  [game ep 120, handoff f2799]

23:14:42.825 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:14:43.733 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:14:44.611 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:14:45.547 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/3stk P2:26%/2stk

23:14:46.529 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/3stk P2:33%/2stk

23:14:47.615 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/3stk P2:33%/2stk

23:14:48.532 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/3stk P2:33%/2stk

23:14:49.512 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/3stk P2:43%/2stk
[23:14:49]   ep 121 [20-39%]: hits=2 dmg=21.5  (ref 3.7/20.5)  [game ep 121, handoff f3264]

23:14:50.017 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/3stk P2:48%/2stk

23:14:50.147 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/3stk P2:48%/2stk

23:14:50.726 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/3stk P2:48%/1stk

23:14:51.600 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:14:52.495 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:14:53.370 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/3stk P2:0%/1stk

23:14:54.293 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/3stk P2:7%/1stk

23:14:55.184 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/3stk P2:13%/1stk

23:14:56.126 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:14:57.021 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/3stk P2:16%/1stk
[23:14:57]   ep 122 [0-19%]: hits=1 dmg=15.5  (ref 3.9/27.0)  [game ep 122, handoff f3873]

23:14:57.899 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:14:58.796 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:14:59.748 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:00.689 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:01.590 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:02.578 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:03.530 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:04.479 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:05.411 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:06.310 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:07.236 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:08.348 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:09.340 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:10.389 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:11.387 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:12.409 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:13.472 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:14.454 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:15.440 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:16.453 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/3stk P2:16%/1stk
[23:15:17]     [reset] position timeout

23:15:17.135 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:17.269 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/3stk P2:16%/1stk

23:15:17.510 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:15:17.524 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:15:17.651 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:15:17]   -- game boundary (game 12 starting)

23:15:17.855 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:15:18.991 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:15:20.052 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:15:21.196 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:15:22.307 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:15:23.352 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:13%/4stk

23:15:24.405 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:20%/4stk

23:15:25.458 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:46%/4stk
[23:15:26]   ep 123 [0-19%]: hits=2 dmg=45.8  (ref 3.9/27.0)  [game ep 123, handoff f98]

23:15:26.210 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:46%/4stk

23:15:26.339 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:46%/4stk

23:15:26.765 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:46%/3stk

23:15:27.835 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:15:28.938 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:15:30.120 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:15:31.271 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:2%/3stk

23:15:32.467 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:15:33.571 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:15:34.602 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:25%/3stk
[23:15:35]   ep 124 [0-19%]: hits=1 dmg=24.5  (ref 3.9/27.0)  [game ep 124, handoff f709]

23:15:35.663 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:15:36.693 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:15:37.766 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:25%/3stk

23:15:38.947 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:15:40.093 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:15:41.116 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:15:42.267 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:36%/3stk
[23:15:42]   ep 125 [20-39%]: hits=2 dmg=11.3  (ref 3.7/20.5)  [game ep 125, handoff f1113]

23:15:43.423 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:15:44.595 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:15:45.773 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:15:46.945 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:15:48.135 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:15:49.261 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:42%/3stk
[23:15:49]   ep 126 [20-39%]: hits=2 dmg=6.2  (ref 3.7/20.5)  [game ep 126, handoff f1468]

23:15:49.866 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:15:50.002 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:42%/3stk

23:15:50.528 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:42%/2stk

23:15:51.697 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:15:52.813 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:15:53.996 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:15:55.200 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:4%/2stk

23:15:56.355 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:15:57.660 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:15:59.147 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:6%/2stk
[23:16:00]   ep 127 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 127, handoff f2085]

23:16:00.633 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:01.801 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:02.880 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:03.960 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:05.071 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:06.146 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:07.304 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:08.446 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:09.442 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:10.485 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:11.625 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:12.677 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:13.816 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:14.893 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:15.948 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:16.946 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:17.977 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:19.030 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:20.052 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:21.171 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:6%/2stk
[23:16:21]     [reset] position timeout

23:16:22.027 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:22.157 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:16:22.739 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:16:23.833 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:16:24.909 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:16:26.115 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:16:27.352 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:5%/1stk

23:16:28.482 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:5%/1stk

23:16:29.584 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:5%/1stk

23:16:30.604 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:29%/1stk
[23:16:31]   ep 128 [0-19%]: hits=3 dmg=29.0  (ref 3.9/27.0)  [game ep 128, handoff f3882]

23:16:31.606 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:29%/1stk

23:16:32.683 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:29%/1stk

23:16:33.826 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:29%/1stk

23:16:34.861 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:29%/1stk

23:16:35.930 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:30%/1stk

23:16:37.103 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:38.227 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:39.364 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:36%/1stk
[23:16:40]   ep 129 [20-39%]: hits=2 dmg=6.6  (ref 3.7/20.5)  [game ep 129, handoff f4374]

23:16:40.493 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:41.615 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:42.731 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:43.865 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:44.996 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:46.093 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:47.202 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:48.348 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:49.513 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:50.676 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:51.841 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:52.980 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:54.124 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:55.167 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:56.291 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:57.370 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:58.443 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:16:59.446 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:17:00.434 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:17:01.543 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:36%/1stk
[23:17:02]     [reset] position timeout

23:17:02.608 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:17:02.736 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:36%/1stk

23:17:03.020 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:17:03.037 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:17:03.165 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:17:03]   -- game boundary (game 13 starting)

23:17:03.361 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:17:04.439 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:17:05.490 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:17:06.617 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:17:07.684 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:17:08.768 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:12%/4stk

23:17:09.778 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:12%/4stk

23:17:10.890 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:12%/4stk
[23:17:11]   ep 130 [0-19%]: hits=2 dmg=12.5  (ref 3.9/27.0)  [game ep 130, handoff f98]

23:17:11.940 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:12%/4stk

23:17:13.056 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:12%/4stk

23:17:14.197 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:17:15.344 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:17:16.532 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:17:17.661 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:19%/4stk
[23:17:18]   ep 131 [0-19%]: hits=2 dmg=6.9  (ref 3.9/27.0)  [game ep 131, handoff f448]

23:17:18.822 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:19%/4stk

23:17:19.900 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:21%/4stk

23:17:21.106 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:17:22.245 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:17:23.340 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:26%/4stk
[23:17:24]   ep 132 [0-19%]: hits=2 dmg=6.3  (ref 3.9/27.0)  [game ep 132, handoff f771]

23:17:24.475 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:17:25.616 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:17:26.802 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:31%/4stk

23:17:27.950 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:31%/4stk

23:17:29.063 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:31%/4stk

23:17:30.201 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:31%/4stk
[23:17:30]   ep 133 [20-39%]: hits=2 dmg=5.8  (ref 3.7/20.5)  [game ep 133, handoff f1088]

23:17:31.584 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:31%/4stk

23:17:32.709 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:31%/4stk

23:17:33.900 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:37%/4stk

23:17:35.069 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:36.244 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:37.399 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:51%/4stk
[23:17:37]   ep 134 [20-39%]: hits=2 dmg=19.2  (ref 3.7/20.5)  [game ep 134, handoff f1450]

23:17:37.681 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:37.812 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:37.944 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.076 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.207 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.336 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.464 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.592 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.720 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.850 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:38.979 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:39.108 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:51%/4stk

23:17:39.619 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:51%/3stk

23:17:40.635 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:17:41.663 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:17:42.774 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:17:43.948 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:17:45.032 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:17:46.068 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:47.392 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:49.086 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:50.617 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:6%/3stk
[23:17:51]   ep 135 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 135, handoff f2777]

23:17:52.219 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:53.430 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:54.603 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:55.724 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:56.792 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:57.786 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:58.783 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:17:59.789 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:00.879 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:01.905 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:03.040 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:04.145 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:05.264 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:06.382 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:07.485 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:08.684 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:09.837 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:10.973 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:12.104 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:13.262 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:6%/3stk
[23:18:13]     [reset] position timeout

23:18:13.718 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:13.847 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:18:14.871 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:6%/2stk

23:18:15.994 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:18:17.102 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:18:18.101 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:18:19.176 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:18:20.347 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:18:21.483 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:18:22.634 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:19%/2stk
[23:18:22]   ep 136 [0-19%]: hits=2 dmg=18.6  (ref 3.9/27.0)  [game ep 136, handoff f4573]

23:18:23.691 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:18:24.826 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:25.991 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:27.142 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:28.796 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:24%/2stk
[23:18:29]   ep 137 [0-19%]: hits=2 dmg=5.8  (ref 3.9/27.0)  [game ep 137, handoff f4900]

23:18:30.372 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:31.617 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:32.796 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:33.886 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:35.000 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:36.134 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:37.281 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:38.402 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:39.495 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:40.674 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:41.784 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:42.951 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:44.127 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:45.274 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:46.375 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:47.553 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:48.692 [info] [MeleePort] Frame 6120: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:49.798 [info] [MeleePort] Frame 6180: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:50.890 [info] [MeleePort] Frame 6240: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:51.866 [info] [MeleePort] Frame 6300: menu_state=2 | P1:0%/4stk P2:24%/2stk
[23:18:52]     [reset] position timeout

23:18:52.663 [info] [MeleePort] Frame 6360: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:52.793 [info] [MeleePort] Frame 6420: menu_state=2 | P1:0%/4stk P2:24%/2stk

23:18:53.438 [info] [MeleePort] Frame 6480: menu_state=2 | P1:0%/4stk P2:24%/1stk

23:18:54.592 [info] [MeleePort] Frame 6540: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:18:55.631 [info] [MeleePort] Frame 6600: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:18:56.631 [info] [MeleePort] Frame 6660: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:18:57.789 [info] [MeleePort] Frame 6720: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:18:58.912 [info] [MeleePort] Frame 6780: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:19:00.007 [info] [MeleePort] Frame 6840: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:19:01.217 [info] [MeleePort] Frame 6900: menu_state=2 | P1:0%/4stk P2:6%/1stk
[23:19:01]   ep 138 [0-19%]: hits=2 dmg=5.6  (ref 3.9/27.0)  [game ep 138, handoff f6696]

23:19:02.404 [info] [MeleePort] Frame 6960: menu_state=2 | P1:0%/4stk P2:6%/1stk

23:19:03.598 [info] [MeleePort] Frame 7020: menu_state=2 | P1:0%/4stk P2:11%/1stk

23:19:04.745 [info] [MeleePort] Frame 7080: menu_state=2 | P1:0%/4stk P2:24%/1stk

23:19:05.852 [info] [MeleePort] Frame 7140: menu_state=2 | P1:0%/4stk P2:24%/1stk

23:19:06.984 [info] [MeleePort] Frame 7200: menu_state=2 | P1:0%/3stk P2:24%/1stk
[23:19:07]   ep 139 [0-19%]: hits=2 dmg=18.3  (ref 3.9/27.0)  [game ep 139, handoff f6987]

23:19:07.976 [info] [MeleePort] Frame 7260: menu_state=2 | P1:0%/3stk P2:24%/1stk

23:19:09.110 [info] [MeleePort] Frame 7320: menu_state=2 | P1:0%/3stk P2:24%/1stk

23:19:10.307 [info] [MeleePort] Frame 7380: menu_state=2 | P1:0%/3stk P2:31%/1stk

23:19:11.461 [info] [MeleePort] Frame 7440: menu_state=2 | P1:0%/3stk P2:31%/1stk

23:19:12.549 [info] [MeleePort] Frame 7500: menu_state=2 | P1:0%/3stk P2:31%/1stk
[23:19:13]   ep 140 [20-39%]: hits=2 dmg=12.5  (ref 3.7/20.5)  [game ep 140, handoff f7319]

23:19:13.713 [info] [MeleePort] Frame 7560: menu_state=2 | P1:0%/3stk P2:36%/1stk

23:19:14.878 [info] [MeleePort] Frame 7620: menu_state=2 | P1:0%/3stk P2:36%/1stk

23:19:16.111 [info] [MeleePort] Frame 7680: menu_state=2 | P1:0%/3stk P2:43%/1stk

23:19:17.285 [info] [MeleePort] Frame 7740: menu_state=2 | P1:0%/3stk P2:48%/1stk

23:19:18.503 [info] [MeleePort] Frame 7800: menu_state=2 | P1:0%/3stk P2:53%/1stk
[23:19:19]   ep 141 [20-39%]: hits=3 dmg=16.4  (ref 3.7/20.5)  [game ep 141, handoff f7616]

23:19:19.721 [info] [MeleePort] Frame 7860: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:19.856 [info] [MeleePort] Frame 7920: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:19.992 [info] [MeleePort] Frame 7980: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.122 [info] [MeleePort] Frame 8040: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.253 [info] [MeleePort] Frame 8100: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.387 [info] [MeleePort] Frame 8160: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.522 [info] [MeleePort] Frame 8220: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.652 [info] [MeleePort] Frame 8280: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.786 [info] [MeleePort] Frame 8340: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:20.923 [info] [MeleePort] Frame 8400: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:21.064 [info] [MeleePort] Frame 8460: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:21.197 [info] [MeleePort] Frame 8520: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:21.343 [info] [MeleePort] Frame 8580: menu_state=2 | P1:0%/3stk P2:53%/1stk

23:19:21.594 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:19:21.607 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:19:21.733 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:19:21]   -- game boundary (game 14 starting)

23:19:21.901 [info] [MeleePort] Frame -120: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:19:23.032 [info] [MeleePort] Frame -60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:19:24.183 [info] [MeleePort] Frame 0: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:19:25.293 [info] [MeleePort] Frame 60: menu_state=2 | P1:0%/4stk P2:0%/4stk

23:19:26.402 [info] [MeleePort] Frame 120: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:19:27.500 [info] [MeleePort] Frame 180: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:19:28.637 [info] [MeleePort] Frame 240: menu_state=2 | P1:0%/4stk P2:7%/4stk

23:19:29.750 [info] [MeleePort] Frame 300: menu_state=2 | P1:0%/4stk P2:14%/4stk
[23:19:30]   ep 142 [0-19%]: hits=2 dmg=14.5  (ref 3.9/27.0)  [game ep 142, handoff f98]

23:19:31.222 [info] [MeleePort] Frame 360: menu_state=2 | P1:0%/4stk P2:14%/4stk

23:19:32.349 [info] [MeleePort] Frame 420: menu_state=2 | P1:0%/4stk P2:14%/4stk

23:19:33.462 [info] [MeleePort] Frame 480: menu_state=2 | P1:0%/4stk P2:14%/4stk

23:19:34.623 [info] [MeleePort] Frame 540: menu_state=2 | P1:0%/4stk P2:21%/4stk

23:19:35.773 [info] [MeleePort] Frame 600: menu_state=2 | P1:0%/4stk P2:21%/4stk

23:19:36.890 [info] [MeleePort] Frame 660: menu_state=2 | P1:0%/4stk P2:26%/4stk

23:19:38.071 [info] [MeleePort] Frame 720: menu_state=2 | P1:0%/4stk P2:31%/4stk
[23:19:38]   ep 143 [0-19%]: hits=3 dmg=21.5  (ref 3.9/27.0)  [game ep 143, handoff f496]

23:19:39.183 [info] [MeleePort] Frame 780: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:40.275 [info] [MeleePort] Frame 840: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:41.387 [info] [MeleePort] Frame 900: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:42.551 [info] [MeleePort] Frame 960: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:43.719 [info] [MeleePort] Frame 1020: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:44.843 [info] [MeleePort] Frame 1080: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:46.048 [info] [MeleePort] Frame 1140: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:47.254 [info] [MeleePort] Frame 1200: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:48.429 [info] [MeleePort] Frame 1260: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:49.610 [info] [MeleePort] Frame 1320: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:50.702 [info] [MeleePort] Frame 1380: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:51.829 [info] [MeleePort] Frame 1440: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:52.966 [info] [MeleePort] Frame 1500: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:54.018 [info] [MeleePort] Frame 1560: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:55.154 [info] [MeleePort] Frame 1620: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:56.319 [info] [MeleePort] Frame 1680: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:57.521 [info] [MeleePort] Frame 1740: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:58.678 [info] [MeleePort] Frame 1800: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:19:59.848 [info] [MeleePort] Frame 1860: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:20:01.015 [info] [MeleePort] Frame 1920: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:20:02.165 [info] [MeleePort] Frame 1980: menu_state=2 | P1:0%/4stk P2:36%/4stk
[23:20:02]     [reset] position timeout

23:20:02.348 [info] [MeleePort] Frame 2040: menu_state=2 | P1:0%/4stk P2:36%/4stk

23:20:02.709 [info] [MeleePort] Frame 2100: menu_state=2 | P1:0%/4stk P2:36%/3stk

23:20:03.798 [info] [MeleePort] Frame 2160: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:20:04.843 [info] [MeleePort] Frame 2220: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:20:06.064 [info] [MeleePort] Frame 2280: menu_state=2 | P1:0%/4stk P2:0%/3stk

23:20:07.253 [info] [MeleePort] Frame 2340: menu_state=2 | P1:0%/4stk P2:2%/3stk

23:20:08.445 [info] [MeleePort] Frame 2400: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:20:09.500 [info] [MeleePort] Frame 2460: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:20:10.578 [info] [MeleePort] Frame 2520: menu_state=2 | P1:0%/4stk P2:6%/3stk
[23:20:11]   ep 144 [0-19%]: hits=2 dmg=6.1  (ref 3.9/27.0)  [game ep 144, handoff f2335]

23:20:11.647 [info] [MeleePort] Frame 2580: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:20:12.755 [info] [MeleePort] Frame 2640: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:20:13.854 [info] [MeleePort] Frame 2700: menu_state=2 | P1:0%/4stk P2:6%/3stk

23:20:15.030 [info] [MeleePort] Frame 2760: menu_state=2 | P1:0%/4stk P2:12%/3stk

23:20:16.159 [info] [MeleePort] Frame 2820: menu_state=2 | P1:0%/4stk P2:17%/3stk

23:20:17.312 [info] [MeleePort] Frame 2880: menu_state=2 | P1:0%/4stk P2:20%/3stk

23:20:18.374 [info] [MeleePort] Frame 2940: menu_state=2 | P1:0%/4stk P2:28%/3stk
[23:20:18]   ep 145 [0-19%]: hits=3 dmg=23.4  (ref 3.9/27.0)  [game ep 145, handoff f2723]

23:20:19.502 [info] [MeleePort] Frame 3000: menu_state=2 | P1:0%/4stk P2:31%/3stk

23:20:20.653 [info] [MeleePort] Frame 3060: menu_state=2 | P1:0%/4stk P2:38%/3stk

23:20:21.893 [info] [MeleePort] Frame 3120: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:23.106 [info] [MeleePort] Frame 3180: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:24.313 [info] [MeleePort] Frame 3240: menu_state=2 | P1:0%/4stk P2:52%/3stk
[23:20:24]   ep 146 [20-39%]: hits=2 dmg=21.2  (ref 3.7/20.5)  [game ep 146, handoff f3017]

23:20:24.741 [info] [MeleePort] Frame 3300: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:24.877 [info] [MeleePort] Frame 3360: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.009 [info] [MeleePort] Frame 3420: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.143 [info] [MeleePort] Frame 3480: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.274 [info] [MeleePort] Frame 3540: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.404 [info] [MeleePort] Frame 3600: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.540 [info] [MeleePort] Frame 3660: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.685 [info] [MeleePort] Frame 3720: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.826 [info] [MeleePort] Frame 3780: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:25.960 [info] [MeleePort] Frame 3840: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:26.088 [info] [MeleePort] Frame 3900: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:26.224 [info] [MeleePort] Frame 3960: menu_state=2 | P1:0%/4stk P2:52%/3stk

23:20:26.682 [info] [MeleePort] Frame 4020: menu_state=2 | P1:0%/4stk P2:52%/2stk

23:20:27.832 [info] [MeleePort] Frame 4080: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:20:29.026 [info] [MeleePort] Frame 4140: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:20:30.202 [info] [MeleePort] Frame 4200: menu_state=2 | P1:0%/4stk P2:0%/2stk

23:20:31.383 [info] [MeleePort] Frame 4260: menu_state=2 | P1:0%/4stk P2:2%/2stk

23:20:32.637 [info] [MeleePort] Frame 4320: menu_state=2 | P1:0%/4stk P2:7%/2stk

23:20:33.775 [info] [MeleePort] Frame 4380: menu_state=2 | P1:0%/4stk P2:7%/2stk

23:20:34.839 [info] [MeleePort] Frame 4440: menu_state=2 | P1:0%/4stk P2:7%/2stk
[23:20:35]   ep 147 [0-19%]: hits=2 dmg=7.0  (ref 3.9/27.0)  [game ep 147, handoff f4250]

23:20:35.966 [info] [MeleePort] Frame 4500: menu_state=2 | P1:0%/4stk P2:7%/2stk

23:20:37.102 [info] [MeleePort] Frame 4560: menu_state=2 | P1:0%/4stk P2:7%/2stk

23:20:38.326 [info] [MeleePort] Frame 4620: menu_state=2 | P1:0%/4stk P2:13%/2stk

23:20:39.474 [info] [MeleePort] Frame 4680: menu_state=2 | P1:0%/4stk P2:13%/2stk

23:20:40.598 [info] [MeleePort] Frame 4740: menu_state=2 | P1:0%/4stk P2:13%/2stk

23:20:41.716 [info] [MeleePort] Frame 4800: menu_state=2 | P1:0%/4stk P2:13%/2stk
[23:20:42]   ep 148 [0-19%]: hits=2 dmg=6.4  (ref 3.9/27.0)  [game ep 148, handoff f4596]

23:20:42.847 [info] [MeleePort] Frame 4860: menu_state=2 | P1:0%/4stk P2:13%/2stk

23:20:44.023 [info] [MeleePort] Frame 4920: menu_state=2 | P1:0%/4stk P2:13%/2stk

23:20:45.205 [info] [MeleePort] Frame 4980: menu_state=2 | P1:0%/4stk P2:18%/2stk

23:20:46.397 [info] [MeleePort] Frame 5040: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:20:47.542 [info] [MeleePort] Frame 5100: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:20:48.555 [info] [MeleePort] Frame 5160: menu_state=2 | P1:0%/4stk P2:19%/2stk
[23:20:49]   ep 149 [0-19%]: hits=2 dmg=5.9  (ref 3.9/27.0)  [game ep 149, handoff f4965]

23:20:49.831 [info] [MeleePort] Frame 5220: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:20:50.959 [info] [MeleePort] Frame 5280: menu_state=2 | P1:0%/4stk P2:19%/2stk

23:20:52.017 [info] [MeleePort] Frame 5340: menu_state=2 | P1:0%/4stk P2:43%/2stk

23:20:53.099 [info] [MeleePort] Frame 5400: menu_state=2 | P1:0%/4stk P2:43%/2stk

23:20:54.231 [info] [MeleePort] Frame 5460: menu_state=2 | P1:0%/4stk P2:48%/2stk

23:20:55.327 [info] [MeleePort] Frame 5520: menu_state=2 | P1:0%/4stk P2:48%/2stk
[23:20:55]   ep 150 [0-19%]: hits=2 dmg=29.0  (ref 3.9/27.0)  [game ep 150, handoff f5294]

23:20:55.688 [info] [MeleePort] Frame 5580: menu_state=2 | P1:0%/4stk P2:48%/2stk

23:20:55.824 [info] [MeleePort] Frame 5640: menu_state=2 | P1:0%/4stk P2:48%/1stk

23:20:55.948 [info] [MeleePort] Frame 5700: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.073 [info] [MeleePort] Frame 5760: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.195 [info] [MeleePort] Frame 5820: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.318 [info] [MeleePort] Frame 5880: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.452 [info] [MeleePort] Frame 5940: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.584 [info] [MeleePort] Frame 6000: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.713 [info] [MeleePort] Frame 6060: menu_state=2 | P1:0%/4stk P2:0%/1stk

23:20:56.919 [info] [MeleePort] Frame 0: menu_state=0 | P1:0%/0stk P2:0%/0stk

23:20:56.932 [info] [MeleePort] Frame 0: menu_state=1 | P1:0%/0stk P2:0%/0stk

23:20:57.059 [info] [MeleePort] Frame 60: menu_state=1 | P1:0%/0stk P2:0%/0stk
[23:20:57]   -- game boundary (game 15 starting)
[23:20:57] [32m✓ console: 150 episodes across 15 games, 1487.6s[0m
[23:20:57] 
[23:20:57] == smoke summary (live counter; authoritative = drill_score.exs on the bank)
[23:20:57]    0-19%: n=88  hits 2.0  >=3 9%  dmg 17.6   (expert n=39: 3.9 / 87% / 27.0)
[23:20:57]    20-39%: n=62  hits 2.1  >=3 6%  dmg 12.3   (expert n=24: 3.7 / 83% / 20.5)
[23:20:57] [32m✓ bank -> /home/blewf/git/exphil/eval_runs/0902_awbc_drill/drill_remeasure (/home/blewf/git/exphil/eval_runs/0902_awbc_drill/drill_remeasure/episodes.jsonl)[0m
[23:20:57] [33m⚠️  Killed 1 orphaned Dolphin(s)[0m
=== drill re-measure end 2026-09-02T23:20:57-05:00
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


23:20:58.057 [info] [ExPhil] Application started

23:20:58.058 [debug] [ExPhil] Supervision tree: [{ExPhil.Training.AsyncCheckpoint, #PID<0.285.0>, :worker, [ExPhil.Training.AsyncCheckpoint]}, {ExPhil.Telemetry.Collector, #PID<0.284.0>, :worker, [ExPhil.Telemetry.Collector]}, {ExPhil.Agents.Supervisor, #PID<0.283.0>, :supervisor, [ExPhil.Agents.Supervisor]}, {ExPhil.Bridge.Supervisor, #PID<0.282.0>, :supervisor, [ExPhil.Bridge.Supervisor]}, {ExPhil.Registry, #PID<0.280.0>, :supervisor, [Registry]}]

╔════════════════════════════════════════════════════════════╗
║                     Drill bank scorer                      ║
╚════════════════════════════════════════════════════════════╝

[23:20:58]   150 episodes across 14 replays

# Drill bank score — eval_runs/0902_awbc_drill/drill_remeasure (uthrow_low_mid)

150 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 113/150.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot 0-19% | 88 | 3.7 | 41 | 17.5 | 0 |
| expert 0-19% | 39 | 3.9 | 87 | 27.0 | 0 |
| bot 20-39% | 62 | 2.5 | 23 | 12.3 | 0 |
| expert 20-39% | 24 | 3.7 | 83 | 20.5 | 0 |


0-19% hits histogram: 1:23  2:29  3:9  4:1  6:1  7:2  8:17  9:5  10:1


20-39% hits histogram: 1:28  2:20  3:5  7:4  8:4  10:1


[23:20:58] [32m✓ wrote eval_runs/0902_awbc_drill/drill_remeasure/RESULTS.md[0m
AWBC DRILL CHAIN DONE 2026-09-02T23:20:58-05:00 — compare eval_runs/0902_awbc_drill/drill_remeasure/RESULTS.md vs the 538-ep baseline (2.9/30/16.4), then live-look transfer (g6 rule)
