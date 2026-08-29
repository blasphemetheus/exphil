arms started 2026-08-28T18:22:27-05:00 resume=checkpoints/fox_gen_v1_20260825_210355_epoch10.axon epochs=3 seed=828
=== B1 start 2026-08-28T18:22:27-05:00
=== B1 end 2026-08-28T18:23:50-05:00 rc=1
=== B2 start 2026-08-28T18:23:50-05:00

(false start 18:22: B1 died in EXLA compile — --stage-internals was missing, checkpoint 296 dims vs 288; chain relaunched with the flag)
arms started 2026-08-28T18:25:31-05:00 resume=checkpoints/fox_gen_v1_20260825_210355_epoch10.axon epochs=3 seed=828
=== B1 start 2026-08-28T18:25:31-05:00
=== B1 end 2026-08-28T21:45:49-05:00 rc=0
=== B2 start 2026-08-28T21:45:49-05:00
=== B2 end 2026-08-29T04:27:07-05:00 rc=0
!!! B2: no 'awbc' text in logs/awbc_B2.log — flag may have been dropped
=== B3 start 2026-08-29T04:27:07-05:00

(04:27 knob-assertion "!!! B2: no awbc text" is a FALSE ALARM: train.exs never prints the word. Verified from checkpoints/fox_gen_v1_B2_20260829_024551_config.json: awbc=true, awbc_reward=standard, awbc_shuffle=false (B1 config: awbc=false). B2 val 5.7152 -> 5.6941 -> 5.6747. Fix the assertion to read the saved _config.json AFTER the chain ends — never edit a running bash script.)
