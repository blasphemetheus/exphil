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
=== B3 end 2026-08-29T10:59:25-05:00 rc=0
!!! B3: no 'awbc' text in logs/awbc_B3.log — flag may have been dropped
=== all arms done 2026-08-29T10:59:25-05:00
-rw-r--r--  1 blewf users 36113232 Aug 28 21:45 fox_gen_v1_B1_20260828_232532.axon
-rw-r--r--  1 blewf users 36113232 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_best.axon
-rw-r--r--  1 blewf users 12036178 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_best_policy.bin
-rw-r--r--  1 blewf users     3662 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_config.json
-rw-r--r--  1 blewf users 36113232 Aug 28 20:31 fox_gen_v1_B1_20260828_232532_epoch1.axon
-rw-r--r--  1 blewf users 36113232 Aug 28 21:08 fox_gen_v1_B1_20260828_232532_epoch2.axon
-rw-r--r--  1 blewf users 36113232 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_epoch3.axon
-rw-r--r--  1 blewf users     2020 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_loss.html
-rw-r--r--  1 blewf users 12036178 Aug 28 21:45 fox_gen_v1_B1_20260828_232532_policy.bin
-rw-r--r--  1 blewf users 36113232 Aug 29 04:27 fox_gen_v1_B2_20260829_024551.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_best.axon
-rw-r--r--  1 blewf users 12036178 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_best_policy.bin
-rw-r--r--  1 blewf users     3664 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_config.json
-rw-r--r--  1 blewf users 36113232 Aug 29 00:00 fox_gen_v1_B2_20260829_024551_epoch1.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 02:13 fox_gen_v1_B2_20260829_024551_epoch2.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_epoch3.axon
-rw-r--r--  1 blewf users     2020 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_loss.html
-rw-r--r--  1 blewf users 12036178 Aug 29 04:27 fox_gen_v1_B2_20260829_024551_policy.bin
-rw-r--r--  1 blewf users 36113232 Aug 29 10:59 fox_gen_v1_B3_20260829_092709.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_best.axon
-rw-r--r--  1 blewf users 12036178 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_best_policy.bin
-rw-r--r--  1 blewf users     3663 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_config.json
-rw-r--r--  1 blewf users 36113232 Aug 29 06:39 fox_gen_v1_B3_20260829_092709_epoch1.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 08:49 fox_gen_v1_B3_20260829_092709_epoch2.axon
-rw-r--r--  1 blewf users 36113232 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_epoch3.axon
-rw-r--r--  1 blewf users     2020 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_loss.html
-rw-r--r--  1 blewf users 12036178 Aug 29 10:59 fox_gen_v1_B3_20260829_092709_policy.bin

(10:59 B3 "no awbc text" = the same FALSE ALARM. Verified from checkpoints/fox_gen_v1_B3_20260829_092709_config.json: awbc=true, awbc_shuffle=true, awbc_reward=standard. Val loss B1 5.7688->5.7502->5.7328 | B2 5.7152->5.6941->5.6747 | B3 5.8061->5.7862->5.7694 — recorded, NOT the verdict; no divergence, no arm disqualified. Assertion fixed in scripts/awbc_arms.sh to read the saved _config.json (task 20). Scoring launched 08-29 ~11:05 as unit awbc-score -> eval_runs/0828_awbc_arms/score/score_table.txt.)

(11:35 score B1 "KNOB ASSERTION FAILED" is FALSE: the banner line 'buttons: 0.5' is in all 8 B1 run logs. Cause = 'sed | grep -q' under set -o pipefail: grep -q exits at the first match, sed takes SIGPIPE (141), pipefail reports failure — reproduced 21/30 on the same input. B1 replays are intact and are scored by the axis globs; read B1 normally. Fix = grep -q on a process substitution (applied to argmax_buttons_bracket.sh now; awbc_score.sh after the unit exits). GOTCHA #104.)
