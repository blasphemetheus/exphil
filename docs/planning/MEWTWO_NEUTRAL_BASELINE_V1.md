# First controlled Mewtwo neutral baseline

This is an initial learned drill, not a claim of winning neutral against people.
First approve the teacher, then fit the model, then evaluate the learned policy.

## Data and decoding, declared before fitting

- Teacher: `MewtwoNeutralTeacher`, with live both-facing trials on FD against Fox.
- Training: the sixteen approved `teacher_v2` controlled sequences, including
  only the initial neutral exchange and completion of its initiating technique.
  Labels are actual next-frame controller readback, not relabeled student actions.
- Prefix observations supply history but receive no training targets.
- All five human recordings remain excluded from fitting this first scripted
  baseline. Preserve them as whole games for later demonstration-informed rounds.
- Small GRU, hidden size 64, two layers, window 16, zero recurrent initialization,
  f32, autoregressive action head, previous action, queue depth 1, reaction delay 0.
- Primary decoding: sampled, temperature 1.0. No deterministic promotion shortcut.
- Initial fit: 80 epochs. Training loss alone cannot qualify the model.

## Qualification after fitting

Use fresh ordinary-start games, rather than the teacher's prepared starting states:
three 30-second games against standing Fox and three against CPU level 6, on FD.
Use the normal synchronous play runner. Record latency, frame coverage, inputs,
actual openings, attack onsets, short hops, wavedashes, shield holds and deaths.

Promote as a first playable neutral drill only if:

1. All six runs deliver inputs at measured latency 1 without inference errors.
2. Every standing-Fox run produces an actual opening; CPU games produce at least
   one opening each. Report losses and trades too, without counting combo hits.
3. Across the runs, down tilt, grab, aerials, actual short hops and actual
   wavedashes occur. Missing options are a failure of the requested behavior set.
4. No uninterrupted shield hold exceeds 120 frames; no unforced SD occurs in the
   standing-Fox runs. CPU-caused stock losses remain visible in the report.
5. The graphical runner sustains ordinary local speed before recommending recording.

Failure retains an experimental checkpoint and identifies the missing behavior.
It does not authorize silently weakening these criteria or changing decoding.
Combo routes remain outside this round.

## Round 2, declared before fitting

Round 1 failed: zero openings in the standing games, zero wavedashes, and only
four Mewtwo openings versus twenty-two Fox openings across the CPU games.
All six sessions delivered inputs at latency 1. The three standing runs repeated
the same sampled trajectory because VM-local seed counters restarted; they are
not three independent reliability measurements. Scorer initialization was fixed
and the original replays were rescored without rerunning games.

Round 2 trains from scratch for 160 epochs on 4,899 frames in 42 approved
sequences. Add the individually approved cases from `shift20_v1` and `shift40_v1`
(Fox positioned 20/40 units from center, mirrored; the latter includes the normal
80-unit starting gap). Two requested-nair cases produced fair and are explicitly
excluded. Original failures remain in their manifests. Both aerials have separate
approved contact/landing tests; approval does not transfer to the failed cases.

Use Peppi's next-frame **physical** inputs. The live snapshot exposes processed
trigger values (digital L becomes 1.0; Z becomes 0.35), which should not become
analog-trigger training targets. The round-1 dataset remains preserved.

The sampler now seeds unkeyed draws from OS randomness, so fresh processes do
not restart an identical trajectory. Explicit keys in sampling-analysis APIs
remain reproducible. Repeat the same six-game sampled qualification and criteria.
The human recordings remain excluded from this scripted round.

## Round 3, declared before fitting

Round 2 learned all requested move categories and opened up standing Fox, but
failed the standing-SD gate and had one CPU run without an opening. Keep it as
an experimental checkpoint. Add twelve approved corner-return and turnaround
sequences (`corner75_v1`, `corner84_v1`, `corner125_v1`, `corner134_v1`), including
starts near both edges with Fox both near center and far across the stage.

Also include nineteen audited successful neutral-opening clips from the user's
whole FD game. Four grabs are supported by capture states; six down tilts, seven
aerials and two other ground attacks have an active attack and simultaneous
hitlag/damage. Attribution is inferred from those states. One unclear opener is
excluded. Targets start with the clean neutral lead-in and stop before the first
contact state, excluding combo continuation. The remaining four whole human
games stay held out; the explicit split is `v3/human_split.json`.

The audit exposed a scorer bug: a long throw animation could satisfy the neutral
lead-in and count throw damage as a second opening. Captures now include high
and low grabs; holders, captives and thrown states cannot start a fresh exchange.
The long-throw regression passes, and prior qualifications are rescored from the
preserved replays. Original reports remain available.

Fit from scratch for 240 epochs on 8,778 frames in 73 sequences, with the same
model size, physical input convention, sampled decoding and six-game criteria.

## Round 4, declared before fitting

Round 3 passed movement coverage and avoided standing-Fox SDs, but produced no
scored first neutral openings in any of the three CPU games. Counter-hits do not
satisfy that requirement. The scripted teacher subsequently produced openings
in all nine ordinary four-stock CPU-6 games: 26 Mewtwo openings, 20 Fox openings,
and one trade. These are development examples, not student qualification.

Add the 26 attributable successful teacher opening clips (2,374 frames). Include
real negative countdown frames as input-only context so windowing does not drop
the opening frame's jump input. Targets stop before first contact.

The isolated contact fixtures used one stock, whereas ordinary matches start
with four; stock count is a model input. Augment each whole scripted sequence
with own stock counts 1..4 and rotated opponent counts. The teacher never reads
stock count, and the exporter rejects sequences with stock transitions. Preserve
all physical input labels. Do not augment human decisions.

Fit from scratch for 160 epochs on 36,484 augmented scripted frames plus the
unchanged 2,031 human FD frames: 38,515 frames in 339 sequences. Keep the same
64-unit model, sampled temperature 1 decoding, and six-game promotion criteria.
The four other human games remain held out. Record the exact source hashes and
arguments alongside the checkpoint.

## Round 5, declared before fitting

Round 4 created two openings in each of two CPU games, but none in the third;
one standing game also had none. All movement categories, shield duration,
standing survival and input delivery passed. The standing failure repeatedly
down-tilted at about 27 units, beyond its effective range. Preserve the failure.

Success-only CPU clips omit most interrupted approaches and post-contact teacher
behavior. Round 5 instead includes the complete 30-second scored portions of the
nine validated CPU teacher games, including neutral losses, hit reactions and
resumption. No finalization inputs are included. The teacher has no combo-routing
policy and no offstage recovery; complete traces retain those limitations.

Also validate the teacher from the ordinary standing-Fox start: three games each
produce three openings with no Mewtwo stock loss. These repeatable passive trials
are execution evidence, not independent evidence of opponent generalization.
Include their complete scored traces to cover the student's too-far down-tilt
failure in the actual starting positions.

Use 26,988 stock-augmented controlled-contact frames, 16,380 CPU frames, 5,460
standing frames, and the unchanged 2,031 human FD frames: 50,859 frames in 247
sequences. Whole live games retain real stock transitions and are not augmented.
Fit from scratch for 160 epochs with hidden size 128; this doubles recurrent
width for the broader state coverage. Other model conventions, temperature 1
sampling, human split and promotion criteria remain unchanged. This is a combined
data/capacity experiment, not an isolated causal test of either change.

Round 5 result: standing openings 5/2/0 with stock losses 1/0/1; CPU openings
0/1/1 against Fox's 2/3/4. All sessions delivered inputs at latency 1, with all
requested movement/attack categories represented and no prolonged shield hold.
Opening consistency and standing survival failed. Best weighted training loss
was 0.0072696; this did not establish closed-loop reliability. Retain the model
as experimental, without graphical promotion. Next work is validated corrective
teacher rollout from student-visited states rather than another untargeted fit.
