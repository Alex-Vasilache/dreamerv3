# Handoff, Wed 11-11 noon CET (overnight 11-10 to 11-11)

Details and tables: `SAMPLE_EFF_STUDY.md` (sections from "Tue 11-10 night: M").
All sim; no robot or phone actions were taken.

## Results

- **M** (dropout x rollouts, with schedule, n = 36): world-model dropout 0.05
  is the one consistent gain, with late crossings at 1.6 vs 2.0 and no
  speed cost. Two rollouts per start is slower.
- **N** (IAS, no ratio schedule, n = 8): uncapped drifts late (2.6 vs 1.7
  with the schedule). Uncapped plus dropout 0.05 and 2 rollouts gives 1.6,
  with t90 2.6, at a ratio of about 250.
- **P** (learning-progress ratio control, Saion n = 16 + IAS n = 8): it does
  not help. Held-out world-model loss stops improving after 15-20 min, so the
  controller pins the ratio at its floor (32-40). Late drift is worse on
  Saion (3.6 vs 1.8) and equal on IAS (2.3). The code stays in, off by
  default (`online_progress.every 0`, `replay.holdout 0`).
- **P on IAS also weakens the case for any ratio control.** Late ratios of
  290, 111 and 44 gave the same late level (2.2-2.3). Pooled IAS uncapped vs
  schedule is 2.5 vs 2.0, within one SD.
- **Q** (IAS array 205138, fresh seeds, 20/24 done) confirms N. Pooled N + Q
  (n = 14-15): uncapped + dropout 0.05 + 2 rollouts has late 1.5 vs 1.7
  with the schedule (p = 0.20) and 2.5 uncapped (p = 0.007), and is faster
  than the schedule (t90 2.6 vs 3.0, p = 0.04).

## Recommended recipe (sim, confirmed; updated Thu 11-12)

This is the pick without the ratio schedule, plus `--agent.dyn.rssm.drop 0.05`.
Round S2 showed the second rollout is not needed: dropout alone is calmer
late on IAS (0.9 vs 1.5) and does not slow P100. Now the robot default.

Next: try it live.

## To do by hand

- Live command: drop `--online_ratio_switch 19500 --online_ratio_late 128`
  (WEEKEND_1107.md) on the next live run; the default no longer needs them.
- Face: push to the phone (`push_python.sh` + `adb push face.html`).
