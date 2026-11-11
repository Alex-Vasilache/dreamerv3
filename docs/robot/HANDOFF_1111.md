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
- **Q** (IAS array 205138, e44420-e44443, fresh seeds, n = 8) is running and
  finishes around noon. It compares uncapped, schedule, and uncapped plus
  dropout 0.05 and 2 rollouts. Analyse with `scan3.py`, arms in that order
  per seed.

## Recommended recipe (sim), pending Q

This is the pick without the ratio schedule, plus
`--agent.dyn.rssm.drop 0.05 --agent.imag_repeat 2`.

If Q does not reproduce N, use the pick uncapped plus dropout 0.05 alone. In
that case the schedule is not worth keeping either.

## To do by hand

- `git push` from the robot checkout. About 22 local commits, including the
  face-rework merge, rounds M/N/P/Q and progress control.
- Face: push to the phone (`push_python.sh` + `adb push face.html`).
- IAS: the runs of rounds M, N, P and Q are still in `robot_runs`. They are
  not archived and not deleted. Saion runs of M and P are archived to the
  bucket.
