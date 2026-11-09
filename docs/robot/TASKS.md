# Robot task list

Kept current by the working session; newest state at the top of each section.
Status: [ ] open · [~] running · [x] done. Plan and results: `PLAN.md`,
`SAMPLE_EFF_STUDY.md`, `WEEKEND_1107.md`.

## Now: sample-efficiency study (`SAMPLE_EFF_STUDY.md`)

- [x] Weekend sweeps documented; report page; live flags corrected (11-09)
- [x] IAS cleanup (runs archived to bucket, ~1.1M -> 80k files); IAS sims write to job /tmp, 5 GB
- [x] Lit reviews: zero-shot (`LIT_ZERO_SHOT.md`), augmentation + DreamerV4 (`LIT_AUGMENT_DV4.md`), fact-checked
- [x] Code: component freezes, world-model dropout, observational dropout, N rollouts per start; smoke-tested
- [~] Step 0: live runs of the weekend pick (e43457, e43606, e43607) vs e40006 -- first read done, needs repeats
- [x] Z2 (IAS): world model is the early bottleneck (trained WM: 0.66 upright in min 1; trained AC on fresh WM: t90 10 min vs 2.9 fresh)
- [x] Z1 (Saion V100): no late drift with the full pick even uncapped (2.0 vs 1.6 crossings/min); freezing either part at min 5 is much worse (22 / 61)
- [x] U (Saion P100): one rollout per context is not better (t90 3.6 vs 3.0 min; late 4.8 vs 2.2 crossings/min); not adopted
- [x] Y (Saion V100): world-model dropout does not speed up the start; slightly calmer late (1.1-1.7 vs 2.0 crossings/min, within SD); candidate for step 6
- [~] X (IAS): observational dropout 0.1 / 0.3 -- 33/40 lost to the time limit, rerun (202005)
- [~] V (IAS): N = 2 / 4 rollouts per start (202006)
- [ ] Step 5: winners on commands; best one live
- [ ] Step 6: reanalysis (what worked, what did not, why), combine the parts that help, confirm n = 12 at three GPU speeds
- [x] Saion checkout unpinned: fast-forwarded to include Alex's 2cc011e/1a68792 (11-10)

## Next: face visualisation (after the study settles)

- [ ] Write-up: concise maths of how emotions and pain are computed and how they drive the face
- [ ] Faster reactions: higher eye-animation rate; remove the lag between a bumper hit and the face
- [ ] Pain: normal colour (drop the red), keep the shape change, more curved top of the eyes
- [ ] Effort in the eyes from action intensity
- [ ] Test each on the sim phone and live

## Later (`PLAN.md`)

- [ ] Intrinsic motivation and emotion equivalents
- [ ] Neuroscience literature on emotion and learning; emotions to speed up learning
