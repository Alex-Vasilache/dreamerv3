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
- [~] U (Saion P100): DreamerV4 one rollout per context (imag_last 1 / 8, batch 64)
- [ ] Y (Saion V100, after Z1): world-model dropout 0.05 / 0.1 / 0.2
- [~] X (IAS, after Z2): observational dropout 0.1 / 0.3
- [ ] V (IAS, after X): N = 2 / 4 rollouts per start
- [ ] Step 5: winners on commands; best one live
- [ ] Step 6: reanalysis (what worked, what did not, why), combine the parts that help, confirm n = 12 at three GPU speeds
- [ ] After Z/Y run starts: pull Alex's 2cc011e/1a68792 into the Saion checkout (pinned until then)

## Next: face visualisation (after the study settles)

- [ ] Write-up: concise maths of how emotions and pain are computed and how they drive the face
- [ ] Faster reactions: higher eye-animation rate; remove the lag between a bumper hit and the face
- [ ] Pain: normal colour (drop the red), keep the shape change, more curved top of the eyes
- [ ] Effort in the eyes from action intensity
- [ ] Test each on the sim phone and live

## Later (`PLAN.md`)

- [ ] Intrinsic motivation and emotion equivalents
- [ ] Neuroscience literature on emotion and learning; emotions to speed up learning
