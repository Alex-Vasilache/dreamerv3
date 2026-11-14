# Robot task list

Kept current by the working session; newest state at the top of each section.
Status: [ ] open · [~] running · [x] done. Plan and results: `PLAN.md`,
`SAMPLE_EFF_STUDY.md`, `WEEKEND_1107.md`.

## Now: emotion model, step by step (plan after Jan, 11-13; `EMOTION.md`, `LIT_AROUSAL_YU.md`, `../../rutledge_rl`)

1. Validate the RL correlates on human data
   - [x] Rutledge task, Jangraw 2023 OSF data (n = 93 / 1,829): equal weights fit worse; r + gamma'(V' - V), gamma' ~0.5-0.6, matches Rutledge; EV is value not advantage; V(s) fits as well as Q
   - [x] Bootstrapper of bootstrappers (1B): already published as kappa-value iteration (Efroni 2018; Tomar 2020); write a paragraph only
   - [~] Re-fit on Rutledge's own GBE app data (Dryad, 47k participants)
   - [ ] Decide pleasure: gamma' value; optional regret term A_max from the dominance rollouts (if the GBE fit supports it)
2. Arousal: competing definitions
   - [x] Angela Yu's work: base-rate surprise on threat beats current versions in 48 sim runs (0.69 / 0.71 vs <= 0.64 / 0.62)
   - [~] Survey of competing definitions (salience, volatility, free energy, stakes, urgency, learning progress, ...), shortlist
   - [ ] Offline bake-off of the shortlist on the 48 logged runs (hits, before hits, pushes)
   - [ ] Questions for the meeting with Angela Yu (in `LIT_AROUSAL_YU.md`)
3. Sim round with the best current definitions
   - [ ] Build behind switches: pleasure gamma' (+ regret), arousal winner(s) incl. Yu's exact version (log posterior/prior pain probability), dominance updated more often
   - [ ] Run: pain/threat as decided (impact low threshold vs bumper), reflex s4z2; n = 8, 60 min, pushes and walls; score every axis at hits, near misses, pushes
   - [ ] Update `EMOTION.md`, `FACE_MATHS.md`, plan page
4. Do emotions help learning?
   - [ ] Literature: which emotions influence learning in humans and how (learning rate, replay/consolidation, exploration, attention, memory), with verified citations
   - [ ] Turn the strongest into robot mechanisms (e.g. arousal -> learning rate or replay priority; pleasure/regret -> exploration), each a switch
   - [ ] Test in sim against the current recipe (time to balance, late stability, pushes); best ones live only with Alex
5. Later (from Jan's list): emotions arise from learning (evolution strategies on a shaping function); emotions in agentic AI; meeting Daniel in December

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
- [x] X (IAS): observational dropout 0.1 neutral, 0.3 slower (+1 min) and restless; not adopted. Schedule helps late on IAS (1.3 vs 2.4)
- [x] V (IAS): N = 4 slower start; N = 2 within noise early, calmer late uncapped; weak candidate
- [~] Step 6a: reanalysis logged; round M = dropout 0.05 x 2 rollouts on the pick, n = 12, V100 e44000-47 / P100 e44050-97 / IAS e44100-47 (204440)
- [ ] Step 5 (after M): M's pick on commands; best one live
- [ ] Step 6b: analyse M; final recipe; if nothing beyond noise, next lever = world-model data efficiency in the first 2 min (wm_ratio)
- [x] Saion checkout unpinned: fast-forwarded to include Alex's 2cc011e/1a68792 (11-10)

## Next: face visualisation (after the study settles)

- [x] Write-up: `FACE_MATHS.md` (signals, pain, axes, delivery, eye mapping, latency budget)
- [x] Faster reactions: pain every control step on the phone; page 60 fps, pain step to 90 % in ~95 ms (was ~200 ms); faster blinks and darts
- [x] Pain: cyan; squint morphs into chevrons `> <` (rounded top read happy, replaced 11-10)
- [x] Effort from mean |applied wheel command|: flat lowered upper lid, narrower, brighter, tremor near full drive
- [~] Tested in headless Chromium + unit test (branch face-rework); merge after round M's Saion jobs have started (sim_phone imports policy_runner); then push to the phone and check live

## Later (`PLAN.md`)

- [ ] Intrinsic motivation and emotion equivalents
- [ ] Neuroscience literature on emotion and learning; emotions to speed up learning
