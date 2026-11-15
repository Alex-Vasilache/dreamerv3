# Handoff, Mon 11-16 noon CET (weekend 11-14 to 11-16)

Draft, updated as results land. Progress page (plain language, figures):
claude.ai artifact "Emotion Research Progress". Task list: `TASKS.md` (top).
Details and tables: `SAMPLE_EFF_STUDY.md` (sections dated 11-14/11-15).

## Needs you first

- Phone app copies of `emotion.py` / `policy_runner.py` are older than the
  repo; sync before the next live run. The robot default arousal stays `kl`
  until then; the new switches (`arousal_mode pearce_hall_now`,
  `dominance_mode evidence`) are off by default.
- Decide: pleasure written with EV = V(s_t), i.e. r_t + γ(V_{t+1} − V_t)?
  (current code already computes o_t + O_{t+1} − O_t; the open point is only
  the γ and the wording).
- The /apps disk issue is solved (Ligneul clipboard copy deleted with your
  OK; behaviour + stress data live on /work, /apps at 76 %).

## Results (plan steps 1-4)

1. **Pleasure** fits human happiness: on Rutledge's app data (n = 42,121)
   the weight-free r + V' − V predicts the next session best (r = .51 vs
   Rutledge's .38). EV is value, not advantage.
2. **Arousal**: Pearce-Hall associability (running average of |TD error|)
   wins the sim bake-off; human pupil data favour the *current* surprise, and
   including the current |δ| is best in sim too (AUC 0.85). Built as
   `arousal_mode pearce_hall_now`.
   **Dominance**: human control ratings follow a Bayesian control estimate
   best, empowerment second; Ligneul data favour empowerment averaged over the
   situation. Functional test (round DC, sim control cuts): empowerment barely
   falls (AUC 0.55) because it is computed inside a world model that has not
   seen the weak wheels; arousal (0.83) and pleasure (0.64) react. Offline,
   a model-free evidence measure (command→tilt slope) does better (0.58-0.63).
   Round DE (8 new runs): the world-model evidence switch (`dominance_mode
   evidence`) does not help (0.52); the model-free command→tilt slope
   replicates (0.65 at both cut strengths, reacts in ~0.5 s). Candidate
   dominance = how much my commands move the outcome now; not built (needs an
   outcome signal per body).
3. **Sim round with the chosen definitions** (EM, 16 runs): definitions hold
   on new runs; arousal predicts the world-model error 1-2 s ahead.
4. **Emotions that help learning** (literature: emotion acts selectively).
   LE1 (pain-tagged replay, stress-λ, fast fear): null. **Regret
   distillation** (train the policy toward the reflex's imagined best command
   before likely pain) helps and replicates: hard push n = 16, time to
   balance −38 % (p 0.013), crossings −19 % (p 0.003), falls −14 %; easy push
   n = 8, time to balance 6.3→4.3 min (p 0.016), crossings −33 %.

## Running at handoff

- **LE4** (Saion, e44940-e44979): regret strength β 0.05/0.1/0.2/0.4 vs base,
  easy push, seeds 831-838. Analysis: `pf_run.py` per run, Wilcoxon by seed.

## Next

- Pick β from LE4; if regret holds, make it a robot default candidate and test
  it live (needs the phone sync).
- Decide whether dominance becomes the command→outcome slope (body-specific
  outcome choice) or stays empowerment.
- Questions for Angela Yu: `LIT_AROUSAL_YU.md`.
