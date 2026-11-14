# Arousal from the world model: Angela Yu's work

Literature note, 2026-11-14. Goal: improve the arousal axis of the emotion
model (EMOTION.md, "Arousal") using the work of Angela J. Yu (UC San Diego
2008-2022; Humboldt Professor at TU Darmstadt / hessian.AI since Dec 2022)
and close collaborators (Dayan, Cohen, Guo, Ryali, He, Meyniel). The PI
meets her soon; questions are listed at the end.

Every paper listed below was checked against Crossref (DOI), PubMed (PMID),
arXiv, or the NeurIPS proceedings index. The formulas were checked against
the full text where we could read it (marked "read"). Where only the
abstract could be read, the entry says so.

## 1. The problem in one paragraph

Arousal is currently one of these:
(a) total surprise `e_t = (1/32)[−log p̂(s_t)] − log p̂(r_t) − log p̂(c_t)`;
(b) "unexpected" `e_t − ê_t`, where `ê_t` is the prior's entropy;
(c) the belief update `KL(q_t ‖ p̂_t)`.
Each can also use the efference copy, and each is z-scored over about 10 s.
In sim, no variant marks bumper hits well (AUC 0.55-0.63; pleasure reaches 0.88).
On the 5 real sessions raw KL reaches 0.78 and the prediction-error variants are near chance.
At a bumper the one-step prior `p̂_t = p(·|h_t)` is already confident: the
deterministic state saw the approach. So surprise and its entropy terms
*fall* at contact.

In Yu's theory this is not a failure of the data. It is a mismatch of
definitions. Her noradrenaline (NE) is never "surprise under the one-step
prediction". It is either:

- **(tonic)** the posterior probability that the *context* has changed, judged
  against a learned rate of change (volatility), with the size of this signal
  compared to expected uncertainty (ACh); or
- **(phasic)** the posterior evidence for a *behaviourally relevant,
  non-default* state, divided by that state's **base rate in the current
  context**. The base rate is a slow, per-context prior, not the moment-to-moment
  prediction. The signal resets once the event has been acted on.

Under the phasic definition, a bump the model saw coming one step ahead
still drives NE strongly. The bad state's probability has gone from its
context base rate, around 0, to around 1. Under our definitions (a)-(c),
the same bump drives nothing, because the comparison is with the one-step prior.

## 2. Papers and what each contributes

Notation for the mapping. `h_t` is the deterministic RSSM state.
`p̂_t(z) = p(z_t | h_t)` is the prior and `q_t(z) = q(z_t | h_t, x_t)` the posterior.
`p̂(r), p̂(c)` are the reward and pain heads; `V_t` is value and `T_t` is threat (the critic of pain).
`π̄` is a slow, per-context baseline: a running average over tens of seconds, or the session.

### 2.1 Yu & Dayan (2002), NeurIPS 15: "Expected and unexpected uncertainty: ACh and NE in the neocortex" (read)
- Verified: NeurIPS proceedings index; DBLP `conf/nips/YuD02`.
  https://papers.nips.cc/paper_files/paper/2002/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html
- **Model.** An adaptive factor analysis. A hidden parameter `μ_t` drifts slowly
  (small, "microscopic" novelty) and is occasionally re-drawn by a binary
  change variable (macroscopic novelty, rare).
  - **ACh** = the variance of the current estimate of `μ` before the
    observation. This is expected uncertainty, and it sets the weight of
    bottom-up evidence against the top-down prediction.
  - **NE** = the posterior probability that a macroscopic change has just
    happened. This is unexpected uncertainty: "the radical divergence of
    bottom-up inputs from prevailing top-down interpretations".
  - Both modulators lower the top-down weight and raise the learning rate.
    NE does so in a burst at a change; ACh does so in a sustained way.
- **Key point for us.** NE is not "high error". It is "error too large to be
  explained by the expected uncertainty, given how often changes happen". In
  practice NE = P(change | x_t) ∝ h·p(x_t | new context) against
  (1−h)·p(x_t | old context, ACh variance).
- **RSSM.** The old context gives `p̂_t` (with its entropy as ACh). The new
  context is a *reset* prior that ignores `h_t`: the unconditional or slow
  average distribution of `z, r, c`. The hazard `h` is learned.

### 2.2 Yu & Dayan (2005), Neuron 46:681-692: "Uncertainty, neuromodulation, and attention" (abstract read; full text not reachable)
- DOI 10.1016/j.neuron.2005.04.026 (Crossref; PMID 15944135).
- The same split, applied to attention. ACh = "known unreliability of
  predictive cues within a context". NE = "unexpected uncertainty, as when
  unsignaled context switches produce strongly unexpected observations".
- The approximate algorithm infers a context switch when NE exceeds a
  threshold that **increases with ACh**, then resets the context estimate.
  The more noise the robot expects, the more surprise it needs before it
  declares a change. (We could not read the exact threshold formula here;
  check it in the paper before quoting it.)
- **RSSM.** NE is a *ratio* or comparison of unexpected to expected
  uncertainty, not their difference. Its target is the "context" (what
  `h_t` assumes about the world), not a single step.

### 2.3 Dayan & Yu (2005), NeurIPS 18: "Norepinephrine and neural interrupts", and Dayan & Yu (2006), Network 17:335-350: "Phasic norepinephrine: a neural interrupt signal for unexpected events" (NeurIPS read; journal abstract read)
- Verified: NeurIPS 2005 via DBLP `conf/nips/DayanY05`; DOI
  10.1080/09548980601004024 (Crossref; PMID 17162459).
- **Model.** An HMM of the monkey vigilance task: start → distractor (0.8) or target (0.2).
  The **phasic NE signal is**

      NE_t = P(target | x_1..x_t) / P(target)

  The denominator is the prior probability of the non-default state, as set
  by the task. A response is triggered when the posterior reaches 0.95, which
  is the same as `NE_t > 0.95 / P(target)`. After the decision, NE returns
  to baseline (1).
- What this explains:
  - NE fires for targets but not for equally rare distractors: it is "not driven by mere rarity".
  - The response weakens when targets are more common: NE ∝ 1/P(target).
  - The response vanishes when no action is required.
  - It is locked more tightly to the response than to the stimulus.
  - It reports "unexpected changes in the world within the context of a task"
    and acts as an **interrupt/reset**.
- The paper also notes a separate bottom-up route: "critical potential
  threats and stressors" should drive NE directly.
- **RSSM.**
  - The relevant non-default states are pain/threat (and big reward).
    The default is the robot's ordinary operating state.
  - Map `P(target | x_1..t)` to the probability of the bad outcome under the
    posterior, `q_t(c>0)` or normalised threat `T_t`.
  - Map `P(target)` to the base rate `π̄_c` of that outcome in the current
    context, which is slow. It is *not* `p̂_t(c)`.
  - This is the single most direct fix for "confidently right about a bad event".

### 2.4 Yu & Cohen (2008), NeurIPS 21: "Sequential effects: superstition or rational behavior?" (read)
- Verified: DBLP `conf/nips/YuC08`; PMID 26412953.
- **Dynamic Belief Model (DBM).** The latent rate `γ_t` stays the same with
  probability α and is re-drawn from a fixed prior `p0` with probability 1−α:

      p(γ_t | x_{1:t−1}) = α p(γ_{t−1} | x_{1:t−1}) + (1−α) p0(γ_t)

  Humans behave as if α < 1 even when the environment is stationary. DBM is
  closely approximated by a leaky (exponential) filter over past observations.
- **RSSM.**
  - The `(1−α) p0` term is a "persistent prior": every prediction is mixed
    with a context-free baseline.
  - It gives a principled *reset model* for 2.1 and 2.2: the "new context"
    likelihood is `p0`.
  - The exponential filter's time constant is a principled choice of window
    for `π̄`. It could replace the ad hoc ~10 s z-score window.

### 2.5 Ryali, Reddy & Yu (2018), NeurIPS 31: "Demystifying excessively volatile human learning: a Bayesian persistent prior and a neural approximation" (read, abstract and introduction)
- Verified: PMID 34366637; bioRxiv DOI 10.1101/077719 (Crossref).
- EXP, a leaky integrator with a pull toward the prior mean, matches DBM's
  predictions without representing uncertainty. Its finding: "when observations are
  information-poor, detecting changes or modulating the learning rate is both
  difficult and thus unnecessary".
- **RSSM.** A cheap per-context baseline `π̄_t = α π̄_{t−1} + (1−α)·x_t` (plus
  a pull toward a global `π_0`) is near-optimal. An explicit change-point
  model is not needed for the *baseline*.

### 2.6 Yu (2012), Nature Neuroscience 15:933-935: "Change is in the eye of the beholder" (title and venue only)
- DOI 10.1038/nn.3150 (Crossref). This is a News & Views on Nassar et al. (2012),
  "Rational regulation of learning dynamics by pupil-linked arousal systems",
  Nat Neurosci 15:1040-1046, DOI 10.1038/nn.3130 (Crossref).
- The quantities in Nassar's model:
  - Change-point probability `Ω_t`, tracked by *pupil change* (phasic).
  - Relative uncertainty `τ_t`, tracked by *baseline pupil* (tonic).
  - Learning rate `α_t = Ω_t + (1−Ω_t) τ_t`.
  - `Ω_t = h·U(x_t) / [h·U(x_t) + (1−h)·N(x_t; μ̂_t, σ²_total)]`, where U is the
    broad, context-free distribution.
- **RSSM.** `Ω_t` is the formal version of 2.1 (definition A2 below).

### 2.7 He, L'Hôtellier, Guo, Paunov, Meyniel & Yu (2022), CCN: "Role of pupil-linked uncertainties and rewards in value-based decision making" (read)
- DOI 10.32470/ccn.2022.1314-0 (Crossref).
- **Task.** A two-armed bandit with change points (hazard 1/24) and cued noise level.
- **Findings.**
  - Transient pupil (taken as unexpected uncertainty) peaks and decays shortly after
    a high-to-low change point, earlier in low noise. It predicts when the
    behavioural switch happens.
  - In forced trials, pupil change is larger for low-to-high change points (learning).
    In free trials it is larger for high-to-low change points (exploration).
- **RSSM.** Phasic arousal is *signed by relevance to the next decision*. A
  drop in value (high-to-low) is what triggers exploration and switching.

### 2.8 He, L'Hôtellier, Paunov, Guo, Meyniel & Yu (2024), CogSci: "Pupil size reflects the relevance of reward prediction error and estimation uncertainty in upcoming choice" (abstract only)
- Verified: DBLP `conf/cogsci/HeLPGMY24`; https://escholarship.org/uc/item/9cr3j67v
- **Finding.** Pupil-linked encoding of the RPE goes with reward-driven switches, and
  encoding of estimation uncertainty goes with uncertainty-driven switches.
  Individual differences track choice bias and performance.
- **RSSM.** Arousal should weight the TD error and posterior uncertainty by how much
  they matter for the *next action*, not report them raw.

### 2.9 Paunov, L'Hôtellier, Guo, He, Yu & Meyniel (2024), bioRxiv: "Multiple and subject-specific roles of uncertainty in reward-guided decision-making" (abstract read)
- DOI 10.1101/2024.03.27.587016 (Crossref). Companion: CCN 2023, DOI 10.32470/ccn.2023.1479-0.
- Uncertainty preference varies across people from seeking to avoiding. Higher
  uncertainty pushes choices toward heuristics (repeat, win-stay/lose-shift).
- **RSSM.** This is background only. Arousal's *effect* on behaviour (more reflexive,
  heuristic action under high uncertainty) is a separate design choice.

### 2.10 Guo & Yu (2021), CogSci: "Revisiting the role of uncertainty-driven exploration in a (perceived) non-stationary world" (read via PMC)
- PMID 34368809.
- DBM-style learning with persistent prior:
  `μ̂_t = (1−α)μ_0 + α(μ̂_{t−1} + g(R_t − μ̂_{t−1}))`.
  Choice is a softmax over `μ̂ + γσ̂`.
- **Finding.** Assuming non-stationarity, people show strong uncertainty-driven
  exploration. Models that assume stationarity underestimate it.
- **RSSM.** This supports a persistent-prior baseline `π̄`, as in 2.4 and 2.5.

### 2.11 Other verified Yu papers, lower relevance
- Yu & Dayan (2002), "Acetylcholine in cortical inference", *Neural Networks*
  15:719-730, DOI 10.1016/S0893-6080(02)00058-8. An HMM in which ACh = 1 − P(most likely context).
- Yu (2007), "Adaptive behavior: humans act as Bayesian learners", *Curr Biol*
  17:R977, DOI 10.1016/j.cub.2007.09.007. A dispatch on volatility learning.
- Cohen, McClure & Yu (2007), *Phil Trans R Soc B* 362:933-942, DOI
  10.1098/rstb.2007.2098. Covers exploration/exploitation and LC-NE adaptive gain:
  tonic NE favours exploration, phasic NE favours exploitation.
- Yu (2006), "Optimal change-detection and spiking neurons", NeurIPS 19,
  DOI 10.7551/mitpress/7503.003.0198.
- Paulus & Yu (2012), "Emotion and decision-making: affect-driven belief systems
  in anxiety and depression", *TiCS* 16:476-483, DOI 10.1016/j.tics.2012.07.009.
  Places valence and arousal inside a POMDP belief-state framework: emotion
  disorders as altered probability and value computation. This is the closest
  Yu paper to our pleasure/arousal design.
- Godara, Alemán & Yu (2024), "Metareasoning in uncertain environments: a
  meta-BAMDP framework", arXiv 2408.01253. Recent Darmstadt work, not about arousal.
- Ide, Shenoy, Yu & Li (2013), *J Neurosci* 33:2039, DOI
  10.1523/JNEUROSCI.2201-12.2013. ACC reports a Bayesian prediction and its
  surprise in the stop-signal task.

Outside Yu, but directly on our KL result: Yanagisawa, Kawamata & Ueda (2019),
*Front Comput Neurosci*, DOI 10.3389/fncom.2019.00002. Arousal = KL(posterior ‖ prior),
i.e. information gain. This matches variant (c) and why it did best on the real sessions.

## 3. Why the current variants miss bumper hits (Yu's reading)

1. **Wrong denominator.** (a)-(c) compare against `p̂_t`, which already
   contains the approach. Dayan & Yu normalise the relevant posterior by its
   *context base rate*. Ours is the one-step prediction, which is the
   denominator that cancels predictable events.
2. **No relevance gating.** Phasic NE responds to the action-relevant
   non-default state, not to any rare input. Our state term (1/32 of 32
   categoricals) mixes relevant and irrelevant surprise. Raw KL may win on the
   real robot partly because a hit moves many latents *and* the pain head at once.
3. **Difference instead of comparison.** `e − ê` subtracts expected uncertainty.
   Yu & Dayan instead set the change *threshold* by expected uncertainty
   (unexpected judged *relative to* expected). When `ê` collapses at a confident
   bumper, the difference behaves erratically.
4. **The z-score is a baseline on the signal, not on the prior.** The
   ~10 s z-score already gives NE's "weaker when common" property. If bumps
   are frequent, each one counts less. But it cannot restore a signal that is
   flat at contact.
5. **No reset.** Phasic NE returns to baseline after the event has been handled.
   A z-score with no reset lets one hit inflate the next 10 s of baseline.

## 4. Proposed arousal definitions (ranked)

All are unsigned, in nats or log-ratios, z-scored as now. `ε = 1e-3`.
"Logged" refers to the per-step dict from `Emotion.step()` in
`dreamerv3/deploy/emotion.py`. It already logs: `surprise` (e), `expected` (ê),
`unexpected`, `update` (KL), the `_eff` variants, `value`, `threat`, `td`,
`o_reward`, `o_pain`, `pain`, `outlook`, `change`.

### A1 (rank 1). Relevance-normalised phasic NE (Dayan & Yu 2006)

    b_t      = q_t(c_t > 0)  (or  T̃_t = T_t / S_T, normalised threat)
    π̄_t      = α π̄_{t−1} + (1−α) b_{t−1}               slow per-context base rate, τ ≈ 30-60 s
    NE^c_t   = KL( Bern(b_t) ‖ Bern(π̄_t) )             evidence for the bad state vs. its base rate
    NE^r_t   = same for big reward (or |V_t − V̄_t| / S_V)
    arousal_t = NE^c_t + NE^r_t ;  after a catch-reflex trigger or contact, hold π̄ and let NE decay (reset)

- The simple form is `log(b_t + ε) − log(π̄_t + ε)`, the paper's own ratio in log units.
- The KL form is symmetric: a sudden *drop* of threat in a dangerous context also arouses.
- **Why first.**
  - It is the paper's own formula, mapped one-to-one.
  - It is high when the model is *confidently right* about a bad event in a context where
    such events are rare.
  - It is relevance-gated by construction, and it falls when bumps become common.
- **Test now.** Compute it offline from the logged `threat` (the `T̃` version) and `value`.
  No new logging is needed.
- **Logging for the exact form.** `q_t(c>0)` from the cost head on the posterior
  sample, and `p̂_t(c>0)` on the prior.
- **Risk.** It overlaps with pleasure, which also uses threat. Report the correlation
  with pleasure and the AUC of arousal after regressing pleasure out.

### A2 (rank 2). Change-point probability against learned volatility (Yu & Dayan 2002/2005; Nassar 2012)

    ℓ^old_t = −e_t                                          log-lik under the one-step prior p̂_t (already logged: surprise)
    ℓ^new_t = (1/32) log π̄_s(s_t) + log π̄_r(r_t) + log π̄_c(c_t)   log-lik under a reset prior (slow marginal, DBM p0)
    logit Ω_t = log(h_t/(1−h_t)) + ℓ^new_t − ℓ^old_t
    h_t = α_h h_{t−1} + (1−α_h) Ω_t                          learned hazard (volatility)
    arousal_t = Ω_t   (phasic);   tonic = h_t   (baseline pupil analogue)

- **What it marks.** "The model's context failed": being picked up, a new floor,
  a stuck wheel. It does *not* mark a well-predicted hit, and per theory it should not.
  Expect lower bumper AUC than A1, but a better match to the real-session KL result.
  Log `Ω` as its own channel even if it is not chosen for the face.
- **Logging.** The per-term split of `surprise` (state, reward, pain separately).
  For the baseline, a slow average of the prior's 32×32 probabilities kept online,
  and per step the value `log π̄_s(s_t)`. That is one float, cheap. Logging 1024
  probabilities per step is the offline alternative.
- **Variant.** Replace `ℓ^old − ℓ^new` with the logged KL `update`. Then
  `Ω_t = σ(log h/(1−h) + β·update_t)` is a volatility-calibrated version of the variant
  that already won on the real data. It needs no new logging; `h` comes from the trace itself.

### A3 (rank 3). Lagged-prior surprise on relevant heads (DBM persistent prior, phasic NE ∝ 1/base rate)

    e^(k)_t = −log p̂_{t|t−k}(c_t) − log p̂_{t|t−k}(r_t)     prediction made k steps earlier, open loop, with intended actions
    arousal_t = max_k∈{1,8,32} [ e^(k)_t − ê^(k)_t ]  ,  or  e^(k=16)_t alone

- **Idea.** Surprise is judged against what the robot believed half a second ago, not one
  step ago. A bump foreseen only at the last moment is surprising. A bump planned for
  (e.g. by the catch reflex) is not.
- **Logging.** A ring buffer of open-loop k-step pain/reward predictions. That is one
  imagination rollout of k steps per control step; the empowerment code already rolls out
  H = 16-32 steps. Offline, it needs the logged `deter`/`stoch` and actions plus the
  checkpoint, to re-imagine. That is more work, which is why it ranks third.

### A4 (rank 4). ACh-gated interrupt (Yu & Dayan 2005 form)

    U_t = leaky avg of (e_t − ê_t)^+ ,  A_t = leaky avg of ê_t          (τ from DBM α)
    arousal_t = σ( (U_t − θ·A_t) / s )   ;  reset U after a trigger

- This replaces the current difference (b) with a comparison against expected
  uncertainty. It needs only logged `surprise` and `expected`.
- **Why last.** At bumpers `ê` falls, so the threshold falls and arousal rises. That
  helps, but the error itself also falls. Keep it as the cheap baseline that tests point 3
  of section 3 in isolation.

### Offline test protocol (all four)

1. **Data.**
   - Sim round PF logs and the 5 real sessions, at the per-step `Emotion.step()` output.
   - Labels: sim bumper-contact flags; real `pain > 0` steps (hits).
2. **Metrics.**
   - Per-session AUC for hit vs. no-hit within ±0.25 s, reported as mean ± std.
   - Lead time: the first step before contact where z > 1.
   - Event-locked averages for hit / near-miss / caught-by-reflex. This is the analogue
     of Dayan & Yu's hit/fa/miss/cr plot.
   - Partial AUC after regressing pleasure out.
   - Hit-rate dependence: does arousal per hit fall in hit-dense stretches (NE ∝ 1/P(target))?
3. **Baselines.** Current (a), (b), (c) with and without efference copy, and pleasure.
4. **Order.**
   - A1 (threat version) and the A2 KL variant need no new logging. Run these first on
     existing logs.
   - Then add the logging for the exact A1 and A2 (`q(c>0)`, `p̂(c>0)`, per-term surprise,
     `log π̄_s(s_t)`), run sim round PF+1, and compare.
   - A3 last.

## 5. Questions for Angela Yu

1. **The denominator.** In phasic NE = P(target | x)/P(target), the denominator is the
   task base rate. For an agent with a learned recurrent world model, what should it be:
   the one-step predictive prior, a slow per-context marginal (DBM `p0`), or a lagged
   k-step prediction? How should its timescale be set: by the learned hazard, or by the
   decision horizon?
2. **Predicted aversive events.** Does LC/pupil respond to a fully *predicted* aversive
   event, e.g. a signalled shock or a collision the agent saw coming? Is that the
   "bottom-up threats and stressors" route of the 2006 paper, separate from
   unexpected uncertainty? Should a robot's arousal therefore have two channels:
   relevance-normalised evidence (A1) and a change-point signal (A2)?
3. **Relevance and control.** Phasic NE vanished when no action was required. Should
   arousal be gated by controllability, which is our dominance/empowerment axis, or
   by the critics (value/threat)? Is "arousal ∝ |Δ belief about a relevant state|"
   a fair reading of her He et al. 2022/2024 pupil results?
4. **Tonic NE and learning.** We already weight world-model training by arousal.
   Would she use `Ω_t` (change-point probability) or the learned hazard `h` to set the
   learning rate or replay priority instead, as in `α = Ω + (1−Ω)τ`? Has her group
   tested NE-like learning-rate control in deep RL?
5. **A shared benchmark.** Could her change-point bandit with pupil (He et al. 2022;
   Paunov et al. 2024) be run by our DreamerV3 agent, so that its arousal trace is
   compared directly with human pupil? This would be an offline validation of
   A1 versus A2, and a concrete collaboration.

## First test (11-14): A1 on the logged threat, 48 sim runs

PF, PF2 and PF3 (e44550-e44581, e44700-e44715, e44760-e44767; 60 min,
pushes and walls), minutes 5-60, AUC per run, mean ± SD, n = 48. A1 here uses
the threat critic's output normalised by its 99.9th percentile as b_t and a
30 s or 60 s exponential average as the base rate; "rising" keeps it only when
b_t is above the base rate. Script: scratchpad yu/a1.py (copy in the study
notes when promoted).

| arousal | at a hit (±0.25 s) | 0.1-0.3 s before |
|---|---|---|
| A1 rising, 30 s base rate | 0.69 ± 0.04 | **0.71 ± 0.08** |
| A1, 30 s | 0.67 ± 0.03 | 0.64 ± 0.05 |
| A1, 60 s | 0.64 ± 0.03 | 0.63 ± 0.05 |
| current total | 0.64 ± 0.04 | 0.58 ± 0.02 |
| current kl | 0.62 ± 0.03 | 0.62 ± 0.02 |
| current unexpected | 0.57 ± 0.02 | 0.53 ± 0.02 |
| threat, z-scored (no base rate) | 0.65 ± 0.05 | 0.72 ± 0.09 |

A1 rising beats the best current version before a hit (vs kl, Wilcoxon
p < 1e-9) and A1 30 s beats total at the hit (p = 0.001). It is strongest
with impact pain (0.71 / 0.75, n = 32) and weaker with bumper pain (0.65 /
0.63, n = 16). Caveat: the warning mostly comes from the threat level
itself (z-scored threat warns as well), which pleasure already contains
through the outlook; A1's added value is at the hit. The exact A1 (posterior
vs base-rate pain probability, reward term, reset after contact) needs
q(c > 0) and p̂(c > 0) logged.
