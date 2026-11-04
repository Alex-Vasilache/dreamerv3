# Intrinsic-motivation correlates of emotion for a DreamerV3 balancing robot — literature review

Compiled 2026-11-05. Each entry gives **Claim / Evidence / RL implication**. All citations
were checked against an indexed record (Europe PMC / PubMed abstract, Crossref, or the arXiv
API), and DOIs/URLs are given as checked. Where a statement goes beyond the abstract I read,
it is marked *[from full text, not re-checked]* or *[unverified]*.

Working definitions under test (PI): **valence = TD error**; **arousal = f(novelty +,
surprise +, empowerment −)**, possibly using *unexpected* surprise (total minus
expected/aleatoric), with empowerment as a possible third "dominance" axis (PAD). Hypothesis:
**high arousal raises the learning rate**, applied to learning in imagination.

---

## 1. Neuroscience of arousal and learning

**Aston-Jones & Cohen (2005).** An integrative theory of locus coeruleus–norepinephrine function: adaptive gain and optimal performance. *Annu. Rev. Neurosci.* 28:403–450. doi:10.1146/annurev.neuro.28.061604.135709
- *Claim:* LC has two modes. **Phasic** LC bursts are triggered by task-relevant decision outcomes and help exploitation. **Tonic** high LC goes with disengagement and exploration when task utility drops. ACC/OFC utility signals drive the switch.
- *Evidence:* Monkey LC recordings plus network models. The review relates tonic LC level to an inverted-U (Yerkes–Dodson) performance curve *[from full text, not re-checked]*.
- *RL implication:* "Arousal" is not one scalar. Distinguish a **phasic** arousal (event-locked surprise that gates learning on that event) from a **tonic** arousal (slow mean of low utility/high uncertainty that should raise exploration or entropy). Valence/utility trend decides which regime applies.

**Yu & Dayan (2005).** Uncertainty, neuromodulation, and attention. *Neuron* 46:681–692. doi:10.1016/j.neuron.2005.04.026
- *Claim:* ACh signals **expected uncertainty** (known unreliability within a context). NE signals **unexpected uncertainty** (unsignalled context switches that produce strongly unexpected observations).
- *Evidence:* Bayesian model of cueing tasks, consistent with pharmacological and physiological data.
- *RL implication:* This is the normative basis for "unexpected surprise = total − expected". Expected noise should *lower* the weight on single observations. Unexpected change should *raise* it. A learning-rate rule driven by raw surprise gets the aleatoric case backwards.

**Behrens, Woolrich, Walton & Rushworth (2007).** Learning the value of information in an uncertain world. *Nat. Neurosci.* 10:1214–1221. doi:10.1038/nn1954
- *Claim:* Humans set their learning rate by the estimated **volatility** of the reward environment.
- *Evidence:* Human task with volatile and stable blocks. ACC fMRI tracked the Bayesian volatility estimate at outcome time, and individual ACC signal predicted individual learning rates.
- *RL implication:* Scale step size with a *running* estimate of change rate (volatility), not with instantaneous error magnitude.

**Nassar, Wilson, Heasly & Gold (2010).** An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment. *J. Neurosci.* 30:12366–12378. doi:10.1523/JNEUROSCI.0822-10.2010
- *Claim:* Outcomes that are unexpected because the environment **changed** should carry more weight than outcomes that are unexpected because of **stochasticity**. The learning rate depends on change-point probability and on uncertainty (run length).
- *Evidence:* Human predictive-inference task. A reduced Bayesian delta rule fit behaviour.
- *RL implication:* α_t = CPP_t + (1 − CPP_t)·(relative uncertainty). This is a drop-in recipe: compute a change-point probability from how far the error falls outside the predicted noise.

**Nassar, Rumsey, Wilson, Parikh, Heasly & Gold (2012).** Rational regulation of learning dynamics by pupil-linked arousal systems. *Nat. Neurosci.* 15:1040–1046. doi:10.1038/nn.3130
- *Claim:* **Pupil-linked arousal regulates the learning rate.** Phasic pupil responses track assessed change points. Baseline pupil tracks uncertainty and expected hazard rate.
- *Evidence:* Pupil metrics predicted how much new data moved beliefs. A task- and luminance-independent pupil manipulation altered that influence.
- *RL implication:* This is the most direct support for the PI's hypothesis "arousal → learning rate". Note that the arousal involved is *change-point/unexpected-uncertainty* arousal, not generic excitement.

**Joshi, Li, Kalwani & Gold (2016).** Relationships between pupil diameter and neuronal activity in the locus coeruleus, colliculi, and cingulate cortex. *Neuron* 89:221–234. doi:10.1016/j.neuron.2015.11.028
- LC spiking anticipates pupil changes, but so do colliculi and cingulate. Treat "arousal" as a functional quantity, not a claim about one nucleus.

**Yerkes & Dodson (1908).** The relation of strength of stimulus to rapidity of habit-formation. *J. Comp. Neurol. Psychol.* 18:459–482. doi:10.1002/cne.920180503
- *Claim / evidence:* Mice learning a discrimination under shocks of different strength learned fastest at intermediate strength for hard discriminations. The "inverted-U law" is a later generalisation of this one study.
- *RL implication:* Any arousal → learning-rate gain should **saturate or reverse** at high arousal (for example a bounded or bell-shaped gain), not grow without limit. Treat the inverted U as a design hint, not an established law.

**McGaugh (2004).** The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annu. Rev. Neurosci.* 27:1–28. doi:10.1146/annurev.neuro.27.070203.144157
- *Claim:* The basolateral amygdala mediates the memory-enhancing effects of stress hormones through noradrenergic and cholinergic activation. It modulates **consolidation** in other regions. Amygdala activation at encoding predicts later recall of both pleasant and unpleasant material.
- *RL implication:* Arousal acts *after* encoding, on what gets consolidated. In Dreamer terms this maps to *what is replayed and weighted during training*, not only to online step size.

**Richter-Levin & Akirav (2003).** Emotional tagging of memory formation — in the search for neural mechanisms. *Brain Res. Rev.* 43:247–256. doi:10.1016/j.brainresrev.2003.08.005
- *Claim:* Amygdala activation **tags** an experience as important by lowering the threshold for synaptic tagging, which turns early-phase memory into late-phase memory.
- *RL implication:* Store an arousal tag with each replay chunk at collection time, and use it later for priority or loss weighting.

**Mather, Clewett, Sakaki & Harley (2016).** Norepinephrine ignites local hotspots of neuronal excitation: how arousal amplifies selectivity in perception and memory (GANE). *Behav. Brain Sci.* 39:e200. doi:10.1017/S0140525X15000667
- *Claim:* Arousal does not raise learning uniformly. It **amplifies high-priority representations and suppresses low-priority ones**.
- *RL implication:* An arousal gain should be *selective* (per-sample or per-loss-term weighting) rather than a global LR multiplier. That argues for priority/weighting schemes over a global α.

## 2. Dopamine and valence

**Schultz, Dayan & Montague (1997).** A neural substrate of prediction and reward. *Science* 275:1593–1599. doi:10.1126/science.275.5306.1593
- *Claim / evidence:* Phasic firing of primate midbrain dopamine neurons matches a temporal-difference reward prediction error.
- *RL implication:* Valence = δ_t = r_t + γV(s_{t+1}) − V(s_t) is well founded. It is a **phasic, signed** signal that goes to zero when rewards are predicted. For a balancing robot with dense, well-predicted reward it will hover near 0, so the face needs a slower component as well (see Broekens, §5).

**Bromberg-Martin, Matsumoto & Hikosaka (2010).** Dopamine in motivational control: rewarding, aversive, and alerting. *Neuron* 68:815–834. doi:10.1016/j.neuron.2010.11.022
- *Claim:* Some dopamine neurons encode **motivational value** (excited by reward, inhibited by aversion). Others encode **motivational salience** (excited by both). Both carry a fast **alerting** signal.
- *RL implication:* This separates the PI's two axes: signed δ = valence, |δ| or unsigned salience = an arousal input. Using |δ| inside arousal is biologically motivated.

**Kakade & Dayan (2002).** Dopamine: generalization and bonuses. *Neural Netw.* 15:549–559. doi:10.1016/S0893-6080(02)00048-5
- *Claim:* Dopamine responses to novel, non-rewarding stimuli are explained if DA **multiplexes exploration/shaping bonuses** with the RPE.
- *RL implication:* Novelty is a natural additive bonus inside the δ channel. For the display, though, keep novelty out of valence so the face does not "smile" at novel falls.

## 3. Surprise, novelty and information gain

**Modirshanechi, Brea & Gerstner (2022).** A taxonomy of surprise definitions. *J. Math. Psychol.* 110:102712. doi:10.1016/j.jmp.2022.102712 (arXiv:2209.01034)
- *Claim:* 18 surprise measures fall into four families: **prediction mismatch** (e.g. Shannon −log p(x)), **change-point probability** (e.g. Bayes Factor surprise), **confidence-corrected**, and **belief update** (e.g. Bayesian surprise, KL(post‖prior)).
- *RL implication:* State explicitly which one is computed. In an RSSM, −log p(o|·) is prediction-mismatch, KL(q(z|h,o)‖p(z|h)) is belief-update, and "Shannon − entropy" is a confidence-corrected-style measure. They dissociate (e.g. a high-entropy prior gives high Shannon but low "unexpected" surprise).

**Itti & Baldi (2009).** Bayesian surprise attracts human attention. *Vision Res.* 49:1295–1306. doi:10.1016/j.visres.2008.09.007
- *Claim / evidence:* Surprise is defined as KL(posterior‖prior). It is high only when data change beliefs, however rare the data are. 72% of human gaze shifts went to above-average-surprise locations (84% for consensus locations).
- *RL implication:* The RSSM's own KL term (the dynamics loss) is literally Bayesian surprise over the latent, so it costs nothing to compute.

**Faraji, Preuschoff & Gerstner (2018).** Balancing new against old information: the role of puzzlement surprise in learning. *Neural Comput.* 30:34–83. doi:10.1162/neco_a_01025
- *Claim:* A surprise that combines data likelihood with **belief commitment (entropy)** can drive the new-vs-old trade-off without knowing the environment's hazard rate.
- *RL implication:* This is a confidence-corrected surprise that sets the learning rate. It is the closest formal precedent for "unexpected surprise → learning rate".

**Liakoni, Modirshanechi, Gerstner & Brea (2021).** Learning in volatile environments with the Bayes Factor Surprise. *Neural Comput.* 33:269–340. doi:10.1162/neco_a_01352
- *Claim:* Exact Bayesian inference in a hierarchical change-point model yields an adaptation rate γ = m·S_BF/(1 + m·S_BF), where S_BF is a prior-vs-current-belief probability ratio. Several existing approximate learners already work this way.
- *RL implication:* This gives a principled, bounded (sigmoid-like) map from surprise to learning rate, consistent with the Yerkes–Dodson-style saturation. The formula is *[from full text, not re-checked]*. The abstract confirms "surprise-modulated trade-off".

**Schmidhuber (2010).** Formal theory of creativity, fun, and intrinsic motivation (1990–2010). *IEEE Trans. Auton. Ment. Dev.* 2:230–247. doi:10.1109/TAMD.2010.2056368
- Intrinsic reward should be **learning progress**, not raw prediction error, or agents are drawn to unlearnable noise (the same aleatoric trap as §1).

**Friston (2010).** The free-energy principle: a unified brain theory? *Nat. Rev. Neurosci.* 11:127–138. doi:10.1038/nrn2787
- Perception, learning and action minimise variational free energy, a bound on surprise. The RSSM ELBO is such an objective, so this is conceptual framing only (a published critique exists, doi:10.1038/nrn2787-c1).

**Depeweg, Hernández-Lobato, Doshi-Velez & Udluft (2018).** Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. *ICML*, PMLR 80:1184–1193. https://proceedings.mlr.press/v80/depeweg18a.html
- *Claim:* Total predictive entropy = **aleatoric** (expected entropy) + **epistemic** (mutual information between prediction and model parameters).
- *RL implication:* This is the formula behind "unexpected surprise" and novelty. Epistemic uncertainty needs a distribution over models (an ensemble). A single RSSM prior's entropy is only total, mostly aleatoric, uncertainty.

**ML exploration bonuses (verified on arXiv):** Achiam & Sastry 2017, *Surprise-based intrinsic motivation for deep RL* (arXiv:1703.01732); Pathak et al. 2017, *Curiosity-driven exploration by self-supervised prediction* (ICM, arXiv:1705.05363); Burda et al. 2019, *Exploration by random network distillation* (RND, arXiv:1810.12894); **Sekar et al. 2020, *Planning to explore via self-supervised world models* (Plan2Explore, ICML, arXiv:2005.05960)**. Plan2Explore is built on Dreamer and uses **ensemble disagreement of one-step latent predictors** as expected information gain. It is the ready-made epistemic-novelty estimator for our RSSM, and it is computed *in imagination*.

## 4. Empowerment and controllability

**Klyubin, Polani & Nehaniv (2005).** Empowerment: a universal agent-centric measure of control. *IEEE CEC 2005*, 1:128–135. doi:10.1109/CEC.2005.1554676
- *Claim:* Empowerment is the **channel capacity** from an agent's n-step action sequences to its future sensor states: max_{p(a)} I(A; S′|s).
- *RL implication:* Empowerment is state-dependent and task-free. It is low where actions have no effect on the future, for example after a fall or at actuator saturation.

**Salge, Glackin & Polani (2014).** Empowerment — an introduction. In *Guided Self-Organization: Inception*, Springer ECC 9:67–114. doi:10.1007/978-3-642-53734-9_4 (arXiv:1310.1863)
- Reviews estimators, including continuous-domain approximations. Use it to choose horizon n and the action distribution.

**Mohamed & Rezende (2015).** Variational information maximisation for intrinsically motivated RL. *NeurIPS 28* (arXiv:1509.08731)
- *Claim:* A variational lower bound on empowerment with a learned inverse ("source") model scales to pixels.
- *RL implication:* For a real-time estimate, a cheap alternative is to evaluate I(A;Z′) under a fixed action distribution (a lower bound on capacity) directly from the RSSM prior (see §9a).

**Karl, Soelch, Becker-Ehmck, Benbouzid, van der Smagt & Bayer (2017).** Unsupervised real-time control through variational empowerment. arXiv:1710.05101
- *Claim:* Maximising a variational empowerment bound in a **learned latent dynamics model** (Deep Variational Bayes Filters) moves agents into states where they can use their full capabilities.
- *RL implication:* This is a direct precedent for empowerment in a learned latent model. Balancing-type behaviour emerging from empowerment is reported in secondary summaries *[unverified, not in abstract]*. For our robot, upright with headroom on the wheels should be the high-empowerment state, so empowerment is a meaningful "dominance/control" readout.

**Maier & Seligman (2016).** Learned helplessness at fifty: insights from neuroscience. *Psychol. Rev.* 123:349–367. doi:10.1037/rev0000033
- *Claim:* The original theory "got it backward". Passivity is the **default, unlearned** response to prolonged aversive events, mediated by dorsal raphe serotonin. What is learned is **control**: vmPFC detects control and inhibits the DRN.
- *RL implication:* In the PI's framing, the default under negative outcomes is high stress or arousal. **Detected control (empowerment) is the inhibitory term.** This supports empowerment entering arousal with a negative sign, gated by negative valence.

**Amat, Baratta, Paul, Bland, Watkins & Maier (2005).** Medial prefrontal cortex determines how stressor controllability affects behavior and dorsal raphe nucleus. *Nat. Neurosci.* 8:365–371. doi:10.1038/nn1399
- *Evidence:* Inactivating ventral mPFC made controllable stress act like uncontrollable stress.
- *RL implication:* The same stressor (negative δ) should produce different arousal depending on estimated control. This is an interaction term (δ⁻ × (1 − E)), not an additive one. The pain analogue is Salomons et al. (2004) in §8.2.

**Huys & Dayan (2009).** A Bayesian formulation of behavioral control. *Cognition* 113:314–328. doi:10.1016/j.cognition.2009.01.008
- *Claim:* Controllability is a property of the **prior over affectively charged environments**. Helplessness can arise from normative inference.
- *RL implication:* Treat empowerment as a slowly learned, generalised prior (a tonic "dominance" mood), not only a per-step quantity.

**Dorfman & Gershman (2019).** Controllability governs the balance between Pavlovian and instrumental action selection. *Nat. Commun.* 10:5826. doi:10.1038/s41467-019-13737-7
- *Claim / evidence:* When rewards are uncontrollable, the brain favours the less flexible Pavlovian predictor. In Go/No-Go experiments the Pavlovian bias was stronger under low control.
- *RL implication:* Low empowerment should *reduce* reliance on and investment in the instrumental (actor) update. This is a possible meta-control use of empowerment.

## 5. Emotion in RL agents and robots; affect models

**Moerland, Broekens & Jonker (2018).** Emotion in reinforcement learning agents and robots: a survey. *Mach. Learn.* 107:443–480. doi:10.1007/s10994-017-5666-0
- *Claim:* The survey classifies emotion elicitation into homeostasis/extrinsic, appraisal, value/reward-based and hard-wired approaches, and functions into reward modification, state modification, **meta-learning** and action selection *[structure from full text, not re-checked]*.
- *RL implication:* Our proposal is a value- and appraisal-based elicitation, with meta-learning (learning rate, horizon, entropy) and replay as functions. The survey is the right place to position the paper.

**Broekens, Jacobs & Jonker (2015).** A reinforcement learning model of joy, distress, hope and fear. *Connect. Sci.* 27:215–233. doi:10.1080/09540091.2015.1031081
- *Claim:* **Joy/distress ≈ TD error** (phasic). **Hope/fear ≈ state value V(s)** (anticipatory).
- *RL implication:* Display valence on two time scales: a phasic flash from δ_t, and a tonic expression from a normalised V(s_t). With dense balancing reward δ is small most of the time, so V carries the visible mood ("fear" as V drops near a fall).

**Sequeira, Melo & Paiva (2011).** Emotion-based intrinsic motivation for reinforcement learning agents. *ACII 2011*, LNCS 6974:326–336. doi:10.1007/978-3-642-24600-5_36. Extended in *Adaptive Behavior* 22 (2014), doi:10.1177/1059712314543837
- *Claim:* Four appraisal dimensions (novelty, motivation, control, valence) become **intrinsic reward features**, and an optimised combination improves fitness.
- *RL implication:* This is a precedent for exactly our quantity set (novelty, valence, *control*) being computed as appraisals inside an RL agent.

**Doya (2002).** Metalearning and neuromodulation. *Neural Netw.* 15:495–506. doi:10.1016/S0893-6080(02)00044-8 — **key reference**
- *Claim (abstract):* "dopamine signals the error in reward prediction, serotonin controls the time scale of reward prediction, noradrenaline controls the randomness in action selection, and acetylcholine controls the speed of memory update."
- *RL implication:* DA = δ (valence). 5-HT = γ (discount/horizon). NE = inverse temperature β (actor entropy). ACh = learning rate α. The PI's hypothesis "arousal → learning rate" mixes the **ACh** role (α) with the **NE** role (randomness) in Doya's mapping. Yu & Dayan (2005) also assign *expected* uncertainty to ACh and *unexpected* uncertainty to NE. Implementing "arousal" with separate α, β and γ channels lets these be tested apart.

**Schweighofer & Doya (2003).** Meta-learning in reinforcement learning. *Neural Netw.* 16:5–9. doi:10.1016/S0893-6080(02)00228-9
- A meta-RL rule tunes α, β and γ online from short- vs long-term reward averages (MDP and nonlinear control task). It is the baseline to beat: it uses no world-model surprise.

**Affect models.**
- Russell (1980), *A circumplex model of affect*, *J. Pers. Soc. Psychol.* 39:1161–1178, doi:10.1037/h0077714: two bipolar axes, valence and arousal.
- Mehrabian (1996), *Pleasure–arousal–dominance*, *Curr. Psychol.* 14:261–292, doi:10.1007/BF02686918: adds **dominance**, which is a natural home for empowerment.
- Scherer (2009), *The dynamic architecture of emotion: evidence for the component process model*, *Cogn. Emot.* 23:1307–1351, doi:10.1080/02699930902928969: emotion arises from sequential appraisal checks (novelty → intrinsic pleasantness → goal conduciveness → **coping potential/control** → norms).

  *RL implication:* Scherer's check order maps onto our signals: novelty/surprise → valence δ → empowerment as coping potential. PAD with D = empowerment avoids folding a control signal into arousal, where it would cancel against surprise and hide information. Use PAD for analysis even if the face shows only 2D.

## 6. Replay and offline learning modulated by salience

**Schaul, Quan, Antonoglou & Silver (2016).** Prioritized experience replay. *ICLR* (arXiv:1511.05952)
- *Claim / evidence:* Sampling transitions by |TD error| (with importance-sampling correction) beat uniform replay on 41 of 49 Atari games.
- *RL implication:* Priority by |δ| (or by model loss) is the engineering analogue of emotional tagging. Our codebase already has it: `replay.fracs.priority`, `replay.prio`, and `replay.priosignal: model` in `dreamerv3/configs.yaml` (default fraction 0).

**Mattar & Daw (2018).** Prioritized memory access explains planning and hippocampal replay. *Nat. Neurosci.* 21:1609–1617. doi:10.1038/s41593-018-0232-z
- *Claim:* The optimal replay priority is **utility = gain × need**. Gain is how much a backup would improve the policy. Need is the expected future occupancy of the state. One model covers planning, learning and consolidation.
- *RL implication:* Surprise alone is "gain-like". Multiply by a "need" proxy (states the current policy will visit, e.g. recency or on-policy density). Otherwise priority chases rare off-policy junk such as post-fall states.

**Payne, Stickgold, Swanberg & Kensinger (2008).** Sleep preferentially enhances memory for emotional components of scenes. *Psychol. Sci.* 19:781–788. doi:10.1111/j.1467-9280.2008.02157.x
- *Claim / evidence:* After sleep, memory for negative objects was preserved but their neutral backgrounds were not. Sleep selectively consolidates the high-value components.
- *RL implication:* Offline phases ("sleep" = training-only epochs, or a higher train ratio when the robot is idle or charging) should favour arousal-tagged segments. Dreamer's imagination from replayed start states is a close functional analogue of replay-based planning.

**Daw, Niv & Dayan (2005).** Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nat. Neurosci.* 8:1704–1711. doi:10.1038/nn1560 (metadata verified; pages from memory)
- *Claim:* The brain arbitrates between model-based and model-free control by their relative **uncertainty**.
- *RL implication:* When model surprise is high, trust imagination less: use shorter horizons or rely more on the critic and real returns.

**DreamerV3 (substrate).** Hafner, Pasukonis, Ba & Lillicrap (2025). Mastering diverse control tasks through world models. *Nature* 640:647–653. doi:10.1038/s41586-025-08744-2 (arXiv:2301.04104)
- *Relevant facts:* The RSSM has categorical latents. Its KL dynamics/representation losses give Bayesian surprise directly. The actor-critic is trained on imagined rollouts started from replayed posterior states. Returns are normalised by a percentile range, which matters for scaling δ consistently across training.

## 7. Making valence and arousal legible on a screen face and in sound

**Breazeal (2003).** Emotion and sociable humanoid robots. *Int. J. Hum.-Comput. Stud.* 59:119–155. doi:10.1016/S1071-5819(03)00018-1
- *Claim:* Kismet maps a point in a 3-D **arousal × valence × stance** affect space to a face by **blending nine basis postures**. Intensity grows towards the extremes. Described on MIT's Kismet pages (https://groups.csail.mit.edu/lbr/sociable/facial-expression.html), which were opened; the full paper PDF was not accessible.
- *Implication:* Use continuous blending from (V, A, D), not discrete emotion labels. This avoids jumpy faces from noisy signals. Stance ≈ approach/avoid, which is close to our dominance/empowerment axis.

**Kalegina, Schroeder, Allchin, Berlin & Cakmak (2018).** Characterizing the design space of rendered robot faces. *HRI '18*, 96–104. doi:10.1145/3171221.3171286
- *Claim / evidence:* A survey of 157 robots with rendered faces, coded on 76 features, plus perception studies. Simple eyes-only and eyes-plus-mouth faces are common. Feature choices (e.g. presence of mouth, pupils, eyelids) shift perceived friendliness and other traits.
- *Implication:* A phone face only needs eyes, eyelids, pupils, a mouth curve and optionally brows. Keep it simple and test legibility with naive viewers.

**Löffler, Schmidt & Tscharn (2018).** Multimodal expression of artificial emotion in social robots using color, motion and sound. *HRI '18*, 334–343. doi:10.1145/3171221.3171261
- *Claim / evidence:* 28 expressions for joy, sadness, fear and anger, tested with N = 33. **Color + planar motion** had the best cost/benefit. Modalities differ: joy is best conveyed by color/motion, **sadness by sound**, fear by motion, anger by color.
- *Implication:* The background colour of the phone screen is a cheap, effective channel (warm/bright for positive valence, cool/dark for negative). Sound is especially useful for negative low-arousal states.

**Juslin & Laukka (2003).** Communication of emotions in vocal expression and music performance: different channels, same code? *Psychol. Bull.* 129:770–814. doi:10.1037/0033-2909.129.5.770
- *Claim / evidence:* A meta-analysis of 104 vocal and 41 music studies. The two channels share an acoustic code. High-arousal emotions go with **faster tempo, higher and more variable pitch, and greater loudness** *[cue table from full text, not re-checked]*.
- *Implication:* Map arousal → tempo/pitch/loudness of procedural sounds. Valence is better carried by mode or consonance and by the face than by pitch.

**Practical mapping (synthesis of the above; design choice, not a cited result):**
- Valence → mouth curvature, brow angle, screen hue.
- Arousal → eye openness, pupil size (the pupil–LC link makes this a nice, faithful touch), blink rate, saccade/jitter speed, sound tempo/pitch.
- Dominance/empowerment → head/eye "posture" (looking up/forward vs down), eyelid droop.
- Phasic δ → brief flashes. Tonic V and arousal → smoothed state (τ ≈ 0.5–2 s).

## 8. Pain: reward, barrier, or both?

Question (PI, for the TU Darmstadt IAS visit): can Peters' constraint-manifold safe RL serve as a correlate of pain? Verification notes for this section are at its end.

### 8.1 The constraint-manifold work (ATACOM)

**Liu, Tateo, Bou-Ammar & Peters (2021).** Robot reinforcement learning on the constraint manifold. *CoRL 2021*, PMLR 164:1357–1366 (volume dated 2022). https://proceedings.mlr.press/v164/liu22c.html
- *Claim:* ATACOM ("Acting on the TAngent space of the COnstraint Manifold") uses known robot models and constraint functions so that the agent explores safely in simulation and keeps the constraints satisfied throughout learning.

**Liu, Bou-Ammar, Peters & Tateo (2025).** Safe reinforcement learning on the constraint manifold: theory and applications. *IEEE Trans. Robot.* 41:3442–3461. doi:10.1109/TRO.2025.3567477 (arXiv:2404.09080). Journal version (2025, not 2024). The method below is from its arXiv v2 full text.
- *Mechanism:* Write the constraints as k(s) ≤ 0 and add slack variables μ ≥ 0, so that c(s, μ) = k(s) + μ = 0. The set M = {c = 0} is a manifold of the state's dimension. The system must be **control-affine**, ṡ = f(s) + G(s)u_s. The slack gets its own actuated dynamics μ̇ = α(μ)u_μ, with α a class-𝒦 function (linear βμ or exp(βμ) − 1). With J_u = [J_k G, A(μ)] and drift ψ = J_k f, the executed control is

  [u_s; u_μ] = −J_u†ψ − λ J_u† c + B_u u

  Here B_u is a (smoothly computed) basis of the null space of J_u, and **u is the RL action, expressed in tangent coordinates**. The first term cancels the drift toward the constraints. The second is the error correction: with gain λ it pulls the state back onto M. The third lets the policy move only along the manifold. The slack is initialised as μ = max(−k(s), tol).
- *Assumptions:* control-affine, locally Lipschitz dynamics; C¹ constraints (the paper notes neural-network constraints qualify); the rows of J_k G for the active constraints have full rank, so at most dim(u) constraints are active at once; and a solvable drift-compensation equation. Input limits are **not** guaranteed. Position constraints under acceleration control (relative degree 2) need a velocity-augmented constraint k* = ζ(k) + J_k ṡ. Uncontrollable state parts (moving obstacles) are handled with estimated velocities.
- *Guarantees:* V = ½‖c‖² is a Lyapunov function. M is positively invariant and attracting from a region that excludes singular points. A dwell-time argument covers stochastic, piecewise-constant RL actions. Under bounded disturbance or model error, the closed loop is input-to-state stable: violation stays within ‖c‖ ≤ η_c when λ is large enough. All guarantees are in continuous time. Experiments include **real robot air hockey**.
- *Per-step quantities it exposes:* the constraint values k_i(s) and slack μ_i (distance to each limit); the violation ‖c‖; the size of the drift-compensation plus correction term, ‖J_u†(ψ + λc)‖; and the distortion of the action space (how much of it B_u removes). ATACOM does not define "how far the executed action is from the policy's", because the policy acts only in tangent coordinates. Getting it needs a reference unconstrained action to compare against.

**Liu, Zhang, Tateo, Jauhri, Hu, Peters & Chalvatzaki (2023).** Safe reinforcement learning of dynamic high-dimensional robotic tasks: navigation, manipulation, interaction. *ICRA 2023*, 9449–9456. doi:10.1109/ICRA48891.2023.10161548 (arXiv:2209.13308). Note that Bou-Ammar is not an author, contrary to the brief.
- ATACOM is applied with dynamic obstacles and humans, using "complex collision constraints learned from data", and deployed on a TIAGo++ robot. This is the first **learned-constraint** variant.

**Günster, Liu, Peters & Tateo (2024).** Handling long-term safety and uncertainty in safe reinforcement learning. *CoRL 2024*, PMLR 270:4670–4697. https://proceedings.mlr.press/v270/gunster25a.html (arXiv:2409.12045)
- *Mechanism (from arXiv HTML):* A distributional cost critic, the **feasibility value** V_F(s) = E[Σγ^t max(k(s_t), 0)] (Gaussian, Wasserstein loss, with a failure buffer), is learned. Its CVaR minus an adaptive threshold δ is then used **as an ATACOM constraint**. The policy's Q is learned over the executed action, so that the constraint can change during training.
- *Implication:* This is the closest existing object to "anticipatory pain enforced as a barrier". A learned cost prediction becomes a hard constraint.

### 8.2 Neuroscience of pain relevant to RL

**Craig (2003).** A new view of pain as a homeostatic emotion. *Trends Neurosci.* 26:303–307. doi:10.1016/S0166-2236(03)00123-1
- Pain is "both a distinct sensation and a motivation". It is a homeostatic emotion, like temperature, itch, hunger and thirst, carried by a homeostatic afferent pathway that directly activates limbic motor cortex. *Implication:* pain is an interoceptive **state variable with its own drive**, not just a reward term.

**Fields (2006).** A motivation-decision model of pain: the role of opioids. In *Proc. 11th World Congress on Pain*, IASP Press, 449–459 (reference as cited in eLife 2023;12:e81795; the chapter itself was not opened). Background: Fields (2004), State-dependent opioid control of pain, *Nat. Rev. Neurosci.* 5:565–575, doi:10.1038/nrn1431.
- *Claim [from secondary sources, not re-checked]:* The brain decides whether to let nociception through, depending on competing motivations. When something more important is at stake, descending opioid control suppresses pain. *Implication:* the weight on pain should be **context-gated** by current goals, not fixed.

**Seymour (2019).** Pain: a precision signal for reinforcement learning and control. *Neuron* 101:1029–1041. doi:10.1016/j.neuron.2019.01.055
- *Claim (abstract):* "the core function of pain is motivational", directing short- and long-term behaviour away from harm. An RL model explains why pain is tuned by many factors and is supported by a distributed network, "recasting pain as a precise and objectifiable control signal". The review's split into innate/Pavlovian (reflex), habit (model-free) and goal-directed (model-based) control, and its treatment of endogenous modulation as an information decision, are *[from memory; full text blocked, not re-checked]*.

**Seymour, O'Doherty, Dayan et al. (2004).** Temporal difference models describe higher-order learning in humans. *Nature* 429:664–667. doi:10.1038/nature02581
- In higher-order aversive conditioning, ventral striatum and anterior insula activity matched **TD prediction errors for pain**. **Seymour et al. (2005)**, *Nat. Neurosci.* 8:1234–1240, doi:10.1038/nn1527: **relief** from pain is learned with reward-like signals (amygdala, midbrain), mirrored by aversion-like signals in lateral OFC/ACC. *Implication:* pain has its own TD channel. Leaving danger produces a positive "relief" error.

**Spinal withdrawal reflexes.** Schouenborg (2002), Modular organisation and spinal somatosensory imprinting, *Brain Res. Rev.* 40:80–91, doi:10.1016/S0165-0173(02)00191-1. Withdrawal is organised in spinal modules, each tuned during development to the withdrawal efficacy of its muscle. *Implication:* the protective barrier is fast, local and not goal-directed, but it is **calibrated by learning**. That is a fixed-structure, tunable-gain correction.

**Salomons, Johnstone, Backonja & Davidson (2004).** Perceived controllability modulates the neural response to pain. *J. Neurosci.* 24:7199–7203. doi:10.1523/JNEUROSCI.1315-04.2004
- The stimulus was held constant. Pain believed to be controllable attenuated ACC, insula and SII activation. This adds to §4 (Maier & Seligman; Amat): **control lowers the affective and neural pain response**, not the input.

**Fear of pain, avoidance and safety.** Vlaeyen & Linton (2000), *Pain* 85:317–332, doi:10.1016/S0304-3959(99)00242-0. In the fear-avoidance model, avoidance maintains pain-related fear, while confrontation extinguishes it. Christianson et al. (2012), *J. Neurosci.* 32:14118–14124, doi:10.1523/JNEUROSCI.3340-12.2012: learned **safety signals** (cues predicting no aversive event) inhibit fear and stress. *Implication:* an over-conservative learned cost critic is the analogue of fear-avoidance. It stops visiting near-limit states, so the critic is never corrected there. A safety-signal analogue (states with confirmed low cost) is needed to release it.

**Dissociating sensation and affect.** Rainville et al. (1997), *Science* 277:968–971, doi:10.1126/science.277.5328.968: hypnotically changing unpleasantness, with intensity held constant, changed ACC but not S1 activity. Berthier, Starkstein & Leiguarda (1988), Asymbolia for pain, *Ann. Neurol.* 24:41–49, doi:10.1002/ana.410240109: six patients with insular damage felt pain without primary sensory loss. They showed **no adequate emotional response, a lack of withdrawal, and were "unable to learn appropriate escape or protective responses"**. *Implication:* intensity (nociception) and aversiveness are separable. Removing the affective/learning component removes **protective learning**, not just the feeling. So a barrier alone is not a sufficient model of pain.

### 8.3 Safe RL with costs as separate signals

- **Constrained MDPs.** Altman (1999), *Constrained Markov Decision Processes*, Chapman & Hall/CRC (reissued 2021, doi:10.1201/9781315140223): maximise return subject to E[Σγ^t c_t] ≤ d. Solved through Lagrangian duality, with a **separate cost value V_c**.
- **Lagrangian and trust-region methods.** Achiam, Held, Tamar & Abbeel (2017), Constrained policy optimization, *ICML*, PMLR 70:22–31. Ray, Achiam & Amodei (2019), *Benchmarking safe exploration in deep RL* (Safety Gym), OpenAI tech. report, https://cdn.openai.com/safexp-short.pdf. These are cost critics with a learned multiplier λ ("pain as a separate critic").
- **In a Dreamer world model.** Huang, Ji, Xia, Zhang & Yang (2024), SafeDreamer, *ICLR 2024* (arXiv:2307.07176): adds Lagrangian methods and a cost model to Dreamer's imagination and planning, reaching near-zero cost on Safety-Gymnasium. This is the direct template for a cost critic trained in imagination on our substrate.
- **Projection and filters ("pain as a barrier").** Dalal et al. (2018), Safe exploration in continuous action spaces, arXiv:1801.08757: a safety layer projects each action in closed form using a **learned linearised** constraint model. This is the data-driven cousin of ATACOM. Alshiekh, Bloem, Ehlers, Könighofer, Niekum & Topcu (2018), Safe RL via shielding, *AAAI* 32, doi:10.1609/aaai.v32i1.11797: a shield overrides unsafe actions, which also works for **discrete** action sets. Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada (2019), Control barrier functions: theory and applications, *ECC 2019*, 3420–3431, doi:10.23919/ECC.2019.8796030: forward invariance of {h ≥ 0} enforced by a minimally invasive filter. ATACOM's slack-and-contraction construction plays the same role.

### 8.4 Answer, and a mapping for our framework

**Pain is best modelled as a combination.** The evidence points to three linked components, not to (a), (b) or (c) alone:
1. A **fast, hard-wired, learning-tuned barrier** (spinal withdrawal; Schouenborg).
2. A **separate aversive learning signal**, with its own TD errors and relief errors, that trains avoidance and prediction (Seymour 2004/2005/2019).
3. An **affective component** that feeds the general valence/arousal state and is modulated by context and control (Craig; Fields; Rainville; Salomons).

Option (a) alone, pain folded into reward, loses the hard guarantee and lets a large task reward buy violations. Option (b) alone, a barrier only, is an "asymbolic" agent: it is protected, but it learns nothing about avoiding the barrier, so the reflex fires forever (Berthier 1988). This matches the safe-RL split: a CMDP cost critic covers component 2, and ATACOM, a shield or a CBF covers component 1.

**Proposed mapping (our proposal, not a reported result):**

| Pain component | Our signal |
|---|---|
| Nociception n_t | Constraint proximity, n_t = Σ_i softplus((k_i(s_t) + m_i)/w_i), with margin m_i. Constraints: tilt θ − θ_hi and θ_lo − θ (bumpers +9.2° / −6.9°, `bumper_hi/lo` = 0.161/−0.120 rad, `bumper_margin` 0.0175); \|wheel command\| − u_max (saturation; the RP2040 limits); distance from home − `home_radius` / `pos_fence`; motor current − I_max *if the phone can measure it* (not checked). |
| Reflex | The existing **runaway brake** (`runaway_s`, `brake_s`), `pos_fence`, `drive_cap0` and **return home** are already hand-coded withdrawal reflexes. An ATACOM projection is a smooth, continuous replacement. |
| Pain intensity P_t | Σ_i max(k_i, 0) + ‖executed − policy action‖ (correction size) + a reflex-trigger indicator. Log it per step, and record the *executed* action in replay (as the drive cap already does). |
| Anticipatory pain / fear F_t | A cost critic V_c(s) trained **in imagination** (SafeDreamer-style), or Günster's CVaR feasibility value. Its TD error δ^c gives phasic pain (positive δ^c) and relief (negative δ^c). |
| Valence | Keep cost **out of the task reward for learning** (Lagrangian, separate critic). For the displayed mood, use valence_t = δ_t − β·(P_t + δ^c_t), so relief (δ^c < 0) reads as positive. |
| Arousal | A_t gains + w_P·P_t + w_F·F_t, since pain is salient whatever its sign (Bromberg-Martin, §2). |
| Controllability (empowerment) | Scale only the **affective** term by control, P_aff = P_t·(1 − κ·Ê_t) (Salomons 2004; Amat 2005). **Never** scale the barrier or the cost constraint by empowerment. Control should reduce the suffering, not the protection. |
| Fear-avoidance risk | A decaying exploration bonus, or a "safety signal" credit, for near-limit states that were visited without violation. This keeps V_c from becoming phobic. |

**Can ATACOM's assumptions hold here?**
- **Physical layer: mostly yes.** A wheeled inverted pendulum is control-affine in motor torque/voltage. The tilt limits are C¹. Tilt has relative degree 2, so the paper's velocity-augmented constraint is needed. There is one action dimension (two in turning mode), so at most one active constraint at a time, which fits a bumper pair, since only one bumper can be close. ATACOM could therefore run outside the world model, on an analytic pendulum model fitted by sysid.
- **Caveats at the physical layer.** Input limits are not guaranteed, and wheel saturation is exactly where falls become unavoidable. The viable set is smaller than the bumper set, so a **learned** feasibility constraint (Günster 2024) is the right k. The guarantees are continuous-time, while the phone applies commands every ~83 ms with latency. And the default robot config uses a **discrete** motion set (`discrete: True`), for which a shield (Alshiekh 2018) or action masking fits better than a tangent-space projection.
- **Inside the RSSM: not as stated.** The GRU transition h_{t+1} = f(h_t, z_t, a_t) is not control-affine, and k must be read through a decoder. A local version is possible: linearise the latent prediction of tilt with respect to a by JAX autodiff at every step, as in Dalal 2018. This is a heuristic with no guarantee. Use it in imagination to produce the fear signal, and keep the real-world barrier on the physical model or the hand-coded reflexes.

### Verification notes (§8)
- Checked: PMLR pages (CoRL 2021/2024, CPO); arXiv 2404.09080 and 2409.12045 (full HTML read), 2209.13308, 1801.08757, 2307.07176; the Safety Gym PDF listing; Crossref metadata for all DOIs; Europe PMC abstracts for all neuroscience entries.
- Not verified: Fields 2006 content (citation taken from eLife's reference list); Seymour 2019's control-hierarchy details (full text blocked); what kind of learned constraint ICRA 2023 uses; Ames 2019 beyond its metadata (standard CBF material).
- The brief's attributions were corrected: the T-RO paper is 2025 (vol. 41), and the ICRA 2023 author list is as given above.

---

## 9. Implications for our design

Pain and constraint signals (nociception, reflex, fear, relief) are mapped in §8.4.

### (a) Computing the signals in a DreamerV3 RSSM (categorical latents, 32×32 by default)

Notation: h_t = deterministic state, p(z_t|h_t) = prior, q(z_t|h_t,o_t) = posterior. All signals are z-scored with running statistics (percentile normalisation, as DreamerV3 does for returns) before they are combined. Compute **on the robot's real stream** (in the policy step) for the face, and **on replay/imagination** for learning.

1. **Valence (phasic):** δ_t = r_t + γ·V(s_{t+1}) − V(s_t) on real transitions (posterior states), with V from the critic (or slow critic), divided by the return-normalisation scale. **Valence (tonic):** V(s_t) normalised (Broekens' hope/fear). The current policy step does not call the critic, so a value-head call must be added for the display. Do not use δ computed *in imagination* as the displayed valence: it measures critic/model self-consistency, not real outcome error.
2. **Total surprise (belief update):** S_B = KL(q(z_t|h_t,o_t) ‖ p(z_t|h_t)), summed over the 32 categoricals. This is free, because it is the dynamics loss before free-bits clipping. Use the raw (unclipped) value. Reward/observation prediction-mismatch can be added as −log p(r_t|s_t) (twohot head).
3. **Expected surprise:** H[p(z_t|h_t)], the prior entropy. This is the expected Shannon surprise of the next latent under the model's own prediction.
4. **Unexpected surprise:** S_U = −log p(ẑ_t|h_t) − H[p(z_t|h_t)], with ẑ_t = argmax or sample of the posterior. This is "Shannon minus its expectation", close to the confidence-corrected family (Faraji 2018; Modirshanechi 2022). A change-point form à la Nassar is CPP_t = σ(k·(S_U − c)). Caveat (Depeweg 2018): H[prior] mixes aleatoric and epistemic uncertainty, so S_U is unexpected *relative to the model's stated confidence*. It is not purely epistemic.
5. **Novelty (epistemic):** Plan2Explore-style disagreement of a small ensemble (K = 5) of one-step latent predictors (h_t, a_t) → z_{t+1} (or their mean features). Variance across members = epistemic uncertainty. It is computable in imagination (novelty of *future* states) as well as on real data. A cheaper fallback is RND on h_t.
6. **Empowerment proxy (action sensitivity of the prior):** sample M actions a^(1..M) from the actor (or uniformly within limits). Compute the next prior for each by stepping the deterministic core: p_m = p(z_{t+1}|h_{t+1}(h_t,z_t,a^(m))). Then
   **E_t = H[(1/M)Σ_m p_m] − (1/M)Σ_m H[p_m]** (Jensen–Shannon / mutual information I(A;Z′) under the sampled action distribution). This is a lower bound on 1-step empowerment (Klyubin 2005). For n-step, roll the M open-loop action sequences for n = 3–8 steps through the prior and compare final-step distributions. Because the robot's control effect on the state takes several steps to build up, n > 1 will probably matter. Validate the proxy: it should be high when upright and near the balance point, and should collapse when tilted beyond recovery and after a fall (consistent with Karl et al. 2017).
7. **Arousal (PI form, with the helplessness/gambling interaction):**
   A_t = g( w_U·S_U + w_N·N_t + w_δ·|δ_t| · (1 − Ê_t) ), with Ê = normalised empowerment in [0,1] and g a bounded saturating function (sigmoid). The |δ|·(1−Ê) term makes low control amplify arousal for *both* signs of δ (helplessness for δ<0, "gambling" for δ>0), as the PI describes and as Amat 2005 / Maier & Seligman 2016 motivate. **Recommended alternative for analysis:** keep PAD = (δ/V, g(w_U S_U + w_N N + w_s|δ|), Ê) as three axes and compare the two encodings on how well they predict learning-relevant events (crashes, dynamics shifts).

### (b) Testable ways arousal/valence could modulate learning in imagination

General caution: DreamerV3's optimiser normalises gradients by their running RMS (LaProp/Adam-style). **A persistent multiplicative loss weight is largely cancelled out.** Only transient or *relative* (per-sample) weights survive. So implement "learning-rate" effects as a multiplier on the **update after normalisation** (i.e. on lr), or as relative per-sample weights within a batch.

1. **Surprise-gated world-model / critic learning rate** — Nassar 2010/2012; Behrens 2007; Yu & Dayan 2005; Liakoni 2021; Doya 2002 (ACh = α).
   lr_t = lr_0 · (1 + κ·CPP̄_t), with CPP̄ = batch/running mean of a change-point probability built from **unexpected** surprise S_U (not raw KL), bounded at ≤ 3–5× (Yerkes–Dodson/Bayes-factor saturation).
   *Prediction:* faster recovery after an induced dynamics shift, with no loss of stability in the stationary regime. Shifts to induce: change the stiction or mass, move the balance point (both are known sim-to-real issues for us). A raw-KL-gated control should do worse in high-noise conditions, because it learns from noise. *Test:* time to re-reach pre-shift return in sim, three seeds × three arms (none / raw-KL / S_U).

2. **Arousal-prioritised replay = imagination start states** — Schaul 2016; Mattar & Daw 2018; McGaugh 2004; Richter-Levin & Akirav 2003; Payne 2008.
   Use the existing `replay.fracs.priority` with a new `priosignal` = arousal tag A_t stored at collection, multiplied by a need proxy (recency). Mix with uniform sampling (e.g. 0.5/0.5).
   *Prediction:* falls, near-falls and push recoveries get more imagination starts, so recovery behaviour improves sooner (higher sustained return earlier; fewer falls per episode after perturbations). Over-prioritising post-fall (low-need) states should *hurt*, which tests Mattar & Daw's need term (arms: A, A×need, uniform).

3. **Arousal/uncertainty-adaptive imagination horizon** — Daw, Niv & Dayan 2005; Yu & Dayan 2005; Doya 2002 (5-HT = time scale).
   H_t = H_0 · (1 − η·Ū_model) for rollouts starting at high-novelty or high-S_U states (truncate or bootstrap earlier with the critic). Optionally lower λ for the same starts.
   *Prediction:* less exploitation of model errors right after a dynamics change. The imagined-vs-real return gap should shrink and the policy should be more stable. Little effect in the stationary regime. *Test:* the imagined-vs-real value gap on held-out real episodes.

4. **Tonic arousal × valence → actor entropy (exploration in imagination)** — Aston-Jones & Cohen 2005 (tonic = explore when utility wanes); Doya 2002 (NE = randomness); Dorfman & Gershman 2019 (low control → reduce instrumental investment).
   Scale the entropy coefficient: η_ent = η_0 · exp(c·Ā_tonic·𝟙[V̄ trend < 0]). Raise entropy only when sustained arousal coincides with falling value ("tonic mode"). Lower it on phasic arousal with positive δ ("phasic/exploit").
   *Prediction:* escape from plateaus (e.g. a stuck balance-point drift or a policy that oscillates). It should be neutral or slightly negative when learning is going well. The face makes the regime visible (restless eyes during the tonic phase).

5. **Consolidation weighting in "sleep" phases** — Payne 2008; McGaugh 2004; GANE (Mather 2016, selective amplification).
   Between deployment sessions (robot idle or charging), run extra training steps whose batches are drawn with arousal-tag priority, and apply per-sample relative weights (normalised to mean 1) to the world-model loss.
   *Prediction:* better retention of rare-event dynamics (lower held-out reconstruction/reward NLL on tagged falls) without harming typical-state error. Because of GANE, *selective* weighting should beat a global LR boost of equal mean.

**Key controls and measurements across all tests:** compare against Schweighofer & Doya (2003)-style meta-parameter adaptation from reward averages and against plain PER on |δ|. Pre-register the dynamics-shift protocol. Report the sustained-threshold crossing step and a late-window mean (not final scores alone). Check that the displayed signals are not dominated by the aleatoric noise of the robot's IMU: S_U and ensemble novelty should drop as the model learns, while H[prior] should not.

---

### Verification notes
- All DOIs above resolved to matching records: Europe PMC (abstracts read) for the neuroscience/psychology papers; Crossref for Moerland, Broekens, Breazeal, Klyubin, Kalegina, Löffler, Mehrabian, Russell, Scherer, Juslin & Laukka, Modirshanechi, Schmidhuber, Schweighofer & Doya, Salge, Yerkes & Dodson, Sequeira; the arXiv API for all arXiv IDs.
- Not fully verified: the exact Bayes-factor learning-rate formula (Liakoni 2021); the Kismet nine-posture detail (seen on MIT project pages, paper PDF blocked); Juslin & Laukka cue specifics; the Moerland survey taxonomy wording; the Aston-Jones inverted-U figure; balance behaviour in Karl et al. 2017; page numbers of Daw et al. 2005.
- The "practical mapping" in §7, the mapping in §8.4 and all of §9 are our proposals, not reported results. §8 has its own verification notes.
