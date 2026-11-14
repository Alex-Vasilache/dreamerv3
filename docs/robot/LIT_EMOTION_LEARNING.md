# Which emotions change learning, and how: mechanisms for the robot

Literature note for plan step 5-6 ("Neuroscience literature on emotion and
learning; emotions to speed up learning"), 2026-11-14. It builds on
[EMOTION_LITERATURE.md](EMOTION_LITERATURE.md) (11-05: arousal, valence,
empowerment, pain) and [LIT_AROUSAL_YU.md](LIT_AROUSAL_YU.md) (11-14: arousal
as base-rate-relative NE). Papers already covered there are cited by name
only. The signals are those of [EMOTION.md](EMOTION.md): pleasure p, arousal,
dominance (empowerment C), pain c (impact), threat T (the `agent.fear` critic).

Verification. Every entry was resolved on 2026-11-14: journal papers by DOI
on the Crossref API, arXiv papers on the arXiv API. Where the abstract was
read (Europe PMC or arXiv) the entry is marked **[abs]**; where only the
record (authors, title, journal, year) was checked it is marked **[meta]**,
and the claim given is the paper's well-known headline, not re-read. Effect
sizes are only quoted where an abstract states them. The mechanisms and
predictions in sections 3-4 are our proposals, not reported results.

## 0. What has already been tried on this robot, and why it failed

| tried (sim) | result | why, as far as we can tell |
|---|---|---|
| Arousal-triggered lr x (1 + k I) on every optimizer group, burst updates, instant push (`online_arousal`) | no help; lost 2 hard seeds (round EA) | arousal was not selective (fired ~15/min; falls at bumper hits because the prior is confident there), so it became a raised global lr; with salience events (1.3/min) neutral |
| Arousal-weighted imagination starts (`agent.arousal_imag`, relative per batch) | negative/neutral (E1/E2) | weighted by arousal at the start state, which does not mark the moments that matter |
| Surprise (model-loss) priority replay; recency share | negative/neutral | model loss chases sensor noise and post-fall states (gain without need, Mattar & Daw 2018) |
| Vigour action gain; arousal-scaled exploration | no help | act on behaviour, not on what is learnt; the policy already sets its own spread |
| Catch reflex alone (old rule; then threat-search s4z2) | alone hurt; s4z2 cuts impact pain 42%, crossings n.s.; always-on s4on halves crossings but slows t90 to 7.2 min | a filter on actions the policy never learns from |
| **Fear / threat critic** | **the one clear win**: crossings 14.1 -> 4.0/min (EA) | a separate learnt signal with its own critic, entering the actor as a relative advantage |

Two lessons fix the design space. (1) DreamerV3's LaProp normalises every
gradient by its running RMS, so any persistent global loss scale cancels;
only (a) per-sample weights relative within a batch, (b) a change of which
samples are drawn, (c) a change of the learning *target* (λ, horizon,
counterfactual terms), or (d) a static post-normalisation lr split between
optimizer groups survive. (2) Biology does not use one global "learn harder"
knob either: in nearly every result below, emotion acts **selectively**
(which memory, which outcome sign, which system), and the selection signal is
outcome-related (reward, pain, threat), not raw surprise. That is also what
won here (fear).

The world model, not the actor-critic, limits the first minutes (Z2,
SAMPLE_EFF_STUDY.md: trained model + fresh actor-critic is 66% upright in
minute 1). So t90 can only move through mechanisms that help the model or
stop the actor from trusting it while it is poor; late crossings move through
the actor-critic and what replay keeps.

## 1. Evidence by emotion

### 1.1 Arousal / noradrenaline and the learning rate

- **Behrens et al. 2007**, *Nat. Neurosci.* 10:1214, doi:10.1038/nn1954 [abs]:
  humans track volatility near-optimally and raise their learning rate with
  it; ACC activity tracks the estimate and predicts individual learning rates.
- **Nassar et al. 2012**, *Nat. Neurosci.* 15:1040, doi:10.1038/nn.3130 [abs]:
  phasic pupil (LC-linked arousal) tracks change-point probability, baseline
  pupil tracks uncertainty; together they predict how much new data moves
  beliefs, and manipulating pupil changes it causally.
- **Yu & Dayan 2005**, *Neuron* 46:681, doi:10.1016/j.neuron.2005.04.026 [meta]:
  ACh = expected, NE = unexpected uncertainty (see LIT_AROUSAL_YU.md).
- **Pulcu & Browning 2017**, *eLife* 6:e27879, doi:10.7554/eLife.27879 [abs]:
  humans keep **separate learning rates for wins and losses**, each set by
  that outcome's own volatility; the pupil (NE) tracked it for losses.
- **Browning et al. 2015**, *Nat. Neurosci.* 18:590, doi:10.1038/nn.3961 [abs]:
  high trait anxiety = failure to adapt the aversive learning rate to
  volatility, with a blunted pupil response.
- **Lefebvre et al. 2017**, *Nat. Hum. Behav.* 1:0067, doi:10.1038/s41562-017-0067
  [meta]: humans learn more from positive than negative prediction errors
  (optimistic asymmetry), tied to striatal signals.
- **Dabney et al. 2020**, *Nature* 577:671, doi:10.1038/s41586-019-1924-6 [abs]:
  VTA dopamine neurons differ in their positive/negative asymmetry, a
  distributional code for value.

*Robustness:* strong (replicated, normative models, causal pupil
manipulation). *But* the effect is a learning rate on **one belief** set by
**that belief's** volatility, not a global plasticity boost. Our failed boost
applied one scalar to every network.

### 1.2 Emotional memory: consolidation, tagging and replay

- **McGaugh 2000**, *Science* 287:248, doi:10.1126/science.287.5451.248 [meta]
  and **McGaugh 2004** (EMOTION_LITERATURE.md): amygdala/NE after an arousing
  event strengthens its consolidation; a single footshock trial (inhibitory
  avoidance) gives a lasting memory.
- **Dunsmoor et al. 2015**, *Nature* 520:345, doi:10.1038/nature14106 [abs]:
  pairing items with shock **retroactively** strengthens memory of earlier,
  related neutral items, after consolidation only, and not for items already
  strongly encoded.
- **Moncada & Viola 2007**, *J. Neurosci.* 27:7476,
  doi:10.1523/JNEUROSCI.1083-07.2007 [abs]: a weak avoidance memory becomes
  long-term if novelty is met close in time (dopamine D1/D5): behavioural
  tagging.
- **Singer & Frank 2009**, *Neuron* 64:910, doi:10.1016/j.neuron.2009.11.016
  [abs]: CA3 replay (SWR) is stronger **after rewarded** trajectories, more so
  during learning, and replays the paths that led to reward.
- **Foster & Wilson 2006**, *Nature* 440:680, doi:10.1038/nature04587 [abs]:
  awake replay right after experience runs **in reverse**, from the outcome
  back.
- **Ambrose, Pfeiffer & Foster 2016**, *Neuron* 91:1124,
  doi:10.1016/j.neuron.2016.07.047 [abs]: only **reverse** replay scales with
  reward change (up with more reward, down with less); forward replay does not.
- **Wu et al. 2017**, *Nat. Neurosci.* 20:571, doi:10.1038/nn.4507 [abs]: after
  footshock, rats avoiding the shock zone replay the **paths from where they
  are to the shock zone**, without entering it.
- **Carey, Tanaka & van der Meer 2019**, *Nat. Neurosci.* 22:1450,
  doi:10.1038/s41593-019-0464-6 [abs]: replay is not simply "what was
  rewarding": after a motivational shift it favours the *non*-preferred arm.
- **Mattar & Daw 2018** (EMOTION_LITERATURE.md): replay priority = gain x need.
- **Wilhelm et al. 2011**, *J. Neurosci.* 31:1563,
  doi:10.1523/JNEUROSCI.3575-10.2011 [abs]: sleep consolidates memories
  **expected to be needed** (need term). **Stickgold & Walker 2013**,
  *Nat. Neurosci.* 16:139, doi:10.1038/nn.3303 [abs]: sleep "triages" memories.
- **Diamond et al. 2007**, *Neural Plast.* 60803, doi:10.1155/2007/60803 [abs]:
  strong emotion briefly opens plasticity, then suppresses new plasticity
  while the memory consolidates.

*Robustness:* consolidation of arousing events, strong (rodent pharmacology,
human). Reward-biased and reverse replay, strong in rodents. Retroactive
tagging, one human study plus the rodent tag-and-capture line (moderate).
*For us:* the prioritised content is the **path leading to** the salient
outcome (reverse replay, Wu 2017, Dunsmoor 2015, Singer & Frank), weighted by
need. Our surprise replay prioritised the moment of high model loss itself
and had no need term.

### 1.3 Mood, happiness and learning

- **Eldar & Niv 2015**, *Nat. Commun.* 6:6149, doi:10.1038/ncomms7149 [abs]:
  a mood change (wheel-of-fortune win/loss) biases how later outcomes are
  valued; in mood-unstable people this is positive feedback that destabilises
  mood (shown in simulation).
- **Eldar, Rutledge, Dolan & Niv 2016**, *TICS* 20:15,
  doi:10.1016/j.tics.2015.07.010 [abs via FACE_LIT_AFFECT.md]: mood = momentum
  of recent prediction errors; it biases perception of the next outcome so as
  to speed learning in environments that are changing as a whole.
- **Rutledge et al. 2014**, *PNAS* 111:12252 (our pleasure model).
  **Rutledge et al. 2015**, *J. Neurosci.* 35:9811,
  doi:10.1523/JNEUROSCI.0702-15.2015 [abs]: L-DOPA raised risky choice for
  gains only (more Pavlovian approach) and raised happiness after some rewards.
- **Blain & Rutledge 2020**, *eLife* 9:e57977, doi:10.7554/eLife.57977 [abs]:
  happiness follows **learning-relevant** prediction errors (probability PE),
  not reward PE: "how we learn matters more than what we get".
- **Fredrickson & Branigan 2005**, *Cogn. Emot.* 19:313,
  doi:10.1080/02699930441000238 [abs]: positive emotion broadened attention
  and action repertoires; negative emotion narrowed repertoires (n = 104).
  **Gasper & Clore 2002**, *Psychol. Sci.* 13:34, doi:10.1111/1467-9280.00406
  [meta]: happy mood -> global, sad -> local processing.
  **Easterbrook 1959**, *Psychol. Rev.* 66:183, doi:10.1037/h0047707 [meta]:
  arousal narrows the range of cues used.

*Robustness:* mood biasing valuation: moderate (one main experiment plus
theory). Broaden/narrow: replicated in lab tasks, small to moderate, and
mostly attentional, not exploration in RL. *For us:* the exploration version
already failed. The useful part is Blain & Rutledge: affect tracks learning
progress, so a "mood" signal is best used as a **learning-progress** signal
that allocates compute, not as an exploration knob.

### 1.4 Fear, threat, pain: fast avoidance, safety, extinction

- **Öhman & Mineka 2001**, *Psychol. Rev.* 108:483,
  doi:10.1037/0033-295X.108.3.483 [abs]: a dedicated, fast fear-learning
  module (amygdala), automatic and hard to override.
- **Garcia & Koelling 1966**, *Psychon. Sci.* 4:123, doi:10.3758/BF03342209
  [meta]: one-trial aversion learning, cue-specific.
- **Maren 2001**, *Annu. Rev. Neurosci.* 24:897,
  doi:10.1146/annurev.neuro.24.1.897 [meta]: Pavlovian fear forms in one or a
  few trials.
- **Krypotos et al. 2015**, *Front. Behav. Neurosci.* 9:189,
  doi:10.3389/fnbeh.2015.00189 [meta]: avoidance learning models (two-factor,
  RL accounts).
- **Christianson et al. 2012** (EMOTION_LITERATURE.md): learned safety signals
  inhibit fear. **Bouton 2004**, *Learn. Mem.* 11:485, doi:10.1101/lm.78804
  [abs]: extinction is new, context-dependent inhibitory learning; the
  original fear is not erased. **Quirk & Mueller 2008** (online 2007),
  *Neuropsychopharmacology* 33:56, doi:10.1038/sj.npp.1301555 [meta].
- **Seymour 2019** (EMOTION_LITERATURE.md): pain as a precision teaching signal.
- **Lipton et al. 2016**, arXiv:1611.01211 [abs]: deep RL agents **forget**
  catastrophes and periodically revisit them ("Sisyphean curse"); a learnt
  fear model penalising imminent catastrophe fixes pathological environments
  and learns faster.

*Robustness:* fast aversive learning, very strong across species. *For us:*
fear already works. Two parts are unused: the speed (one-trial: our threat
critic learns at the same lr as everything else) and protection from
forgetting (our late collapse once came from evicting recovery examples).

### 1.5 Curiosity, novelty and memory

- **Gruber, Gelman & Ranganath 2014**, *Neuron* 84:486,
  doi:10.1016/j.neuron.2014.08.060 [abs]: high curiosity improves memory for
  the answers **and for incidental material** met in that state, via
  midbrain-hippocampus coupling.
- **Kang et al. 2009**, *Psychol. Sci.* 20:963,
  doi:10.1111/j.1467-9280.2009.02402.x [abs]: curiosity activates caudate;
  it improves recall of surprising answers 1-2 weeks later.
- **Bunzeck & Düzel 2006**, *Neuron* 51:369, doi:10.1016/j.neuron.2006.06.021
  [abs]: SN/VTA codes absolute novelty (not rarity or negative valence), and
  novelty enhances learning in the same context for minutes.
- **Wittmann et al. 2008**, *Neuron* 58:967, doi:10.1016/j.neuron.2008.04.027
  [abs]: novelty drives choice via striatum (a novelty bonus).
- **Lisman & Grace 2005**, *Neuron* 46:703, doi:10.1016/j.neuron.2005.05.002
  [meta]: the hippocampus-VTA loop gates novel information into long-term
  memory. **Kakade & Dayan 2002** (EMOTION_LITERATURE.md): novelty bonuses.
- **Kauvar et al. 2023**, *Curious Replay*, arXiv:2306.15934 [abs]: priority
  from a count term (rarely replayed) plus model loss, in **DreamerV3**:
  Crafter 14.5 -> 19.4, DMC unchanged, faster adaptation when the environment
  changes.
- **Bierling et al. 2025**, *DreamerV3-XP*, arXiv:2510.21418 [abs]: trajectory
  priority by return, reconstruction loss and value error, plus reward-
  ensemble disagreement: faster learning, mainly in sparse-reward tasks.

*Robustness:* curiosity/novelty -> memory, moderate-strong. *For us:* dense
reward, one task; Kauvar's gain was on Crafter, none on DMC. Low priority
(our surprise and recency replay were neutral to negative).

### 1.6 Regret, disappointment and counterfactual learning

- **Camille et al. 2004**, *Science* 304:1167, doi:10.1126/science.1094550
  [abs]: regret needs counterfactual comparison with the unchosen option;
  orbitofrontal patients feel no regret and do not learn to avoid it.
- **Coricelli et al. 2005**, *Nat. Neurosci.* 8:1255, doi:10.1038/nn1514
  [abs]: regret (seeing the unchosen outcome) engages mOFC, ACC, hippocampus;
  people become increasingly regret-averse, and the same circuit fires
  before the next choice.
- **Lohrenz et al. 2007**, *PNAS* 104:9493, doi:10.1073/pnas.0608842104
  [abs]: a **fictive error** (best possible minus obtained) strongly predicts
  the next investment change (n = 54) and correlates with striatal signal.
- **Amsel 1958**, *Psychol. Bull.* 55:102, doi:10.1037/h0043125 [meta]:
  frustrative non-reward energises behaviour and underlies the partial-
  reinforcement extinction effect.
- ML: **Hindsight Experience Replay**, Andrychowicz et al. 2017,
  arXiv:1707.01495 [meta]; **counterfactual credit assignment**, Mesnard et
  al. 2020, arXiv:2011.09464 [meta]; **Recovery RL**, Thananjeyan et al. 2020,
  arXiv:2010.15920 [abs]: a safety critic plus a separate recovery policy
  traded violations against success 2-20x better than 5 safe-RL baselines in
  sim and 3x on a real robot.

*Robustness:* fictive/regret signals exist and guide the next choice:
moderate-strong in humans. *For us:* the world model can evaluate the
unchosen commands at any real state (the catch reflex already does this at 4
steps). Regret = a learning signal from that comparison, which we compute and
then throw away.

### 1.7 Controllability, helplessness, effort

- **Maier & Seligman 2016**; **Amat et al. 2005** (EMOTION_LITERATURE.md):
  uncontrollable stress -> passivity; detected control (mPFC) protects.
- **Dorfman & Gershman 2019** (EMOTION_LITERATURE.md): lower controllability
  -> more Pavlovian, less instrumental control.
- **Eisenberger 1992**, *Psychol. Rev.* 99:248, doi:10.1037/0033-295X.99.2.248
  [abs]: rewarded effort makes effort itself less aversive (learned
  industriousness).

*Robustness:* helplessness and controllability: strong in rodents, moderate
in humans. *For us:* instrumental learning from states where actions do not
change the outcome (on a bumper, empowerment collapses 3-7 nats) is
noise for the actor.

### 1.8 Stress: inverted U and model-based vs habitual

- **Yerkes & Dodson 1908** (EMOTION_LITERATURE.md); **Teigen 1994**,
  *Theory Psychol.* 4:525, doi:10.1177/0959354394044004 [meta]: the
  "law" was reinterpreted over decades; its general inverted-U form is weakly
  supported. **Lupien et al. 2007**, *Brain Cogn.* 65:209,
  doi:10.1016/j.bandc.2007.02.007 [meta]: inverted-U effects of
  glucocorticoids on memory.
- **Arnsten 2009**, *Nat. Rev. Neurosci.* 10:410, doi:10.1038/nrn2648 [abs]:
  even mild uncontrollable stress rapidly impairs prefrontal function.
- **Schwabe & Wolf 2009**, *J. Neurosci.* 29:7191,
  doi:10.1523/JNEUROSCI.0979-09.2009 [abs]: stress before training made
  behaviour habitual (insensitive to devaluation).
- **Otto et al. 2013**, *PNAS* 110:20941, doi:10.1073/pnas.1312011110 [abs]:
  the cortisol response reduced model-based, not model-free, control; high
  working memory protected.
- **Daw, Niv & Dayan 2005** (EMOTION_LITERATURE.md): arbitration by relative
  uncertainty. **Kool, Gershman & Cushman 2017**, *Psychol. Sci.* 28:1321,
  doi:10.1177/0956797617708288 [abs]: people use more model-based control
  when it is more accurate **and** stakes are high.
- ML analogue: **STEVE**, Buckman et al. 2018, arXiv:1807.01675 [abs]: per
  example, interpolate rollout horizons by model uncertainty, use the model
  only where it is accurate; order-of-magnitude sample-efficiency gain over
  model-free baselines.

*Robustness:* stress -> habit, moderate-strong (several paradigms); inverted
U, weak as a general law. *For us:* "stress -> trust the model less" is the
normative version (Daw 2005), and a per-sample horizon is how ML made it pay.

### 1.9 Emotion as meta-learning, and ML that measured a benefit

- **Doya 2002**; **Schweighofer & Doya 2003** (EMOTION_LITERATURE.md): DA =
  TD error, 5-HT = discount, NE = inverse temperature, ACh = learning rate.
- **Krichmar 2008**, *Adapt. Behav.* 16:385, doi:10.1177/1059712308095775
  [meta]: neuromodulation framework for robot controllers.
- **Moerland, Broekens & Jonker 2018** (survey); **Gadanho & Hallam 2001**,
  *Adapt. Behav.* 9:42, doi:10.1177/105971230200900102 [meta]: an emotion
  system on a simulated robot used to decide when to switch behaviours and
  as reinforcement. **Salichs & Malfaz 2012**, *IEEE TAC* 3:56,
  doi:10.1109/T-AFFC.2011.32 [meta]: fear from worst-case Q-values for
  dangerous states. **Oudeyer, Kaplan & Hafner 2007**, *IEEE TEC* 11:265,
  doi:10.1109/TEVC.2006.890271 [meta]: learning progress as intrinsic
  motivation.
- **Miconi et al. 2019**, *Backpropamine*, arXiv:2002.10585 [meta]: a learnt
  neuromodulatory signal gating plasticity improves meta-learning tasks.
- **Moskovitz et al. 2021**, *Tactical Optimism and Pessimism*,
  arXiv:2102.03765 [abs]: the best degree of optimism changes across tasks
  and during learning; a bandit switching it online set SOTA on pixel tasks.
- **Lee et al. 2021**, *SUNRISE*, arXiv:2007.04938 [abs]: per-sample Bellman
  weights from ensemble uncertainty, a relative weighting that improved SAC
  and Rainbow.
- **Oh et al. 2018**, *Self-Imitation Learning*, arXiv:1806.05635 [abs]:
  re-learning from one's own better-than-expected past (return > V) aids hard
  exploration.
- **Isele & Cosgun 2018**, arXiv:1802.10269 [abs]: of four long-term replay
  rules (surprise, reward, distribution matching, coverage), **distribution
  matching** prevented forgetting best; surprise and reward selection did not.
- **Huang et al. 2023**, *SafeDreamer*, arXiv:2307.07176 [abs]: Lagrangian cost
  critic in Dreamer: near-zero cost on Safety-Gymnasium (our fear, in effect).

*Robustness of the ML side:* no deep-RL paper we found measured a benefit of
a **global** emotion-driven learning-rate boost; the measured gains are all
**selective**: per-sample weights (SUNRISE), sample choice (Curious Replay,
SIL), per-example horizon (STEVE), separate cost critic or recovery policy
(SafeDreamer, Recovery RL, intrinsic fear). An arXiv search (11-14) for
emotion-modulated learning rates in deep RL returned nothing relevant.

## 2. Summary table

Strength: S strong (replicated, causal), M moderate, W weak or contested.

| emotion / signal | effect on learning | evidence | robot signal | target in DreamerV3 | status here |
|---|---|---|---|---|---|
| Arousal (NE), volatility | lr per belief, set by that belief's volatility | S: Behrens 2007, Nassar 2012, Pulcu & Browning 2017 | pain/reward volatility | per-group lr (threat critic vs value critic vs WM) | global version failed |
| Emotional tagging, consolidation | arousing events and the events before them are kept | S: McGaugh; M: Dunsmoor 2015 (retroactive) | pain, pleasure drop in the next 0.25-1 s | replay share + relative per-start weights for **pre-event windows** | untested (start-state arousal weighting failed) |
| Reward/fear replay | reverse replay of paths to reward or shock | S (rodent): Singer & Frank 2009, Ambrose 2016, Wu 2017 | same | imagination starts on the path to the event | untested |
| Need-weighted replay | consolidate what will be needed | S: Mattar & Daw 2018; M: Wilhelm 2011 | on-policy density / recency | multiply priority by need | surprise without need failed |
| Mood (momentum) | biases outcome valuation; tracks learning | M: Eldar & Niv 2015, Blain & Rutledge 2020 | slow pleasure; holdout learning progress | compute allocation (ratio), not exploration | exploration failed; `online_progress` exists |
| Positive/negative mood | broadens/narrows attention | M: Fredrickson 2005 | pleasure | entropy | failed (exploration) |
| Fear, pain | one-trial aversive learning; avoidance | S: Öhman & Mineka 2001, McGaugh | pain c, threat T | threat critic (has), its lr, protected memory | fear works; speed unused |
| Safety, extinction | fear inhibited by safety learning, not erased | S: Bouton 2004, Christianson 2012 | low T after near-miss | keep pain memories (no eviction) | eviction caused a collapse once |
| Curiosity, novelty | better memory for material met while curious | M-S: Gruber 2014, Bunzeck 2006 | novelty, count | priority on new data (Curious Replay) | recency/surprise neutral |
| Regret, fictive error | learning from the unchosen option | M-S: Lohrenz 2007, Coricelli 2005, Camille 2004 | best alternative's outlook minus taken | actor loss toward better counterfactual command | computed by catch search, discarded |
| Controllability | no instrumental learning without control | S (rodent): Maier & Seligman 2016; M: Dorfman & Gershman 2019 | dominance C (or action sensitivity) | relative per-start actor weight | untested |
| Stress | shift to model-free / habit | M-S: Schwabe & Wolf 2009, Otto 2013, Daw 2005 | world-model error at the start | per-start λ / horizon | untested |
| Inverted U | moderate arousal best | W: Teigen 1994 | | clip any modulation | |
| Neuromodulators as meta-parameters | DA/5-HT/NE/ACh set δ, γ, β, α | theory: Doya 2002 | | | lr and β versions failed |

## 3. Mechanisms for the DreamerV3 robot (all proposals)

Test protocol for all: RF arm b as control (S2 pick + impact pain, threat
critic weight 1, no brake, `sim_push 5:0.05:0.15`, `sim_walls 0.5`,
`sim_phone_threads 1`), 60 min, n = 8 (4 Saion V100 seeds 811-814, 4 IAS
815-818), Mann-Whitney. Control values (SAMPLE_EFF_STUDY.md, RF):
**t90 4.4 ± 0.7 min, crossings 12.7 ± 4.3 /min (min 11-60), falls per push
0.17 ± 0.06.** With that spread and n = 8, a change below ~30% in crossings is
not detectable; predictions are stated at that resolution.

**M1. Retroactive pain tagging of imagination starts ("reverse replay").**
Signal: future salience in a window, g_t = max_{t<k<=t+W} (c_k + max(0, −Δp̃_k)),
W = 16-65 steps (0.25-1 s), tagged in replay after the fact (the learner sees
the whole chunk). Need: multiply by a recency/on-policy factor and exclude
steps already in contact (impact > ι0 or empowerment z < −2: post-hit states
have no need). Target: (a) a priority selector share of 0.25 over tagged
steps, priority fixed (not model loss, no `zero_on_sample`), so pain memories
are never evicted from sampling; (b) per-start actor, critic and threat loss
weights 1 + w·g, renormalised to mean 1 per batch (the `arousal_imag` code
path with a new signal). Why it can work where arousal weighting failed: the
signal is selective (pain/|TD| events are 97% within 1 s of a hit, arousal
falls there), it lands on the states where a different action still helps,
and the need term avoids post-fall junk. Evidence: Singer & Frank 2009,
Ambrose 2016, Wu 2017, Dunsmoor 2015, Mattar & Daw 2018, Lipton 2016.
Prediction: crossings −30% (≤ 9/min) and falls per push −30% (≤ 0.12), t90
unchanged (± 0.5 min). Falsified if crossings stay within 15% of control.
Control arm worth adding: same weights on the event steps themselves
(no window) should do worse, as Mattar & Daw predict.

**M2. Regret: distil the counterfactual command into the actor.**
Signal: at real replayed states with g_t > 0 (M1's window), roll the 7
candidate commands a + {0, ±0.25, ±0.5, ±1} for 4 steps in the world model
(the s4z2 search) and score each by outlook O = V/S̄_V − w T/S̄_T. Regret
ρ_t = max_j O_j − O_taken. Target: an extra actor term
−λ_r · ρ̄-normalised · log π(a*_t | s_t) on those states (a* = best
candidate), weight relative within the batch (mean 1 over tagged samples,
total share fixed, e.g. 10% of the actor loss). Why it can work: the reflex
computes exactly this comparison online and discards it; s4on showed the
comparison is right (crossings −53%) but filtering half the actions slowed
learning (t90 7.2). Teaching the policy instead puts the benefit into π, and
unlike the "N rollouts per start" test (V, uniform, within noise) the extra
compute goes only to pre-pain states. Evidence: Lohrenz 2007, Coricelli 2005,
Camille 2004, Recovery RL, SIL. Prediction: crossings ≤ 8/min (−35%), impact
pain −40%, t90 ≤ 4.5 min (not s4on's 7.2); the online reflex, if left on,
should fire less over time. Falsified if crossings fall but t90 rises above
5.5 (then it is just a slower s4on), or if neither moves.

**M3. Stress: per-start bootstrap by world-model error (trust the model less
where it is surprised).**
Signal: the posterior-prior KL (or unexpected surprise) at the replayed start
and along its first imagined steps' training-time counterparts, z-scored per
batch. Target: per-start λ_i = λ0 · (1 − κ σ(z_i)) (and optionally horizon cut
at the step where the predicted pain/continue head is least certain), so
uncertain starts bootstrap from the critic sooner. This changes the target,
not a scale, so LaProp cannot cancel it. Why it can work: Z2 showed the world
model is the early bottleneck and Z1 that the actor exploits a poor model
(3x faster driving, 22 crossings/min); STEVE got its gain from exactly this
per-example choice. Evidence: Daw 2005, Otto 2013, Schwabe & Wolf 2009, Kool
2017, STEVE. Prediction: t90 −0.7 min (to ≤ 3.7), late crossings unchanged
(± 15%). Falsified if t90 is within 0.3 min of control. Risk: the critic is
also poor early; bootstrap too early and t90 rises (then κ is too large).

**M4. Fast fear: a higher post-normalisation lr for the threat head and
threat critic only, set by pain volatility.**
Signal: none at first (static split), then the running volatility of pain
events (Behrens-style: rate of change of the pain rate). Target: the
`MultiOptimizer` group lr of the cost head and cval critic ×3 (static), or
×(1 + k·volatility) clipped at 3; every other group unchanged. Why it can
work where the boost failed: it is a persistent split between networks (a
ratio of two lrs, which the RMS normalisation does not remove), not a
transient global boost, and it follows the human finding that wins and
losses get separate rates (Pulcu & Browning 2017) and that fear is learnt in
one trial. Prediction: crossings in minutes 3-10 −30%; late crossings −20%;
t90 unchanged. Falsified if minutes 3-10 do not move. Risk: a noisier threat
critic early; check that the threat's AUC for crossings within 0.25 s (0.76
at 4 steps now) does not drop.

**M5. Controllability gate on the actor (no instrumental lesson without
control).**
Signal: action sensitivity at the start, cheaply as the spread of the 4-step
outlook over the 7 M2 candidates, D_i = SD_j O_j (or dominance C where
logged), z-scored per batch. Target: actor loss per start × σ(z(D_i)),
renormalised to mean 1; critic and threat losses untouched (they still learn
the value of hopeless states). Why it can work: on a bumper empowerment
collapses 3-7 nats; the actor's advantage there is noise that still moves π.
Evidence: Maier & Seligman 2016, Amat 2005, Dorfman & Gershman 2019,
SUNRISE (relative per-sample weights). Prediction: wobble (rate SD) −10%,
crossings −20%; t90 unchanged. Weakest prediction: falsified if crossings
and wobble are within the seed spread. Cheap once M2's rollouts exist.

**M6 (low priority). Curiosity window in the first minutes.**
Signal: replay count (never-replayed chunks) during the first 3 min only.
Target: a 0.3 count-priority share for the world model's batches, off after
t90 or 5 min. Evidence: Gruber 2014, Bunzeck & Düzel 2006, Moncada & Viola
2007, Curious Replay. Prediction: t90 −0.5 min. Listed last because recency
and surprise replay were already neutral and Curious Replay showed no gain
on DMC.

Not proposed again: global lr boosts (any signal), arousal-scaled exploration
or vigour, mood-driven entropy (Fredrickson's effect is attentional and our
version failed), model-loss priority without need.

## 4. Ranked shortlist (test first)

| rank | mechanism | signal → target | 60-min sim prediction (vs RF b) | why it differs from what failed |
|---|---|---|---|---|
| 1 | M1 retroactive pain tagging | future pain / pleasure drop in next 0.25-1 s × need → fixed-priority replay share 0.25 + relative per-start AC/threat weights | crossings ≤ 9/min (−30%), falls/push ≤ 0.12, t90 ± 0.5 | selective signal on pre-hit states, need term, no eviction |
| 2 | M2 regret distillation | best-minus-taken 4-step outlook at pre-pain states → extra relative actor term toward the best command | crossings ≤ 8/min, impact pain −40%, t90 ≤ 4.5 | new gradient (counterfactual), learnt not filtered (s4on slowed t90) |
| 3 | M3 stress-adaptive bootstrap | world-model surprise at the start → per-start λ / horizon | t90 ≤ 3.7 min, late crossings ± 15% | changes the target, aimed at the early world-model bottleneck |
| 4 | M4 fast fear | (pain volatility) → threat head/critic group lr ×3 | crossings min 3-10 −30%, late −20% | persistent per-network split, not a transient global boost |
| 5 | M5 controllability gate | outlook spread over commands (dominance) → relative per-start actor weight | crossings −20%, wobble −10% | removes noise rather than adding gain |

Suggested order: M1 and M4 are a few lines each (selector signal, group lr)
and can share one round with M3; M2 and M5 share the candidate rollouts and
go in a second round, M2 on top of the round-1 winner.
