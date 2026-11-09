# Modality-agnostic augmentation, DreamerV4, and multi-rollout imagination

2026-11-09. Literature review for the PI's two questions: (1) augmentation not
tied to the input modality (perturbing/masking activations or the RSSM state),
and (2) what DreamerV4 does, especially "several imagination rollouts per
replay entry", and whether that speeds learning or scales with compute.
Setting: DreamerV3 (JAX RSSM + actor-critic in imagination) trained online on
the phone robot at 65 Hz from a low-dim proprio vector; balance in ~3 min.
Companion to `SAMPLE_EFFICIENCY_SURVEY.md` (2026-10-06), which covers replay
ratio, DroQ/BBF, MDP design.

Verification: arXiv IDs below were checked by web search on 2026-11-09 unless
marked **[unverified]**. DreamerV4 details were read from the arXiv HTML.

## 0. Headline findings

1. **DreamerV4 does not start several rollouts per replay entry. It does the
   opposite.** Section 3.3 states "we start only one rollout from each context,
   prioritizing data diversity and reducing memory consumption", explicitly
   as a departure from earlier Dreamers. DreamerV3 already starts one rollout
   from every replayed timestep (B·T starts; ours: 16x64 = 1024 starts per
   update, see `rssm.py:starts`, capped by `agent.imag_last`). So "more starts
   per sample" is something we already have at B·T scale; the untested axis is
   **N > 1 rollouts per start** (repeated stochastic rollouts), which neither
   paper ablates.
2. DreamerV4's self-supervision comes from **masked autoencoding in the
   tokenizer** (patch drop rate p ~ U(0, 0.9)) and **noise-level conditioning
   in the dynamics** (shortcut forcing, a diffusion-forcing variant). Both are
   "corrupt the input, predict the clean target" objectives and the second one
   is modality-agnostic: it corrupts latents, not pixels.
3. For low-dim proprio, the strongest evidence for "neuron-level"
   augmentation is indirect: dropout + LayerNorm makes high replay ratios work
   (DroQ), latent self-prediction losses carry most of the gain in
   SPR/EfficientZero/TD-MPC2, and masking at the latent/feature level (MLR,
   M-CURL, MTM, HRSSM) helps where tested. Almost all masking results are on
   pixels or offline state data; none is on a real robot from scratch.
4. No paper we found reports dropout/noise injected into a DreamerV3 RSSM
   deter/stoch state as a sample-efficiency method. That is open ground, cheap
   to try, and should be treated as an experiment, not an expected win.

## 1. Modality-agnostic augmentation and perturbation

"Modality-agnostic" here means the perturbation acts on a vector of
activations or latents, so the same code works for image or proprio input.

| method | what is perturbed | loss | reported gain | modality-agnostic? |
|---|---|---|---|---|
| Dropout (Srivastava 2014, JMLR) / DropConnect (Wan 2013, ICML, no arXiv) | hidden units / weights | unchanged task loss | regulariser; no RL sample-efficiency claim by itself | yes |
| DroQ (Hiraoka 2021, arXiv 2110.02034) | dropout + LayerNorm in Q-nets | TD loss | matches REDQ sample efficiency at UTD 20 with a small ensemble; used in "Walk in the Park" (A1 walks in 20 min) | yes |
| DropBlock (Ghiasi 2018, arXiv 1810.12890) | contiguous regions of conv feature maps | task loss | ImageNet/detection; for us it reduces to dropping groups of units | conv-specific; group dropout is agnostic |
| Zoneout (Krueger 2016, arXiv 1606.01305) | RNN hidden units randomly keep previous value | task loss | supervised RNN regulariser; maps directly onto the GRU deter state | yes |
| Observational dropout (Freeman, Metz, Ha 2019, arXiv 1910.13038) | whole observation replaced by model's own prediction with prob p | RL reward only | world model emerges without a forward-prediction loss; not a sample-efficiency result | yes |
| HRSSM (Sun et al. 2024, arXiv 2405.06263) | masked encoder features inside an RSSM + bisimulation | latent reconstruction of masked parts | robustness to distractors when added to DreamerV3 (visual) | mostly; tested on images |
| MLR (Yu et al. 2022, arXiv 2201.12096) | random spatio-temporal cube masks on pixel sequences | predict EMA-target latents of the unmasked sequence (cosine) | beats CURL/DrQ/SPR-type baselines on DMC-100k and Atari-100k | input masking is pixel-shaped; the latent-target loss is generic |
| M-CURL (Zhu et al. 2020, arXiv 2010.07470) | frame features masked, transformer reconstructs from context | contrastive (InfoNCE) | beats CURL on 14/16 DMC and 21/26 Atari tasks | yes at the feature level |
| MWM (Seo et al. 2022, arXiv 2206.14244) | conv features masked (MAE) + early-conv; reward prediction aux | MAE recon + DreamerV2 WM loss | outperforms DreamerV2 on Meta-World / RLBench visual manipulation | no, visual |
| MTM (Wu et al. 2023, arXiv 2305.02968) | random subsets of state/action tokens in trajectories | reconstruct full trajectory | one net acts as forward/inverse model and offline agent; representations speed up RL | **yes, tested on proprio states** (offline) |
| SPR (Schwarzer et al. 2020, arXiv 2007.05929) | augmented input; latent model rolled K=5 steps | cosine between predicted and EMA-target future latents | Atari-100k median HNS 0.415 (from ~0.16 for prior SOTA) | loss yes; their augmentation is image-based (ablation: loss helps even without aug) |
| EfficientZero (Ye et al. 2021, arXiv 2111.00210) | none at input beyond image aug | SimSiam consistency between predicted and encoded next latent | Atari-100k mean 190% / median 116% HNS; consistency loss is the largest ablation term | loss yes |
| TD-MPC2 (Hansen et al. 2023, arXiv 2310.16828) | none (state tasks); latent dynamics rolled H steps | MSE latent consistency to stop-grad encoded future + reward + TD; no decoder | strong on 104 state/pixel tasks; consistency loss is the representation signal | yes |
| TCRL (Zhao et al. 2023, arXiv 2306.09466) | none | temporal-consistency latent loss only (cosine > MSE) | 2.4x faster training on DMC Humanoid/Dog than ensembles at matched sample efficiency | yes, state-based |
| Denoising AE (Vincent 2008, ICML) / denoising VAE (Im et al. 2015, arXiv 1511.06406) | input corrupted (noise or masking) | reconstruct clean input | better representations / tighter bound; no RL claim | yes |
| mixup (Zhang 2017, arXiv 1710.09412), manifold mixup (Verma 2018, arXiv 1806.05236), MixReg (Wang 2020, arXiv 2010.10814 **[ID unverified]**) | convex mixes of inputs or hidden layers (and targets) | mixed targets | MixReg: Procgen generalisation, not sample efficiency | yes in principle; mixing dynamic states has no clean target for a world model |
| RAD state-based (Laskin et al. 2020, arXiv 2004.14990) | proprio vector: Gaussian noise, random amplitude scaling | unchanged RL loss | mixed on OpenAI Gym state tasks (helps some, hurts others) **[details from memory]** | yes |
| DrQ / DrQ-v2 (Kostrikov 2020, arXiv 2004.13649; Yarats 2021, arXiv 2107.09645) | image shifts; Q-target averaged over augs | TD | large pixel-DMC gains; no state-input version | no |
| S4RL (Sinha et al. 2021, arXiv 2103.06326) | states: Gaussian/uniform noise, adversarial, mixup over 7 schemes | CQL with smoothed Q over augmented states | gains on D4RL, MetaWorld, RoboSuite (offline) | yes, proprio |
| NoisyNets / parameter noise (Fortunato 2017, arXiv 1706.10295; Plappert 2017, arXiv 1706.01905) | weights of policy/Q | RL loss | exploration, not representation | yes |

Reading of the evidence for our setting:

- **What carries the gain is the target, not the corruption.** SPR, MLR,
  EfficientZero, TD-MPC2 and TCRL all predict a stop-gradient / EMA latent of
  a future or unmasked input. With proprio input, the input-level corruption
  matters less (state vectors have no spatial redundancy to exploit) while the
  latent-prediction loss still applies.
- **Masking needs redundancy.** Masking 50-90% works for images and video
  because neighbouring patches/frames are redundant. Our 10-20 dim proprio
  vector has little redundancy within a timestep but strong redundancy across
  time at 65 Hz. So the useful masking axis is **time** (drop whole
  observations, as in observational dropout / MTM), not dimensions.
- **Dropout is the most proven "neuron-level" tool for high replay ratios**
  (DroQ, and BRO/SR-SPR in the companion survey). Its role there is to stop
  overfitting of value/Q nets when each sample is reused many times, which is
  exactly our regime (train_ratio 32-512).
- **Gaps:** no study of dropout/zoneout/noise on RSSM deter or stoch state for
  sample efficiency; no masked-reconstruction study on a real robot trained
  from scratch.

## 2. DreamerV4

Hafner, Yan, Lillicrap, "Training Agents Inside of Scalable World Models",
arXiv 2509.24527 (29 Sep 2025). Offline: 2,541 h of Minecraft contractor video
with actions; first agent to obtain diamonds purely from offline data (0.7% of
episodes; >90% up to stone pickaxe, 29% iron pickaxe).

| component | what it does | self-supervision signal |
|---|---|---|
| causal tokenizer (~400M) | per-frame patch transformer, causal in time, latents projected low-dim + tanh | masked autoencoding: patch drop rate p ~ U(0, 0.9) per image, masked patches replaced by a learned embedding; loss MSE + 0.2 LPIPS |
| dynamics (~1.6B) | block-causal transformer: space-only attention, time attention every 4th layer, GQA; 192-frame context | shortcut forcing: diffusion forcing (each frame has its own noise level τ) + shortcut models (step-size conditioning, bootstrap loss for large steps); predicts clean x not velocity; loss weight w(τ)=0.9τ+0.1 |
| sampling | autoregressive in time, K=4 denoising steps per frame; past context lightly noised (τ_ctx=0.1) | real-time interactive inference on one GPU (no FPS stated in text) |
| agent finetune | agent tokens with task embedding; they attend to everything, nothing attends back; policy and reward heads with multi-token prediction L=8 | BC + reward NLL, video loss kept on |
| imagination RL | rollouts from dataset contexts; transformer frozen, only policy and value heads trained | value: TD(λ), γ=0.997; policy: PMPO (sign of advantage, positive and negative sets averaged separately, α=0.5) + reverse KL to frozen BC prior, β=0.3 |

How causality and masking give self-supervision: the tokenizer must
reconstruct whole frames from as little as 10% of patches, which forces
spatially consistent codes. In the dynamics model every frame gets an
independent noise level, so the same training step teaches denoising from
pure noise (generation) and from almost-clean context (filtering); the causal
mask means each position is a next-step prediction task. Treating "how
corrupted is this latent" as an input is the modality-agnostic idea most worth
borrowing: it is masking/noise on latents with a known corruption level.

What DreamerV4 does **not** provide: no ablation numbers in the arXiv text we
could read (the paper says an ablation study exists; figures not parsed), no
online/real-robot result, no evidence about rollouts per start. PMPO is from
Abdolmaleki et al., arXiv 2410.04166 (ICLR 2025).

### 2.1 Multi-rollout imagination: what the evidence says

| scheme | starts per update | rollouts per start | evidence |
|---|---|---|---|
| DreamerV1-3 (arXiv 2301.04104 for V3) | every replayed step, B·T | 1 | V3 reports larger replay ratio and larger model both raise data efficiency; no ablation on starts |
| DreamerV4 | one per dataset context | 1 | chosen for diversity and memory, not ablated |
| MBPO (Janner et al. 2019, arXiv 1906.08253) | many replay states | short branched rollouts, ~400 model rollouts per env step | more model rollouts per real step help as long as horizon stays short; long rollouts compound error |
| MuZero Reanalyse (Schrittwieser 2021, arXiv 2104.06294) | replayed states re-searched with the latest model | MCTS (many simulated branches per state) | spending compute on re-planning old data improves data efficiency; EfficientZero uses it too |
| TD-MPC2 | replay states | MPPI: hundreds of sampled action sequences per state at acting time | planning compute, not learning-signal compute |
| GRPO-style group baselines (LLM RL) | — | N samples per prompt, advantage vs group mean | lower-variance advantage without a critic; untested in Dreamer **[our extrapolation]** |

Bottom line: there is good evidence that *spending more compute per real
sample* helps (replay ratio, reanalysis, MBPO model rollouts), but none that
N>1 rollouts from the same RSSM start beats spending that compute on more B·T
starts or more gradient steps. With 1024 highly correlated starts per batch
(65 Hz neighbours are nearly identical), extra rollouts per start may add
little diversity. It is cheap to test because it does not touch the world
model update.

### 2.2 Replay ratio, resets, plasticity

- Primacy bias (Nikishin et al. 2022, arXiv 2205.07802): early data
  overfits the nets; periodic resets of the last layers (or all of the
  actor-critic) recover learning and allow higher replay ratios.
- D'Oro et al., "Sample-Efficient RL by Breaking the Replay Ratio Barrier",
  ICLR 2023 oral (OpenReview OpC-9aBBVJe, no arXiv ID found): with full or
  partial resets, SAC and SPR scale to an order of magnitude more updates per
  sample (SR-SAC, SR-SPR) on DMC and Atari-100k.
- BBF (Schwarzer et al. 2023, arXiv 2305.19452): SPR + larger nets + resets +
  annealed n-step/discount; human-level Atari-100k.
- Shrink-and-perturb (Ash & Adams 2019, arXiv 1910.08475) is the soft version
  of a reset.

On a real robot a reset of the acting policy costs real falls; the usual fix
is to reset only the learner copy and keep acting with the old policy until
the reset one recovers in imagination (our own idea; not from these papers).

## 3. What to try on our robot (ranked)

Rank = expected value per hour of work, favouring modality-agnostic latent
perturbation and multi-rollout imagination. All are sim-first (fitted MuJoCo,
time-to-balance and late-window score over ≥3 seeds), then one live run.

1. **Imagination repeats N per start (`agent.imag_repeat`).** In
   `agent.py` around `self.dyn.starts(...)` (lines ~1043 and ~1521), tile the
   start entries N times along batch (N=1, 2, 4, 8) before `imagine`; keep the
   stochastic prior and policy sampling so branches diverge. Optionally
   subtract the per-start mean return from advantages (group baseline).
   Expected: lower-variance actor gradient, smoother learning curves; uses
   spare GPU without changing world-model data reuse. Risk: little gain
   because B·T starts are already correlated; more exploitation of model
   error; memory grows linearly (reduce `imag_last` to compensate and compare
   at fixed rollout budget N·K_imag).
2. **Latent dropout / zoneout in the RSSM during WM training.** Apply dropout
   (p 0.05-0.2) to the deter state and img/obs MLP hidden layers, or zoneout
   on the GRU update, only in the training `observe` pass; disable in policy
   and imagination. Loss unchanged. Expected: less overfitting at high
   train_ratio (the DroQ mechanism moved into the world model), allows
   pushing train_ratio further. Risk: no direct precedent for RSSMs; too much
   dropout blurs the posterior and slows early learning, which matters most
   at a 3 min budget.
3. **Observational dropout in time (posterior→prior swap).** In `observe`,
   with probability p per step (0.1-0.3), feed the prior instead of the
   posterior into the next step (i.e. ignore the encoder token), while the
   decoder/reward losses still target the true observation. Expected: the
   dynamics are trained on multi-step open-loop prediction, which is what
   imagination uses; a modality-agnostic "masked latent reconstruction".
   Risk: higher WM loss early; interacts with KL balancing (keep dyn/rep KL on
   unmasked steps only).
4. **Self-predictive latent loss (SPR/TD-MPC2-style).** Add a head that
   rolls the prior k=1..5 steps with the real actions and regresses (cosine)
   onto a stop-gradient, EMA-target encoder embedding of o_{t+k}. Weight 0.1-1
   relative to the decoder loss. Expected: most robust gain source in the
   literature; keeps the latent useful for control rather than for
   reconstructing noisy sensor dims. Risk: more code (EMA encoder, new loss);
   with a decoder already present the marginal gain may be small.
5. **Proprio input masking/noise (denoising WM).** Before the encoder, add
   Gaussian noise (σ ≈ 0.1 of per-dim running std) and zero random dims
   (p 0.1) — decoder still targets the clean obs. Expected: robustness to
   sensor glitches on the phone and a mild regulariser. Risk: lowest upside;
   noise on tilt rate can hide the signal balance relies on; RAD reports mixed
   results for state noise.
6. **Learner-side resets at high replay ratio.** Every M updates, reset the
   actor-critic heads (or shrink-and-perturb, α=0.8) of the learner copy only;
   push to the robot after imagined return recovers. Pair with item 2 and
   train_ratio 2-4x current. Expected: the replay-ratio scaling seen in
   SR-SPR/BBF. Risk: our runs are only ~10k-20k robot steps; primacy bias may
   not have time to bite, and a reset that reaches the robot costs falls.

Not recommended now: pixel-style augmentation (DrQ/RAD shifts, DropBlock),
latent mixup (no clean dynamic target), or a DreamerV4-style transformer
tokenizer / shortcut dynamics (large rewrite; its gains were shown offline on
video at TPU-pod scale, not for low-dim online control).
