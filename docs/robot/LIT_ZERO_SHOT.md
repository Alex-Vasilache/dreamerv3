# Zero-shot learning for the robot: a focused literature review

2026-11-09. The PI's question is how the robot could act well with zero, or
almost zero, task-specific online samples. Today it learns balancing from
scratch in about 3 min live, and command following in about 10 min in the
fitted sim (`PLAN.md`, `SIM_COMMAND.md`). It runs DreamerV3 at 65 Hz on
proprio input.

"Zero-shot" means four different things in the literature. They differ in
what is zero and what is paid for up front:

| sense | zero of what | paid for with | section |
|---|---|---|---|
| A. zero-shot sim-to-real | real-robot samples before deployment | a simulator, randomisation, sometimes an adaptation module | 1 |
| B. zero-shot RL (unsupervised RL) | reward-specific training for a *new task* | reward-free pretraining on broad data in the same environment | 2 |
| C. foundation world models / in-context RL | gradient steps at test time | very large pretraining sets, or meta-training over many tasks | 3 |
| D. task-agnostic exploration pretraining | task-reward samples (the exploration is reward-free) | a reward-free phase in the target environment | 4 |

Only A removes real samples for a task we already know (balancing). B and D
remove the task-specific *learning* when the task changes, for example a new
command or reward. C removes the *gradient steps*, but not the data that
adaptation needs.

Verification: every arXiv ID below was checked against the arXiv record
(title, authors, date) on 2026-11-09, and the numbers against the abstract or
paper text. Anything still marked "(unverified)" was not found in the paper.

## 1. Zero-shot sim-to-real (sense A)

| work | setup | what transfers | real samples / time | needs |
|---|---|---|---|---|
| Domain / dynamics randomisation (Tobin et al. 2017, arXiv 1703.06907; Peng et al. 2018, arXiv 1710.06537) | train in sim with randomised visuals or dynamics | the policy, unchanged | 0 | a randomisation range that covers the real system; costs some nominal performance |
| System ID + actuator net (Hwangbo et al. 2019, arXiv 1901.08652) | fit a neural actuator model from real logs, then RL in sim | the policy | 0 for the task; under 4 min of real actuator data (>1M samples at 400 Hz, 12 actuators in parallel) | a measured model of the part the rigid-body sim gets wrong |
| Learning to Walk in Minutes (Rudin et al., CoRL 2021, arXiv 2109.11978) | PPO on thousands of parallel Isaac Gym robots | the policy, to ANYmal | 0 real; under 4 min of training for flat terrain, about 20 min for rough, on one GPU | a fast, massively parallel sim |
| RMA (Kumar et al., RSS 2021, arXiv 2107.04034) | base policy plus an adaptation module that infers "extrinsics" (mass, friction, terrain) from recent state-action history; both trained in sim | both modules; adaptation happens online, without gradients | 0; deployed on A1 without fine-tuning over rocky, slippery and deformable ground, grass, sand, pebbles and stairs; adapts in fractions of a second | broad randomisation of the parameters the module must infer |
| Real-world humanoid with a transformer (Radosavovic et al. 2023, arXiv 2303.03381) | causal transformer over proprio-action history, large-scale RL in randomised sims | the policy; it adapts in-context to terrain | 0; Digit walks outdoors on unseen terrain | thousands of randomised environments; a longer context helped (their context-length ablation) |
| DayDreamer (Wu et al., CoRL 2022, arXiv 2206.14176) | DreamerV2 trained only on the real robot (not sim-to-real) | nothing; the baseline for real-only learning | A1 walks after about 1 h | none of the above |
| A Walk in the Park (Smith et al. 2022, arXiv 2208.07860) | DroQ, high update-to-data ratio, real only | nothing | A1 walks in about 20 min | careful MDP design |
| SimDist (Levy et al., RSS 2026, arXiv 2603.15759) | pretrain a latent world model in sim, plan with it; on the robot keep encoder, reward and value, fine-tune only the latent dynamics (manipulation and quadruped locomotion) | encoder, reward, value | the abstract claims rapid improvement where prior fine-tuning methods stall or degrade; no numbers in the abstract | sim close enough that the latent space and value are reusable |
| Online imitation-pretrained WM recipe (Wang et al. 2025, arXiv 2510.02538) | imitation learning: world model pretrained with online imitation in sim, then offline fine-tuning on limited real expert data; manipulation | the world model | success +23.3% sim-to-real (+31.7% sim-to-sim) over offline imitation baselines | real expert demonstrations |

What the field agrees on: zero-shot transfer of a *policy* works when the
gap is covered by randomisation or inferred online (RMA, the Digit
transformer). Naive fine-tuning of a sim-pretrained model-based agent on the
robot often *gets worse first*. SimDist exists because of that, and our own
logs show it too (below). DayDreamer and Walk in the Park are not zero-shot.
They are the real-only references that sim-to-real has to beat, and our
3 min balance time already beats both.

**Our own evidence** (`SIM_COMMAND.md` §9, round 3/4, October):
- Sim policies already transfer zero-shot. Before any update, e2605 on carpet
  tracked forward at 0.71 (0.84 in its sim). e2995 on the table stood almost
  still, with tilt SD 0.6-1.0°, which is better than in its own sim.
- Warm-starting with the learner on broke both (e2732, e2996) within the first
  minute. The learner did about 178k replayed samples on the first 536 real
  steps (prefill is one batch at train_ratio 8192), and dyn loss rose from
  1.54 to 2.5.
- So the transfer is not the bottleneck. The fine-tuning schedule is. This
  failure matches the SimDist motivation and the late-collapse ratio finding
  in `PLAN.md` (CR1/CR2).

## 2. Zero-shot RL / unsupervised RL (sense B)

Setup: one reward-free pretraining phase learns a representation and a family
of policies. At test time a reward is given, as a few labelled states, a
reward function or a goal state. The policy for it is read off with no
planning and no further RL.

| work | method | test-time spec | reported result | needs |
|---|---|---|---|---|
| Forward-backward (FB) (Touati & Ollivier, NeurIPS 2021, arXiv 2103.07945) | learn F(s,a,z) and B(s') such that F·B approximates the successor measure; policy π_z for each z | z = E[B(s) r(s)] over labelled states, or B(goal) | near-optimal for rewards given after training; mazes, pixel Ms. Pacman, a simulated arm | a reward-free dataset that covers the states the rewards care about |
| Does zero-shot RL exist? (Touati, Rapin, Ollivier, ICLR 2023, arXiv 2209.14935) | systematic study: FB versus successor features (SF) with many base features, offline on URLB/ExORL buffers | as above | FB reaches 81% of supervised offline TD3 averaged over tasks and buffers, 85% on the best (RND) buffer, consistently. SF only good with Laplacian features; AE, inverse-dynamics, transition-model, contrastive and APS features inconsistent | a good exploration buffer (5M transitions per ExORL buffer, 10M for the maze) |
| HILP (Park, Kreiman, Levine, ICML 2024, arXiv 2402.15567) | Hilbert-space state embedding where distance is temporal distance; latent-conditioned policy | goal, or linear reward via the embedding | zero-shot goal-reaching and reward tasks on simulated locomotion/manipulation, often beating specialised methods | offline unlabelled data |
| Meta Motivo / FB-CPR (Tirinzoni et al., ICLR 2025, arXiv 2504.11054) | FB regularised to imitate unlabelled mocap trajectories | tracking, goal pose, or reward | one model does all three zero-shot on a simulated humanoid, competitive with task-specific methods | a subset of AMASS mocap (8,902 training motions, ~29 h) plus 30M online interaction steps in MuJoCo (SMPL humanoid) |
| BFM-Zero (Li et al., ICLR 2026, arXiv 2511.04131) | FB-based behaviour foundation model with reward shaping, domain randomisation and asymmetric history-dependent training | as Motivo; few-shot optimisation adaptation | real Unitree G1: pose reaching, reward optimisation, adaptation; no numbers in the abstract | a sim-to-real stack as in section 1 |
| FB-MEBE (Hu et al., 2026, arXiv 2603.25464) | online FB with maximum-entropy *behaviour* exploration plus a naturalness critic, no external dataset | reward at test time | zero-shot deployment of recovered policies to a real quadruped without fine-tuning; no numbers in the abstract | a sim; exploration is the bottleneck it addresses |

What transfers: one network for a whole *family* of rewards. The failure mode
is data coverage. If pretraining never visited "drive backward at 0.5 m/s
while upright", no z will produce it. Balancing is an unstable, narrow region
of state space, so coverage there takes deliberate exploration. FB-MEBE
targets exactly this. None of these methods is DreamerV3-based. FB is a
model-free actor-critic over latent z.

## 3. Foundation world models and in-context adaptation (sense C)

| work | setup | what is zero-shot | scale |
|---|---|---|---|
| Genie (Bruce et al., ICML 2024, arXiv 2402.15391) | 11B video world model with a latent-action interface learned without action labels | playable environments from one image | 30k h of 2D-platformer video, filtered from over 200k h of internet game video |
| Dreamer 4 (Hafner, Yan, Lillicrap 2025, arXiv 2509.24527) | shortcut-forcing transformer world model; RL entirely in imagination | first agent to get Minecraft diamonds from offline data only; real-time inference on one GPU | 2,541 h of Minecraft contractor video (VPT); action conditioning learnt from a small action-labelled subset; trained on 256-1024 TPU-v5p; no robotics experiments |
| RL² (Duan et al. 2016, arXiv 1611.02779) | an RNN whose hidden state carries the "fast" learning algorithm, trained by "slow" RL over a task distribution | adaptation to a new MDP with no gradient steps | bandits, small MDPs, vision navigation |
| Algorithm Distillation (Laskin et al., ICLR 2023, arXiv 2210.14215) | transformer trained on whole learning histories of a source RL algorithm | improves in-context, without weight updates, more data-efficiently than the source | many learning histories per task family |
| Model-based meta-RL on a real millirobot (Nagabandi et al., ICLR 2019, arXiv 1803.11347) | dynamics model meta-trained to adapt from recent experience (recurrent ReBAL or gradient GrBAL), then MPC | online adaptation to a lost leg, new terrain, slopes, calibration error | about 30 min of real data per training terrain, three terrains (1.5-3 h equivalent in the sim tasks) |
| RMA and the Digit transformer (section 1) | in-context system identification from proprio history | floor or terrain changes at run time | sim randomisation |

Dreamer's RSSM is already an in-context learner of this kind. Its recurrent
state is a filter over recent history. If it is trained on a *randomised*
sim, it can infer floor friction or balance-point offset from a few seconds
of proprio. That is the RMA idea without a separate module. Foundation-scale
world models (Genie, Dreamer 4) need data we do not have, and they solve
the perception and semantics problem, which is not ours. Our robot's
dynamics are small and already modelled.

## 4. Task-agnostic exploration pretraining (sense D)

| work | setup | zero-shot claim | needs |
|---|---|---|---|
| Plan2Explore (Sekar et al., ICML 2020, arXiv 2005.05960) | Dreamer world model; the explorer maximises ensemble disagreement (expected information gain) in imagination | after reward-free exploration, new tasks are solved in imagination with no extra environment samples, close to an oracle with reward access (20 DMC tasks from pixels) | a reward-free phase in the *target* environment; a reward head trained from labels at test time |
| LEXA (Mendonca et al., NeurIPS 2021, arXiv 2110.09514) | world model plus an explorer and an "achiever" (goal-conditioned) trained in imagination | reaches user-given goal images zero-shot; 40-task benchmark | goals expressible as observations |
| Choreographer (Mazzaglia et al., ICLR 2023, arXiv 2211.13350) | skills discovered in the world model's latent space; a meta-controller picks and chains skills in imagination | adapts to downstream URLB tasks, beating earlier pixel and state methods | a reward-free phase, then a short adaptation |

These are closest to our stack, because all three are Dreamer-family. The
cost is the reward-free phase itself. On the real robot that phase is
falls, so it only makes sense in the sim.

## 5. What this means for our robot (ranked)

**1. Start live from the sim-pretrained world model and policy, and fix the
fine-tuning schedule (sense A).** The policy part of zero-shot already works
(e2605, e2995). What failed was fine-tuning. Change three things:
- Prefill several hundred real steps before the first update, with the
  policy still acting.
- Cap the replay ratio low at first (about 64-128) and raise it later. This
  is the reverse of the CR2 schedule.
- Optionally freeze the encoder, reward and critic and adapt only the
  dynamics (SimDist), or mix sim replay into the batch.

Cost: about a day of code (prefill length, ratio ramp, freeze flags) plus
one sim-to-sim test, then one live session. Risk: low. The worst case is
today's 3 min. Expected gain: balancing from step 0 instead of minute 3, and
commands without the 10 min. The sim's remaining errors (inertia, backlash,
balance point) set how much fine-tuning is still needed.

**2. Make the sim policy adapt in-context to the floor (senses A and C), and
run it with the learner off.** Widen the randomisation that already exists
(sensor lag 0-15 ms, floors) to friction, balance-point offset, inertia,
backlash and slope. Rely on the RSSM state, perhaps with a longer burn-in,
to infer them, as RMA and the Digit transformer do with proprio history.
Measure it in the sim on held-out floor parameters, then live on the
Germany floor. Cost: sim sweeps only (about 1-2 days of GPU), no new
architecture. Risk: medium. Broad randomisation lowers nominal performance,
and a 65 Hz unstable plant may need more history than an RSSM keeps. It
pairs with option 1: zero-shot first, then gentle fine-tuning only if needed.

**3. Reward-free FB pretraining in the sim, so that new commands are
zero-shot (sense B).** This is the only option that makes *new tasks* free.
Examples are new command profiles, "stay near here", or emotion-driven
rewards (`PLAN.md` step 3, "task-reward-free learning"). Train FB, or
FB-MEBE-style exploration, in the fitted sim. A command becomes
z = E[B(s)·r_cmd(s)] at run time. Then deploy with the section-1
randomisation (BFM-Zero shows this on a humanoid). Cost: high, 2-4 weeks. FB
is a separate model-free learner, not a Dreamer config, and coverage of the
upright manifold needs deliberate exploration. Risk: medium-high. In the
2023 study zero-shot quality was 81% of supervised on average (85% on the
best buffer). Our balance margin is thin, so 80-85% may mean falling. A cheaper version inside Dreamer:
train a command-conditioned policy over randomly sampled commands in sim
(already done for `robot_command`). That is zero-shot over the command
family we choose, with no FB.

**4. Plan2Explore/LEXA-style reward-free Dreamer pretraining in the sim
(sense D).** Explore the sim with disagreement reward, then fit reward heads
for balance or command tasks and train the actor in imagination with zero
new environment steps. Cost: medium. Plan2Explore-style disagreement
ensembles are a known Dreamer add-on, but our repo does not have one wired
for the robot (not checked). Risk: medium. It speeds up *later* tasks more
than it speeds up the first balance, and it does nothing for sim-to-real.

**5. Foundation world models or algorithm distillation (sense C at scale).**
Not recommended now. These methods need far more data than we have (about 1.5 h of
real meta-training data even in the millirobot case; internet-scale data for
Genie and Dreamer 4), and they target perception, which our proprio task
does not need. Revisit only if the camera and charger-finding work (step 7)
becomes the priority.

**Could not verify:** real-data amounts for SimDist, BFM-Zero and FB-MEBE
(the abstracts give none and the papers were not read in full).

Checked 2026-11-09: 24 citations verified (plus the SIM_COMMAND.md §9 and
PLAN.md CR1/CR2 references), 5 corrected: zero-shot RL "85%" -> 81% average,
85% on the best buffer; Genie 200k h -> 30k h used (filtered from 200k);
RMA terrains (now as listed in the abstract); millirobot 1.5-3 h -> ~30 min per
terrain; arXiv 2510.02538 is online imitation pretraining, not reward-free.
Five "unverified" flags resolved from the papers (Meta Motivo data, ExORL
size, Hwangbo actuator data, Plan2Explore 20 tasks, Dreamer 4 data).
