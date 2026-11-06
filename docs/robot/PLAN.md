# Robot plan, in order (from 2026-11-06)

Alex's plan: one step at a time, each started only when the one before works.
Each step below lists what already exists, so a new session starts from it
instead of redoing it. Work done ahead of its turn is parked here, not used.

1. Fast live demo on the Germany floor (+ late collapse)  <- current
2. Emotion computation, including pain: decide and document
3. Task-reward-free learning?
4. Face visualisation
5. Neuroscience literature on emotion and learning
6. Emotions to speed up learning
7. Later: live joystick control, camera + charger finding, HRI, RHI/RRI, general items

The full plan text is at the end.

## 1. Fast live demo, Germany floor (+ late collapse)

Goal: the robot learns to balance in a few minutes on the lab floor, and
keeps balancing, not getting worse later.

**Where it stands.**
- The best live recipe is **e30114**. It reruns e30107 with the phone position
  bug fixed.
  - Upright 0.70-0.94 at minutes 3-7.
  - 3-21 crossings a minute.
  - Alex stopped it at minute 7.
- **e30107** had minutes 7-9 at October level, then fell back. That fall-back
  is the "late collapse" seen live.
- Command line, run history and bugs: `docs/robot/LIVE_20261105.md`.
- The run dirs e30101-e30114 are on Saion `/work` and are not archived yet.

**The same recipe in the fitted sim.**
- Settings: sim table, n = 4, `docs/robot/EMOTION.md` "Sim results".
- Without pushes, it is up by minute 3 in 4/4 seeds.
- Crossings then drift back up after minute 15, to 15-20 a minute
  (`figs/emotion/night_1106/figs/ec_np.png`, black line).
- The sim may therefore show a mild form of the late collapse. That is a cheap
  place to study it.

**Earlier late-collapse finding** (`SIM_COMMAND.md` §5).
- On carpet the policy goes still, creeps onto a bumper 25-55 min in, and stays.
- Cause: the FIFO replay had evicted every example of getting off a bumper.
- Keeping all experience (`replay.size 1e6`) fixed it.
- The `robot` block keeps 1e6 (`replay.size`, `online_actor_replay_size`), and the live runs were shorter than the old 26 min buffer anyway, so replay eviction is unlikely to explain the live fall-back.

**Late collapse = learner replay ratio (2026-11-06 evening, rounds CR1/CR2).**
- The learner is uncapped (`run.train_ratio 8192`), so its replay ratio is set
  by the GPU. A V100 gets ~420 replayed steps per robot step and a P100 ~280.
  Live, e10100-e30101 trained at ~285 and settled to 1-8 crossings a minute,
  e10100 for 15 min. From e30102 on, the trainer ran at 430-560 (V100 speed),
  and those runs fall back after minute 7.
- Sim rounds EA-EC split their seeds over V100/P100. Paired over 20 arms,
  V100 had +3.6 ± 1.2 crossings/min after min 15 (worse in 16/20), with no
  faster learning (`figs/collapse/ratio_evidence.png`).
- CR1 (e19490-e19505, e30114 recipe, sim table, no pushes, n = 4, seeds 791-794):

  | train ratio | first min upright ≥ 0.9 | crossings/min, min 15-29 |
  |---|---|---|
  | uncapped (~420) | 3.8 ± 1.3 | 18.4 ± 3.1 |
  | 256 | 4.5 ± 1.1 | 12.6 ± 1.9 |
  | 128 | 7.8 ± 0.8 | 9.5 ± 2.0 |
  | 64 | 10.5 ± 1.8 | 10.6 ± 1.9 |
  | uncapped, 3 min -> 128 | 3.2 ± 0.4 | 13.9 ± 6.6 |
  | **uncapped, 5 min -> 128** | 3.5 ± 0.9 | 10.1 ± 1.8 |
  | uncapped, 8 min -> 128 | 3.2 ± 0.4 | 9.0 ± 2.7 |
  | uncapped, 5 min -> 64 | 3.5 ± 0.9 | 13.6 ± 4.6 |

  The last four rows are CR2 (e19506-e19521), using the new
  `online_ratio_switch` (actor steps) and `online_ratio_late` keys (off by
  default). At the switch, the debt from the uncapped phase is dropped.
  Figure: `figs/collapse/cr_arms.png`.
- **Live pick:** the e30114 flags plus `--online_ratio_switch 19500
  --online_ratio_late 128` (5 min at 65 Hz). It needs no app change.
- Caveat: the sim's after-15-min drift is milder than live. The live fall-back
  is at min 7-10, so the 5 min switch comes before it.

**Candidate fixes, untested live.**
- *Fear critic + catch reflex:*
  - Flags: `--env.robot.cost_obs True --agent.fear.weight 1.0 --env.robot.affect_catch 0.5`.
  - In sim on the e30114 recipe it cuts crossings from 13-16 to about 5.5 a
    minute, and keeps them low after minute 15.
  - Fear alone needs no app change. The catch needs app line 1 below.
  - This is step-6 machinery, but it is the strongest demo fix found so far.
    Whether to use it now is Alex's call.
- *Early sign-flipping:* the early policy flips sign at full power in about
  half of its steps. Half smoothing (`action_rate_quad 0.25`) is in the
  `robot` block. Full smoothing lost a sim seed.

**Tools.**
- `tools/robot/dev/demo_stats.py <run>`: per-minute upright, crossings and
  wobble of a live run.
- `tools/robot/dev/r6/emostats.py` and `fig_emo4.py`: the same for sim rounds.
- Sim with the phone's own loop:
  - `--env.robot.sim_phone True`;
  - pushes: `sim_push every:lo:hi`;
  - walls: `sim_walls` (`docs/robot/SIM_PHONE.md`, `EMOTION.md`).
- Param files: `sbatch/sim_bal_e30114.txt`, `sim_bal_emo5.txt`, `sim_bal_emo6.txt`.

**Open questions.**
- Does e30114 hold past minute 9 live? It has never run longer than 7 minutes.
- Is the sim's after-minute-15 drift the same thing as the live late collapse?

## 2. Emotion computation, including pain

**Built** (`docs/robot/EMOTION.md`, `dreamerv3/deploy/emotion.py`).
- *Valence* = TD error / RMS. *Mood* = z(V).
- *Arousal* = σ(novelty + unexpected surprise + reward error − empowerment
  + helplessness + 3·pain).
  - Novelty is the posterior-prior KL.
  - Unexpected surprise is total surprise minus expected surprise.
- *Dominance* = empowerment.
- *Pain* (`pain_signal`) is a sensation from the body, not a prediction:
  - it rises over the last 2° before a bumper;
  - plus the size of any reflex overriding the policy.
  - It raises arousal and lowers face valence.
  - It is not its own axis yet.
- *Fear* = the fear critic's predicted pain. It is shown only when trained
  with `cost_obs`.
- Literature: `docs/robot/EMOTION_LITERATURE.md` (pain is §8).

**Findings to decide on.**
- Arousal as defined falls at bumper hits, by about 1.4 SD:
  - The drop is carried by novelty, because the model is confident there.
  - It rises little after pushes.
  - |TD valence| + pain ("salience") is what marks those moments: 97% of its
    events fall within 1 s of a hit.
- Report, section 3: `figs/emotion/night_1106/report.html`, figure `trig.png`.
- So the definition of arousal, and whether pain is an axis or a term inside
  arousal and valence, is the open decision for this step.

## 3. Task-reward-free learning?

- Not started.
- Empowerment collapses at contact, by -3 to -7 nats against a bumper
  (`EMOTION.md`, "First result").
- So "avoid low empowerment" would draw the bumper boundary unaided. That is
  the starting evidence for this step.

## 4. Face visualisation

- *Now:* `tools/robot/face/face.html`.
  - Springs, a 3D valence-arousal-dominance box, a pain wince, fear.
  - Served by the app at `/face`.
  - Changes: `LIVE_20261105.md`, "App and tooling".
- *Asked for:* a pixel robot face with fewer degrees of freedom (Cozmo, Wall-E,
  Reachy Mini, Duck), the expression literature, and the face separated on
  screen from the info panels (axes, reward, actions).

## 5-6. Literature on emotion and learning; emotions to speed up learning

Done ahead of turn on 2026-11-06. The code is in, and every part is off by
default.

**Results: sim, n = 4 per arm, with pushes and walls.**
- *Fear critic* (pain learnt as its own cost, not as reward): the one clear win.
  On the robot defaults, crossings fell from 14.1 to 4.0 a minute.
- *Catch reflex:* helps only on top of fear.
- *Arousal-triggered weight changes:* no help, and lost seeds. The parts are
  an lr multiplier after the optimizer's normalisation, burst updates, and an
  immediate push.
- *Vigour* and *arousal-scaled exploration:* no help.
- *Earlier negatives:* arousal-weighted imagination, and surprise or recency
  replay.

**Where it lives.**
- Settings, code map and results: `EMOTION.md`, "Emotion acting on behaviour
  and learning".
- Report: `figs/emotion/night_1106/report.html`, also at
  https://claude.ai/artifact/3TxrKuGtzwhbcYrByMpPwp.
- Literature summary: `EMOTION_LITERATURE.md`.

**Not yet on the phone.**
- Three dreamerBridge `main.py` lines:
  1. `act = runner.modulate(act)` after `runner.act`;
  2. `runner.configure(state.get('emotion'))` in `apply_pause`;
  3. `emotion_event=runner.event` in the obs message.
- Copy `dreamerv3/deploy/*.py` and the face page to the app with
  `tools/robot/dev/check_app_copies.sh --fix`.

## Sim data (Saion bucket, `results/dreamerv3/`)

| Runs | What |
|---|---|
| e19410-e19437 | Round EA |
| e19438-e19441 | e30114 recipe |
| e19442-e19465 | Round EB |
| e19466-e19489 | Round EC |
| e19490-e19505 | Round CR1 (train ratio) |
| e19506-e19521 | Round CR2 (ratio schedule) |

All runs include `logdir/emotion.csv` and `online_shared/experience`.
Experiment numbers: the robot series `e<N>` was last used at e19521 (sim) / e30114 (live).

## Full plan (Alex, 2026-11-06)

```markdown
### Live Demo Balancing Learning
- Fix fast learning for germany floor
- Look into late collapse (it can get to balancing, but then doesnt work and keeps getting worse?)

### Emotion+Pain Computation
- Define and compute emotional arousal and valence and empowerment
    - Valence defined as TD-error
    - Arousal:
        - Novelty (uncertainty about future states and rewards)
            - Excitatory: increases arousal
        - Surprise (prediction error of states and rewards)
            - Excitatory: increases arousal
            - Maybe to be modeled as unexpected surprise, by subtracting expected surprise (aleatoric, unpredictable aspects of environment such as pure static noise) from total immediate surprise.
    - Empowerment (agent's actions influence on future states and rewards)
        - Inhibitory: decreases arousal
            - Negative Reward: e.g. Learned helplessness - in long states of negative or very unpredictable stimuli, there is extremely high arousal, because the individual has no control (0 empowerment) of the outcomes.
            - Positive Reward: e.g. Gambling - states of high positive reward, but without control (0 empowerment), like operating a slot machine generate high arousal.
        - Another theory is that Empowerment acts as its own 3rd axis that determines dominance.
    - Maximizing all 3 terms (weighted in some way) is equivalent to striving for a neutral arousal state?
    - How is pain computed?
        - Does it have its own axis?

### Face Visualisation
- Reduce degrees of freedom in face explicitly to get away from uncanny valley
    - Make the display of "face" be obviously robot-like with "pixels" that glow and move
    - Research how other robots do it like Cosmo, Wall-E, Reachy Mini, Micro Duck and look deeply into how each of these are expressive and how they communicate and what expressions are generated and when and why (as a result of which "emotions" ?)
        - Also take inspiration from their design of the face and decide what is the best design for us
- Look into the literature of human/animal facial expressions and exactly what muscles / regions of face, tilting of entire body / face etc are triggered by which of our emotion correlates (high empowerment = high confidence/controll/dominance, high valence does what? high arousal does what?)
    - On what timescales do these emotions and facial expressions get integrated and displayed in biology?
- Improve the phone face visualization page
    - Separate the face explicitly and clearly from other display metrics that are informative. It has to be clear that one part is the actual "feeling" and face and expression of the robot as if it were alive, and the other parts are just for more information of the behind the scenes
    - Decide how to display the emotion metrics alongside the 3d and the 3 axes (+also how pain fits in this) such that it is clear what is happening, maybe we also have to show current reward there too in a visual clear way, and also maybe we show the actions the robot is outputing again in a visually informative way
    - And all this fit into a single page that is not overwheling, overloading sensoric information that looks clean and perfectly designed

### Emotions to speed up Learning
- Document Neuroscience Literature on emotions and learning
- Implement into robot

### Reward-Free Learning?
- Can emotions be used to "infer" task reward or something simillar
- E.g. states with low empowerment (hitting obstacles, hitting bumper) are to be avoided obviously and we want to aim for high empowerment
- How do other emotions come into play as a target for learning and what behaviour do they give rise to?

### Robot-Control Live?
- Train phone robot to not only balance, but follow directional commands so that it can be controlled with a joystick (forward-backward, left-right)

### Image + Proprio charger finding
- Extend policy input with front and back-cam images
- Design charger reward (instantaneous, hunger-like, etc?)
- Extend balancing reward with charger reward
- Train policy

### Human-Robot Interaction (HRI)
- Develop interface to immediately pass rewards to learning robot (web app) to steer behavior
- Add auditory input to the policy to allow learning of voice instructions

### Robot-Human / Robot-Robot Interaction (RHI/RRI)
- Add image, audio, text generation as actions to allow interaction of robot with humans and other robots

### General
- Arousal as intrinsic reward for training the agent
- HRL if no learning of long-horizon rewards
- Continual Learning techniques like normalization if losing learning plasticity/adaptation
- Should we switch to dreamerv4 like imagination batches from single starting point multiple action posibilities?
- Attention display if using multimodal input to show where focus is now
```

The order, from Alex's handwritten note: fast demo first. Only once it works
satisfyingly, decide and document how the emotion factors are computed,
including pain. Then whether they can run the robot without a task reward (can
a lack of empowerment set the boundaries: no sitting on a bumper, no hitting
obstacles). Then the face: pixelated and robot-like, fewer degrees of freedom,
following how Cozmo, Wall-E, the Duck and others express emotion and the
literature on how these emotions (VAD) produce facial expressions, with reward
and action displays kept separate on the screen. Only after that, emotions to
speed up learning and the neuroscience literature.
