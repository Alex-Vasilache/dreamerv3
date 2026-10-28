# DreamerV3 on the OIST smartphone robot

This repo trains DreamerV3 on the real
smartphone robot: a phone on a two-wheeled base, learning to balance and to
follow joystick commands. The policy runs **on the phone**; a learner on a GPU
(your computer or a Slurm cluster) trains on everything the phone records and
pushes it new weights every ~2-3 s.

```
  phone (dreamerBridge app)                         learner (this repo)
 ┌──────────────────────────────┐   observations   ┌──────────────────────────┐
 │ sensors -> policy -> wheels  │ ───────────────> │ replay -> world model    │
 │ 25 / 50 / 100 Hz, on device  │ <─────────────── │ -> actor-critic          │
 └──────────────┬───────────────┘   new weights    └──────────────────────────┘
                │ USB serial                        dreamerv3/main.py, env
 ┌──────────────┴───────────────┐                   embodied/envs/robot.py
 │ base (RP2040 firmware)       │
 └──────────────────────────────┘
```

The learner listens on a port; the phone connects to it over WiFi. So the
only network requirement is that **the phone can reach the learner's
address** (your laptop on the same WiFi, or a cluster compute node that the
WiFi can route to -- true for Saion from the OIST internal WiFi).

Contents: [Repos](#repos) · [Install](#install) · [Train](#train) ·
[Watch](#watch) · [Configs and reward](#configs-and-reward) ·
[Joystick](#joystick-and-command-following) · [Saved policies](#saved-policies) ·
[No hardware](#without-hardware) · [Flash a base](#flash-a-base) ·
[Troubleshooting](#troubleshooting) · [Tools](#tools) · [Background](#background)

## Repos

| repo | branch | what |
|---|---|---|
| [oist/dreamerv3](https://github.com/oist/dreamerv3) | `main` | learner, robot env, these tools |
| [Alex-Vasilache/smartphone-robot-android](https://github.com/Alex-Vasilache/smartphone-robot-android) | `dreamer-bridge` | phone app `dreamerBridge` (fork of [oist/smartphone-robot-android](https://github.com/oist/smartphone-robot-android)) |
| [Alex-Vasilache/smartphone-robot-firmware](https://github.com/Alex-Vasilache/smartphone-robot-firmware) | `telemetry-cache` | base firmware: motor watchdog and I2C recovery on top of upstream's fast-reply telemetry cache (fork of [oist/smartphone-robot-firmware](https://github.com/oist/smartphone-robot-firmware)) |

All commands below run from the dreamerv3 checkout.

## Install

### 1. Learner

```bash
git clone https://github.com/oist/dreamerv3.git
cd dreamerv3
python3 -m venv .venv          # Python 3.10-3.13 (tested on 3.13)
```

JAX for your machine, then the rest:

```bash
.venv/bin/pip install "jax[cuda12]==0.4.34"                          # Linux + NVIDIA GPU
.venv/bin/pip install jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1    # Apple silicon GPU
.venv/bin/pip install jax==0.4.34                                    # CPU only (tests)
.venv/bin/pip install -r requirements-robot.txt
.venv/bin/pip install tensorboard torch     # optional, for tools/robot/tensorboard.sh
```

The platform is picked automatically (CUDA on Linux, Metal on macOS when
`jax-metal` is installed). Force one with `DREAMERV3_PLATFORM=cpu`.

Your settings -- where the learner runs, the phone's address, cluster paths --
go in one gitignored file:

```bash
cp tools/robot/robot.env.example tools/robot/robot.env
```

and edit it; every key is explained there. With `ROBOT_TRAINER=local` (the
default) nothing else is needed. You also need `adb` (Android
platform-tools: `brew install --cask android-platform-tools`, or from Android Studio).

**On a Slurm cluster** (optional): clone this repo there, make a Python
environment with the CUDA JAX above, and set in `robot.env`:
`ROBOT_TRAINER=slurm`, `ROBOT_SSH` (an ssh alias that logs in without a
password prompt), `ROBOT_REMOTE_CODE` (the clone), `ROBOT_RUNS` (a directory
for runs on the shared filesystem), `ROBOT_SETUP` (the command that activates
that environment inside a job) and `ROBOT_SLURM_FLAGS` (partition and GPU).
A Saion example:

```bash
ROBOT_TRAINER=slurm
ROBOT_SSH=saion
ROBOT_REMOTE_CODE=/apps/unit/<Unit>/<you>/dreamerv3
ROBOT_RUNS=/work/<Unit>/<you>/robot
ROBOT_SETUP='module load python/3.11 && source /apps/unit/<Unit>/<you>/venv/bin/activate'
ROBOT_SLURM_FLAGS="-p gpu-v100 --gres=gpu:v100:1"
```

`train.sh` runs `git pull` in `ROBOT_REMOTE_CODE` before each submission, so
push your changes before starting a run.

### 2. Phone app

The phone is an Android phone with Developer options on (tap Build number 7
times) and **USB debugging** and **Wireless debugging** enabled. Build
needs a full JDK 17+ (Android Studio's bundled one works):

```bash
git clone -b dreamer-bridge https://github.com/Alex-Vasilache/smartphone-robot-android.git
cd smartphone-robot-android
cp config.template.json config.json      # values unused: train.sh points the phone at the learner
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"   # macOS + Android Studio
./gradlew :dreamerBridge:assembleDebug
adb install -r apps/dreamerBridge/build/outputs/apk/debug/dreamerBridge-debug.apk
```

Set `ROBOT_ANDROID_DIR` in `robot.env` to this checkout. More on the
Android side: `docs/getting-started.md` and `docs/dreamer-bridge.md` there.

**Connecting adb wirelessly.** First time, pair: Settings → Developer options
→ Wireless debugging → *Pair device with pairing code*, then
`adb pair <ip>:<pairing port>` and enter the code. Afterwards, with Wireless
debugging switched on, `tools/robot/phone_connect.sh` finds the (changing)
port and connects; set `PHONE_IP` in `robot.env` first. Over a cable, plain
`adb devices` works.

### 3. Base firmware

Bases need the firmware from `telemetry-cache`: ~7 ms serial replies (from
upstream's telemetry-cache work), plus this fork's motor watchdog, so the wheels
stop by themselves when commands stop. A base with the stock firmware
balances poorly and can spin a wheel at power-up. See [Flash a base](#flash-a-base).

## Train

1. Wireless debugging on, then `tools/robot/phone_connect.sh`.
2. Phone on the base, base switched on (switch on the bottom **UP**, Phone
   Side). Check both say `true`:
   ```bash
   adb shell dumpsys usb | grep -E 'host_connected|source_power'
   ```
3. Stand the robot upright with room to move, and start:

   ```bash
   tools/robot/train.sh                          # balance at 65 Hz
   HZ=25 tools/robot/train.sh                    # another rate
   ```

   It starts the learner (locally, or submits the Slurm job), waits for it to
   listen, writes its address into the phone's `trainer.json`, starts the app
   and waits until the phone connects (~1-2 min). An optional name comes
   first, extra `dreamerv3/main.py` flags after it:
   `tools/robot/train.sh balance_carpet --env.robot.action_rate_penalty 0.1`.
4. Stop: `tools/robot/stop.sh` (cancels the job or kills the local learner,
   and stops the app). The base stops the wheels 250 ms after commands stop.

**Resume** a run with the same rate and configs:
`RUN_DIR=<its run dir> tools/robot/train.sh resume`.

**Live demos.** `tools/robot/demo.sh` (same arguments as `train.sh`) starts
a job that keeps its GPU between audiences: **RESTART TRAINING** on the
dashboard (press twice) starts training again from random weights, with
fresh graphs, in about a second and a half. The job keeps the next session
started and parked beside the live one (`tools/robot/demo_supervisor.py`), so
a restart skips the minute-long start of a new learner; the robot holds its
wheels from the press until the new session's first weights arrive. Allow
~1.5 min after starting the job, and after each restart, before the next
restart is that fast; a restart pressed sooner keeps the current session
(wheels held) until the next one is ready, then switches. Sessions are `<run dir>/s01`, `s02`, ... and
`<run dir>/sessions.tsv` lists them. Stop with `tools/robot/stop.sh`.

**Pause** the robot (wheels held) or the training (learner idle) with the
dashboard's **PAUSE ROBOT** / **PAUSE TRAINING** buttons; each press toggles.
`PAUSED=1 tools/robot/train.sh` (or `demo.sh`) connects everything but
starts with both paused: start the robot to show the untrained behaviour,
then start training. Minute 0 of the dashboard's charts is then when training
started, and what came before reads as -1m, -2m. The state is kept in
`<logdir>/online_shared/pause.json`. Robot steps count toward `STEPS` while
training is paused.

**Settings** (environment variables): `HZ` (65; the phone's rate, passed
to the learner as `--env.robot.hz`), `CONFIGS` (default `robot`),
`STEPS` (100000: ~26 min at 65 Hz, ~67 at 25), `GPU` (Slurm:
use `ROBOT_SLURM_FLAGS_<GPU>` from robot.env), `NO_BASE=1` (phone alone, for
testing the link). Runs land in `$ROBOT_RUNS/<name>_<date>/`.

**A local learner wants a real GPU.** A laptop (CPU or Apple GPU) does
learn, but trains far fewer samples per robot step than a cluster GPU, so it
learns more slowly in wall-clock time.

## Watch

On the phone screen: the reward at the top (~1.0 per step is perfect) and the
`policy` stamp, which changes with every weight push. Or:

```bash
tools/robot/tensorboard.sh                  # the latest run
tools/robot/tensorboard.sh <run> <run> ...  # compare runs
```

It prints the address (first free port from 6006). Each run shows as
`<run>/actor` (x = robot steps: `episode/score`, `fps/policy`,
`epstats/log/dropped/sum`, `epstats/log/policy_age_s/avg`, sensors) and
`<run>/learner` (x = trained samples: losses, `fps/train`).

Healthy: `fps/policy` at the chosen rate, `dropped/sum` 0, `policy_age_s`
~2-3 s, and `logdir/online_shared/policy/latest` in the run dir advancing.
A learner can crash while the phone keeps driving with its last weights, so
that last check is the one that matters.

## Configs and reward

Config blocks in `dreamerv3/configs.yaml`, layered left to right:

| block | what |
|---|---|
| `robot` | the default, complete on its own: balancing, flat DreamerV3, continuous per-wheel actions, the fast settings from the real-time cartpole sweep ([docs/robot/SIM_PHONE.md](docs/robot/SIM_PHONE.md)) |
| `robot_command` | after `robot`: balance while following a command, see below |
| `robot_hrl` | after `robot`, with `director director_og` first: the HRL agent (OG Director), tuned for the robot ([docs/robot/SIM_ROBOT.md](docs/robot/SIM_ROBOT.md)); acts from the trainer, not the phone |
| `robot_sim` | last: the robot task against a simulated balancing body, paced to real time, no robot |
| `sim_phone`, `sim_phone_fast` | cartpole paced to real time, no robot |

The rate is never set in a block: it is `env.robot.hz` (65), which
`train.sh` sets from `HZ`. The return horizon (`env.robot.horizon_s`, 2 s) and the episode (`env.robot.episode_s`, 20 s) are in
seconds and converted to steps at that rate.

**Rewards.** `robot` pays an upright score in [0, 1] minus linear costs on
tilt rate, drift and wheel speed, and on chatter: 0.2 × the mean change of
each wheel's action per step (e1982, 2026-10-22).

Reward and motor flags, passed after the name:

| flag | default | meaning |
|---|---|---|
| `env.robot.action_rate_penalty` | 0.2 | chatter cost: flag × mean \|aₜ − aₜ₋₁\| over wheels |
| `env.robot.command_scale` | 1.0 | phone scales every wheel command: caps motor power |
| `env.robot.wheel_penalty` | 0.3 | cost on wheel speed |
| `env.robot.drift_penalty` | 0.1 | cost on net forward speed |
| `env.robot.rate_penalty` | 0.05 | cost on tilt rate |
| `env.robot.weights_every` | 0 | minimum seconds between weight pushes; `1e9` stops them |

The reward peaks at 1.0 per step, so a 20 s episode scores at most ~500 at
25 Hz, ~1000 at 50 Hz and ~2000 at 100 Hz. All reward terms are computed on
the learner (`embodied/envs/robot.py`), so changing the task needs no app
rebuild.

## Joystick and command following

The app serves a page at `http://<phone's WiFi address>:8080` (shown on the
robot's screen): **DRIVE** is a joystick, **SCREEN** mirrors the display.
Off that WiFi: `adb forward tcp:8080 tcp:8080`, then http://localhost:8080.

`http://<phone>:8080/dash` is a dashboard for showing training to an
audience: the reward curve since the run started, the age of each weight
push, the learner's losses (forwarded by the trainer every ~10 s) and a live
blueprint of the robot from the side, front and top, with the floor moving
and turning under it in each. `?theme=dark` or `?theme=light` picks the
look; the button toggles it. The tilt is drawn 30 ms ahead of the latest
reading, from its rate, to make up for the trip to the page; `?lead=<ms>`
changes that. Under `tools/robot/demo.sh` it has a **RESTART TRAINING**
button (see Train). The policy menu in its header runs any saved policy
(the Dreamer Player's list) in place of training: the phone lets the trainer
go, which waits for it, and choosing **TRAINING** reconnects on the training
weights. The learner gets no new experience meanwhile, so it idles.

Only a policy trained on the command task follows it:

```bash
CONFIGS="robot robot_command" tools/robot/train.sh cmd
```

The policy sees a command (forward, turn) in [-1, 1]. While training, the
phone holds a random one for 4-8 s (zero 25% of the time), ramped at 0.5/s;
touching the stick overrides it. Reward: exp(-(error/0.4)²) tracking of the
command as a wheel-speed vector (full stick = 2500 mm/s), −2 within 1.5° of a
bumper. Other command rewards are one flag away
(`--env.robot.command_reward`, see `robot.py:_command_reward`). In TensorBoard compare `epstats/log/cmd_forward` with
`epstats/log/forward`, likewise `turn`. The variants and their parameters are
documented beside each block in `configs.yaml`.

## Saved policies

The app opens in training. **Back** ends training and opens the policy list:
tap a policy to run it with no learner (STOP returns to the list), long-press
to rename or delete. Every training run keeps its latest weights there under
its run name, updated with every push. To save a run's current weights under
a name of your choice:

```bash
tools/robot/save_policy.sh balance_table            # the latest run
tools/robot/save_policy.sh balance_table <run dir>  # a given run
```

Copies are kept in `~/logdir/robot_policies/`.

Whenever a saved policy drives (the Dreamer Player, or the dashboard's
policy menu), the phone records every step: sensors, the policy's action,
the command and what the base was sent, to `recordings/` in its files. No
trainer is needed. Fetch them, giving the floor the robot was on:

```bash
tools/robot/pull_recordings.sh table
```

They land in `~/logdir/robot_recordings/`. Each is also added under
`sysid/<policy>_<floor>/balance.csv` for `sysid_fit.py`, and with a Slurm
trainer copied to `$ROBOT_RUNS/sysid/recordings/`. STOP loses at most the
last second.

**From the sim.** An agent trained in the MuJoCo sim (`--configs robot
[robot_command] robot_sim`, docs/robot/SIM_COMMAND.md) can go on the phone the
same way, or seed a real training run with its weights:

```bash
tools/robot/from_sim.sh try cmd_sim <sim run dir> [step]       # to the policy list
CONFIGS="robot robot_command" tools/robot/from_sim.sh train <sim run dir> [step] [name]
```

`step` picks a 5-minute snapshot (`logdir/ckpt_milestones/`, env steps,
19500 per 5 min); the default is the run's final checkpoint. `train` takes
the world model, actor, critic and optimizer state; replay starts empty.

## Without hardware

- **No base:** `NO_BASE=1 HZ=100 STEPS=30000 tools/robot/train.sh linktest`
  runs the app without wheels, paced by `HZ` alone, to test the link.
- **No phone:** start a learner and point the stand-in phone at it; it runs
  the app's own policy and link code on a fixed clock:
  ```bash
  ROBOT_RUNS=~/logdir/robot NAME=fake PORT=3000 STEPS=5000 bash tools/robot/trainer_job.sh &
  .venv/bin/python tools/robot/dev/fake_onboard_phone.py --port 3000 --hz 25 --seconds 120
  kill %1
  ```
- **No robot at all:** `sim_phone` trains cartpole swingup paced to wall-clock
  time through the same code path (needs `dm-control`, which needs Python
  ≤ 3.12): `.venv/bin/python dreamerv3/main.py --configs sim_phone --logdir ~/logdir/sim`.
- **Simulated robot:** `robot_sim` swaps the phone for a MuJoCo model of the
  robot with its own observations and rewards, paced to real time; layer it
  last: `--configs robot robot_sim` (rate: `--env.robot.hz 25`). The body
  (geometry, mass, inertia) is from tojo-soraai's
  [rlSim2Real](https://github.com/tojo-soraai/smartphone-robot-android/tree/rl-sim2real/apps/rlSim2Real/mujoco)
  model; bumpers, motors (top speed, dead zone, stiction), the one-tick delay
  and sensor noise are calibrated to our measurements
  (`embodied/envs/robot_mujoco.py`). `--env.robot.sim_model pendulum` is the
  older planar model, used for the flat DreamerV3 vs HRL results in
  [docs/robot/SIM_ROBOT.md](docs/robot/SIM_ROBOT.md). Positive tilt is a
  forward lean and forward output tips the body back, as on the robot (e1982
  replay); until 2026-10-23 both sims had that coupling backwards, which
  mirrors the task but does not make it harder, so results within the sim
  stand. `embodied/tests/test_robot_sim.py` checks signs and bumper stops.
  The motors and the floor are fitted to recordings of the robot:
  `tools/robot/dev/sysid_probe.py` drives it through scripted wheel tests
  (on a stand, then per floor) and `tools/robot/dev/sysid_fit.py` fits the
  model to them. `--env.robot.sim_surface carpet` (default) or `table` picks
  the floor; on carpet the robot learns to balance faster, and the fit shows
  why: carpet damps the body's motion where the table lets small pushes
  carry it over to the other bumper.

## Flash a base

One-time: `picotool` and docker (macOS: `brew install colima docker picotool`),
the firmware checkout, and `ROBOT_FIRMWARE_DIR` set in `robot.env`:

```bash
git clone -b telemetry-cache https://github.com/Alex-Vasilache/smartphone-robot-firmware.git
```

1. Turn the base over and push the **USB switch DOWN** (External).
2. Data cable from the computer to the base's **external USB port**.
3. `tools/robot/flash_base.sh --build` (the first time; later without
   `--build` reuses the build). It must end with `OK: base answers`. If
   nothing is detected, replug while holding **BOOTSEL** and run it again.
4. Unplug and push the switch back **UP** (Phone Side).

## Troubleshooting

| symptom | fix |
|---|---|
| `host_connected=false` | reseat the phone on the base; check the base is on and its switch is UP |
| Pico appears/disappears every second (`adb logcat \| grep UsbHostManager`) | reseat the phone; if that fails, reinstall the app |
| phone stopped powering the base after the app was killed | unplug and replug the phone |
| wheels never move, phone log says `no base` | the app is in link-test mode: `tools/robot/stop.sh`, then `train.sh` again |
| "Allow … to access Pico?" on the phone | `tools/robot/phone_allow_usb.sh`, or tap OK |
| phone shows `no trainer at …` | learner not up yet, or the phone cannot reach its address (check the phone and learner networks) |
| `cannot reach …: timed out` in logcat | the phone is dozing: `adb shell input keyevent KEYCODE_WAKEUP` |
| `adb devices` empty | Wireless debugging switched itself off: turn it on, `tools/robot/phone_connect.sh` |
| a wheel spins at power-up | old firmware: flash the base |
| learner exits after ~30 s | a crashed run left `logdir/error_learner`; `train.sh` deletes it on resume, otherwise delete it by hand |
| `RPI-RP2` drive won't mount | ignore it; `flash_base.sh` uses picotool |

## Tools

| | |
|---|---|
| `tools/robot/train.sh` | start a run and point the phone at it |
| `tools/robot/demo.sh` | a run for live demos, restartable from the dashboard |
| `tools/robot/stop.sh` | stop the learner and the app |
| `tools/robot/tensorboard.sh` | live TensorBoard for local or cluster runs |
| `tools/robot/save_policy.sh` | name and keep a run's current weights on the phone |
| `tools/robot/from_sim.sh` | a sim-trained agent on the phone, or as the start of a real run |
| `tools/robot/pull_recordings.sh` | fetch what saved policies did on the robot |
| `tools/robot/phone_connect.sh` | adb over WiFi, finding the port |
| `tools/robot/phone_allow_usb.sh` | tap OK on the USB permission dialog |
| `tools/robot/flash_base.sh` | build and flash base firmware |
| `tools/robot/trainer_job.sh` | the learner itself (Slurm job or local process) |
| `tools/robot/demo_supervisor.py` | runs the learner as restartable sessions (`demo.sh`) |
| `tools/robot/common.sh`, `robot.env.example` | shared settings |
| `tools/robot/dev/` | development: fake phones, policy export and its numerical test, protocol test, latency probes, the sim's system identification (`sysid_probe.py`, `sysid_fit.py`), learner benchmark, sweep plots, and `check_app_copies.sh` (the app carries copies of three `dreamerv3/deploy/` modules; run it after changing them) |

Code: `embodied/envs/robot.py` (env, reward, link to the phone),
`dreamerv3/deploy/` (on-phone policy in numpy, export, link protocol), robot
blocks in `dreamerv3/configs.yaml`.

## Background

Lab notes behind the design, with measurements:
[ROBOT_BRIDGE_FINDINGS](docs/robot/ROBOT_BRIDGE_FINDINGS.md) (control rate,
latency, firmware), [ONDEVICE_POLICY](docs/robot/ONDEVICE_POLICY.md) (why the
policy runs on the phone), [SIM_PHONE](docs/robot/SIM_PHONE.md) (real-time
cartpole sweep), [SAMPLE_EFFICIENCY_SURVEY](docs/robot/SAMPLE_EFFICIENCY_SURVEY.md).
They are dated records: scripts they name may since have been replaced, see
`docs/robot/history/` for the older guides.
