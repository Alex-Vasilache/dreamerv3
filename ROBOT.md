# DreamerV3 on the OIST smartphone robot

This branch (`smartphone-robot-bridge`) trains DreamerV3 on the real
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
| [Alex-Vasilache/dreamerv3](https://github.com/Alex-Vasilache/dreamerv3) | `smartphone-robot-bridge` | learner, robot env, these tools |
| [Alex-Vasilache/smartphone-robot-android](https://github.com/Alex-Vasilache/smartphone-robot-android) | `dreamer-bridge` | phone app `dreamerBridge` (fork of [oist/smartphone-robot-android](https://github.com/oist/smartphone-robot-android)) |
| [Alex-Vasilache/smartphone-robot-firmware](https://github.com/Alex-Vasilache/smartphone-robot-firmware) | `telemetry-cache` | base firmware: motor watchdog and I2C recovery on top of upstream's fast-reply telemetry cache (fork of [oist/smartphone-robot-firmware](https://github.com/oist/smartphone-robot-firmware)) |

All commands below run from the dreamerv3 checkout.

## Install

### 1. Learner

```bash
git clone -b smartphone-robot-bridge https://github.com/Alex-Vasilache/dreamerv3.git
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

**On a Slurm cluster** (optional): clone the same branch there, make a Python
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

   | rate | command |
   |---|---|
   | 25 Hz | `tools/robot/train.sh` |
   | 50 Hz | `HZ=50 tools/robot/train.sh` |
   | 100 Hz | `HZ=100 tools/robot/train.sh` |

   It starts the learner (locally, or submits the Slurm job), waits for it to
   listen, writes its address into the phone's `trainer.json`, starts the app
   and waits until the phone connects (~1-2 min). An optional name comes
   first, extra `dreamerv3/main.py` flags after it:
   `tools/robot/train.sh balance_carpet --env.robot.action_rate_penalty 0.1`.
4. Stop: `tools/robot/stop.sh` (cancels the job or kills the local learner,
   and stops the app). The base stops the wheels 250 ms after commands stop.

**Resume** a run with the same rate and configs:
`RUN_DIR=<its run dir> tools/robot/train.sh resume`.

**Settings** (environment variables): `HZ` (25; picks the `robot_<HZ>hz`
preset), `CONFIGS` (default `robot_daydreamer robot_fast robot_<HZ>hz`),
`STEPS` (100000: ~67 min at 25 Hz, ~33 at 50, ~17 at 100), `GPU` (Slurm:
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
| `robot` | base: flat DreamerV3 (no HRL), size1m, online learning |
| `robot_daydreamer` | balancing task, continuous per-wheel actions |
| `robot_fast` | faster-learning hyperparameters from the real-time cartpole sweep ([docs/robot/SIM_PHONE.md](docs/robot/SIM_PHONE.md)) |
| `robot_25hz`, `_50hz`, `_65hz`, `_100hz` | 2 s horizon and 20 s episodes at that rate |
| `robot_command` (+ `_cur`, `_sum`, `_taper_sum`, `_lean`) | balance while following a command, see below |
| `sim_phone`, `sim_phone_fast` | cartpole paced to real time, no robot |
| `hrl_robot`, `hrl_robot_fast` | after the robot blocks, with `director director_og` first: the HRL agent ('now' Director), tuned for the robot ([docs/robot/SIM_ROBOT.md](docs/robot/SIM_ROBOT.md)); acts from the trainer, not the phone |
| `sim_robot` + `sim_25hz` / `sim_65hz` | the robot task against a simulated balancing body, paced to real time, no robot |

Reward and motor flags, passed after the name:

| flag | default | meaning |
|---|---|---|
| `env.robot.action_rate_penalty` | 0 | chatter cost: flag × mean \|aₜ − aₜ₋₁\| over wheels |
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

Only a policy trained on the command task follows it:

```bash
CONFIGS="robot_daydreamer robot_fast robot_25hz robot_command" \
  tools/robot/train.sh cmd_25hz --env.robot.action_rate_penalty 0.05
```

The policy sees a command (forward, turn) in [-1, 1]. While training, the
phone holds a random one for 2-5 s (zero 30% of the time); touching the stick
overrides it. Reward: balance × exp(-(error/0.5)²) tracking of forward speed
and turn rate. In TensorBoard compare `epstats/log/cmd_forward` with
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
- **Simulated robot:** `sim_robot` swaps the phone for a two-wheeled
  inverted-pendulum model with the robot's own observations and rewards
  (pure numpy, no `dm-control`); layer it last with `sim_25hz` or `sim_65hz`:
  `--configs robot_daydreamer robot_fast robot_65hz sim_robot sim_65hz`.
  Results for flat DreamerV3 vs HRL at 25/65 Hz: [docs/robot/SIM_ROBOT.md](docs/robot/SIM_ROBOT.md).

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
| `tools/robot/stop.sh` | stop the learner and the app |
| `tools/robot/tensorboard.sh` | live TensorBoard for local or cluster runs |
| `tools/robot/save_policy.sh` | name and keep a run's current weights on the phone |
| `tools/robot/phone_connect.sh` | adb over WiFi, finding the port |
| `tools/robot/phone_allow_usb.sh` | tap OK on the USB permission dialog |
| `tools/robot/flash_base.sh` | build and flash base firmware |
| `tools/robot/trainer_job.sh` | the learner itself (Slurm job or local process) |
| `tools/robot/common.sh`, `robot.env.example` | shared settings |
| `tools/robot/dev/` | development: fake phones, policy export and its numerical test, protocol test, latency probes, learner benchmark, sweep plots, and `check_app_copies.sh` (the app carries copies of three `dreamerv3/deploy/` modules; run it after changing them) |

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
