# Robot runbook

The phone runs the policy at 25, 50 or 100 Hz. A Saion GPU job trains on what
the phone records and pushes it new weights every ~2 s.

| repo (on the Mac) | branch | what |
|---|---|---|
| `~/work/research/dreamerv3` | `smartphone-robot-bridge` | trainer, these tools |
| `~/StudioProjects/smartphone-robot-android` | `dreamer-bridge` | phone app `dreamerBridge` |
| `~/StudioProjects/smartphone-robot-firmware` | `telemetry-cache` | base firmware |

Run every command from `~/work/research/dreamerv3`; no venv needs activating.
Code blocks hold commands only, so they paste cleanly into zsh.

## Start a training

1. On the phone: Settings → Developer options → **Wireless debugging** on.
   Then connect (the port changes each time; this finds it):
   ```bash
   tools/phone_adb_connect.sh
   ```
2. Put the phone on the base, switch the base on, and check that both lines
   say `true`:
   ```bash
   adb shell dumpsys usb | grep -E 'host_connected|source_power'
   ```
3. Stand the robot upright with room to move, and start one of:

   | rate | command |
   |---|---|
   | 25 Hz | `tools/robot_train.sh` |
   | 50 Hz | `HZ=50 CONFIGS="robot_daydreamer robot_fast robot_50hz" tools/robot_train.sh` |
   | 100 Hz | `HZ=100 CONFIGS="robot_daydreamer robot_fast robot_50hz" tools/robot_train.sh --agent.horizon 200 --env.robot.length 2000` |

   It picks the next experiment number (prints `== submitting e<N>_robot_50hz`),
   submits the job, points the phone at the node and starts the app; the phone
   connects after ~1-2 min. A name argument is appended to the number
   (`tools/robot_train.sh robot_50hz_newbase`); one starting with `e<N>_` is
   used as given.
4. Watch: the phone screen (reward on top, ~1.0 per step is perfect; the
   `policy` number changes every ~2 s), or TensorBoard (below).
5. Stop:
   ```bash
   ssh saion scancel <job>
   adb shell am force-stop jp.oist.abcvlib.dreamerBridge
   ```
   The base stops the wheels by itself 250 ms after commands stop.
6. Log the run in `EXPERIMENTS.md` on Saion.

**Resume** a stopped run with the same rate and configs:
`RUN_DIR=/work/DoyaU/vasilache/work/robot_v100_<…> tools/robot_train.sh robot_25hz_resume`.

**Script settings**: `HZ` (default 25), `CONFIGS` (default
`robot_daydreamer robot_fast robot_25hz`), `STEPS` (default 100000: ~67 min at
25 Hz, ~33 at 50, ~17 at 100). Anything after the name goes to `main.py`.

## TensorBoard

```bash
tools/saion_tensorboard.sh
```

Follows the latest robot job; pass run dirs to compare runs instead. It
prints the address (usually http://localhost:6007; Cursor holds 6006) and
refreshes every 20 s. Ctrl-C stops it. Each run shows twice:

- `<run>/actor`, x = robot steps: `episode/score`, `fps/policy`,
  `epstats/log/dropped/sum`, `epstats/log/policy_age_s/avg`, sensors.
- `<run>/learner`, x = trained samples: losses, `fps/train`.

A healthy run has `fps/policy` at the chosen rate, `dropped/sum` 0, and
`policy_age_s/avg` ~2-3 s. One-time install:
`.venv/bin/pip install tensorboard torch`.

## Reward and motor flags

Pass as extra arguments, e.g.
`tools/robot_train.sh --env.robot.action_rate_penalty 0.1`.

| flag | 25 Hz | 50/100 Hz | meaning |
|---|---|---|---|
| `env.robot.action_rate_penalty` | 0 | 0.1 | chatter cost per step: flag × mean over wheels of \|aₜ − aₜ₋₁\| (commands in [−1, 1]; at most 2 × flag) |
| `env.robot.command_scale` | 1.0 | 0.7 | phone multiplies every wheel command by this: caps motor power |
| `env.robot.wheel_penalty` | 0.3 | 0.3 | cost on wheel speed (at most 0.3 per step) |
| `env.robot.drift_penalty` | 0.1 | 0.1 | cost on net forward speed |
| `env.robot.rate_penalty` | 0.05 | 0.05 | cost on tilt rate |
| `env.robot.weights_every` | 0 | 0 | minimum seconds between weight pushes; `1e9` stops them |

The reward peaks at 1.0 per step, so a 20 s episode scores at most ~500 at
25 Hz, ~1000 at 50 Hz and ~2000 at 100 Hz.

## Update the phone app

After changing the app:
```bash
cd ~/StudioProjects/smartphone-robot-android
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :dreamerBridge:assembleDebug
adb install -r apps/dreamerBridge/build/outputs/apk/debug/dreamerBridge-debug.apk
```

## Flash a base

One-time on the Mac: `brew install colima docker picotool`.

1. Turn the base over and push the **USB switch DOWN** (External).
2. Connect a data cable from the Mac to the base's **external USB port**.
3. Flash (add `--build` after changing the firmware); it must end with
   `OK: base answers`:
   ```bash
   tools/flash_base.sh
   ```
   If nothing is detected, replug while holding **BOOTSEL** and run it again.
4. Unplug and push the switch back **UP** (Phone Side).

The bases should run `ae3b2ba` or later:
`git -C ~/StudioProjects/smartphone-robot-firmware log --oneline -1`.

## Test the link without a base

`NO_BASE=1` starts the app without the base (no wheels, paced by `HZ` alone).
Use a non-experiment name:
```bash
NO_BASE=1 HZ=100 STEPS=30000 CONFIGS="robot_daydreamer robot_fast robot_50hz" tools/robot_train.sh linktest_100hz --agent.horizon 200 --env.robot.length 2000
```
Without a phone, `tools/fake_onboard_phone.py --hz 50` stands in for it against
a trainer on the Mac.

Measured 2026-10-07 (phone alone, ~4 min each). Every rate held with 0 dropped
observations and a push every ~2-3 s. The phone's sample→drive time:

| rate | p50 / p99 / max |
|---|---|
| 25 Hz | 6.1 / 11.4 / 17.8 ms |
| 50 Hz | 4.0 / 5.7 / 15.0 ms |
| 100 Hz | 2.6 / 4.2 / 8.5 ms |
| 100 Hz, no pushes | 2.5 / 3.1 / 4.0 ms |

On Saion a push takes ~0.5 s on the WiFi and the weights are ~0.6 s old on
arrival.

## Troubleshooting

| symptom | fix |
|---|---|
| `host_connected=false` | reseat the phone on the base; check the base is on and its switch is UP |
| Pico appears/disappears every second (`adb logcat \| grep UsbHostManager`) | reseat the phone; if that fails, reinstall the app |
| wheels never move, phone log says `no base` | the app was started in link-test mode: `adb shell am force-stop jp.oist.abcvlib.dreamerBridge`, rerun `tools/robot_train.sh` |
| "Allow … to access Pico?" on the phone | `tools/phone_allow_usb.sh`, or tap OK |
| phone shows `no trainer at …` | job not running yet, or stale `trainer.json`: rerun `tools/robot_train.sh` |
| `cannot reach …: timed out` in logcat | the phone is dozing: `adb shell input keyevent KEYCODE_WAKEUP` |
| `adb devices` empty | Wireless debugging switched itself off: turn it on, `tools/phone_adb_connect.sh` |
| a wheel spins at power-up | old firmware: flash the base |
| `RPI-RP2` drive won't mount | ignore it; `tools/flash_base.sh` uses picotool |
| job exits after ~30 s | delete the stale `logdir/error_learner` in the run dir |
