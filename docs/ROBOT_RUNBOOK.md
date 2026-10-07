# Robot runbook: flash a base, start a training

The phone runs the policy itself (onboard mode) at 25, 50 or 100 Hz; Saion
trains on what it records and pushes new weights to the phone every ~2 s. The
base (RP2040) runs the fast firmware, which answers a wheel command in ~8 ms.

How the link stays out of the control loop (2026-10-07): the trainer reads the
socket on its own thread and keeps every observation (`log/dropped` is 0
onboard; `log/backlog` is how many are still queued), and sends weights in
32 KB slices from another thread. The phone receives and parses them on a
thread too, and swaps them in between two steps. `log/policy_age_s` is how old
the phone's weights were at each step: it sawtooths between ~0.6 and ~2.6 s.

| repo (on the Mac) | branch | what |
|---|---|---|
| `~/work/research/dreamerv3` | `smartphone-robot-bridge` | trainer, these tools |
| `~/StudioProjects/smartphone-robot-android` | `dreamer-bridge` | phone app `dreamerBridge` |
| `~/StudioProjects/smartphone-robot-firmware` | `telemetry-cache` (fork `Alex-Vasilache/…`) | base firmware |

## 0. One-time setup (Mac)

Run everything from `~/work/research/dreamerv3`. No venv needs activating:
the tools are bash plus the system `python3`, and the training runs on Saion.
The repo's `.venv` is used only by `tools/saion_tensorboard.sh` (which calls it
directly) and by a trainer on the Mac (section 4). In zsh, paste commands without their trailing `# comments`: zsh
passes them on as arguments unless `setopt interactivecomments` is set.

```bash
# build image runs in colima; picotool flashes
brew install colima docker picotool
```

## 1. Flash a base

1. Turn the base over. Push the **External/Phone USB switch DOWN** (External).
2. Plug a data cable from the Mac into the base's **external USB port** (not the
   phone connector).
3. Run (from `~/work/research/dreamerv3`):
   ```bash
   # add --build after changing the firmware
   tools/flash_base.sh
   ```
   It must end with `OK: base answers`. A running base is put into its
   bootloader automatically. If nothing is detected at all, unplug, hold
   **BOOTSEL** while plugging back in, and run it again.
4. Unplug, push the switch back **UP** (Phone Side).

All three bases run `1fd6c67`; check with `git -C ~/StudioProjects/smartphone-robot-firmware log --oneline -1`.

## 2. Connect the phone (each session)

1. Phone: Settings → Developer options → **Wireless debugging** on. Then
   ```bash
   # finds the port (it changes) and connects
   tools/phone_adb_connect.sh
   ```
2. Put the phone on the base. Check it powers the base:
   ```bash
   # both true
   adb shell dumpsys usb | grep -E 'host_connected|source_power'
   ```
3. Only after changing the app:
   ```bash
   cd ~/StudioProjects/smartphone-robot-android
   JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
     ./gradlew :dreamerBridge:assembleDebug
   adb install -r apps/dreamerBridge/build/outputs/apk/debug/dreamerBridge-debug.apk
   ```
   Then `adb shell run-as jp.oist.abcvlib.dreamerBridge ls files/dreamer_policy`
   should list `policy.npz`: a phone without weights starts on the trainer's
   policy and switches to its own at the first push.

## 3. Start a training

1. Robot upright on the floor with room to move, then one of. The experiment
   number is picked for you: the highest `e<N>` on Saion (`EXPERIMENTS*.md`,
   slurm log names, the last month's job names) plus one, printed as
   `== submitting e<N>_…`. A name that already starts with `e<N>_` is kept.
   ```bash
   # 25 Hz -> e<N>_robot_25hz: 2 s horizon = 50 steps, 20 s episodes
   tools/robot_train.sh

   # 50 Hz -> e<N>_robot_50hz: wheel-chatter penalty, motors capped at 70%
   HZ=50 CONFIGS="robot_daydreamer robot_fast robot_50hz" tools/robot_train.sh

   # 100 Hz -> e<N>_robot_100hz: robot_50hz with horizon and episode doubled
   HZ=100 CONFIGS="robot_daydreamer robot_fast robot_50hz" tools/robot_train.sh \
     --agent.horizon 200 --env.robot.length 2000
   ```
   It wakes the phone (a dozing phone has no network for the app), submits the
   job, waits for a node, writes the node's address to the phone
   (`trainer.json`), starts the app, taps the USB prompt, and waits until the
   phone is connected (~1-2 min). Add a word to tell runs apart:
   `tools/robot_train.sh robot_50hz_newbase` gives `e<N>_robot_50hz_newbase`.
2. Watch: the phone screen (reward on top, ~1.0 per step is perfect; the
   `policy` number changes every ~2 s), and on Saion
   `tail -f <run dir>/logdir/scores.jsonl` and the job's `.out` file, which logs
   every push (`pushed policy … send 0.5s, age 0.6s`).
   For plots, see *TensorBoard* below.
3. Check the link in `<run dir>/logdir/metrics.jsonl`: `fps/policy` at the
   rate you chose, `epstats/log/dropped/sum` 0, `epstats/log/policy_age_s/avg`
   ~2-3 s.
4. Stop: `ssh saion scancel <job>` and
   `adb shell am force-stop jp.oist.abcvlib.dreamerBridge`. The base stops the
   wheels by itself 250 ms after the commands stop.
5. Log the run in `EXPERIMENTS.md` on Saion.

To resume an earlier run (same configs, same rate), pass its run dir:
`RUN_DIR=/work/DoyaU/vasilache/work/robot_v100_<…> tools/robot_train.sh robot_25hz_resume`.

Settings: `CONFIGS` (default `robot_daydreamer robot_fast robot_25hz`),
`STEPS` (default 100000: ~67 min at 25 Hz, ~33 at 50, ~17 at 100), `HZ`
(default 25). Weight pushes: `online_publish_every` (2 s in the robot configs)
sets the cadence; `--env.robot.weights_every <s>` adds a floor between pushes,
and `1e9` turns them off after the first.

### TensorBoard

Saion has no tensorboard; the runs there write only `metrics.jsonl` and
`scores.jsonl`. `tools/saion_tensorboard.sh` copies those to the Mac every
20 s, converts them to event files under `~/logdir/saion_tb/`, and serves them:

```bash
tools/saion_tensorboard.sh
```

That follows the latest robot job. To compare runs, pass their run dirs:

```bash
tools/saion_tensorboard.sh /work/DoyaU/vasilache/work/robot_v100_<a> /work/DoyaU/vasilache/work/robot_v100_<b>
```

It prints the address: the first free port from 6006 up, usually
http://localhost:6007 because Cursor holds 6006. Each run shows as two:
`<run>/actor` (x = robot steps: `episode/score`, `fps/policy`,
`epstats/log/dropped/sum`, `epstats/log/policy_age_s/avg`, the sensors) and
`<run>/learner` (x = trained samples: losses, `fps/train`). Ctrl-C stops it.
Needs the repo's `.venv` with tensorboard and torch, once:
`.venv/bin/pip install tensorboard torch`.

### Reward and motor settings

Reward and motor settings, passed as flags to `tools/robot_train.sh` (e.g.
`tools/robot_train.sh --env.robot.action_rate_penalty 0.1`):

| flag | 25 Hz | 50/100 Hz | what it does |
|---|---|---|---|
| `env.robot.action_rate_penalty` | 0 | 0.1 | chatter cost: this × mean over both wheels of \|aₜ − aₜ₋₁\|, the command in the policy's [−1, 1] units (the action the phone applied; aₜ₋₁ = 0 at episode start). At most 2× per step. A cost, not a filter, so the loop gets no delay |
| `env.robot.command_scale` | 1.0 | 0.7 | the phone multiplies every wheel command by this, capping motor power while the policy keeps its full range |
| `env.robot.wheel_penalty` | 0.3 | 0.3 | effort cost: this × mean of \|wheel speed\| (counts × 1e-4, clipped at 1 per wheel); at most 0.3 per step |
| `env.robot.drift_penalty` | 0.1 | 0.1 | cost on net forward speed, same units and clip |
| `env.robot.rate_penalty` | 0.05 | 0.05 | cost on \|tilt rate\| |

The reward is at most 1.0 per step (tilt near upright), so a 20 s episode
scores at most ~500 at 25 Hz, ~1000 at 50 Hz and ~2000 at 100 Hz.

## 4. Test the link without a base

`NO_BASE=1` puts `"no_base": true` in `trainer.json`: the app starts without
USB, skips the serial link and the wheels, and paces on `max_hz` alone. Use a
non-experiment name, it is a smoke test:

```bash
NO_BASE=1 HZ=100 STEPS=30000 CONFIGS="robot_daydreamer robot_fast robot_50hz" \
  tools/robot_train.sh linktest_100hz --agent.horizon 200 --env.robot.length 2000
```

A trainer on the Mac works the same way: push `{"ip": "<Mac IP>", "port": 3000,
"max_hz": 50, "no_base": true}` as `trainer.json` and run
`.venv/bin/python -u dreamerv3/main.py --logdir ~/logdir/link50 --script train
--configs robot_daydreamer robot_fast robot_50hz --env.robot.onboard True`.
Without a phone at all, `tools/fake_onboard_phone.py --hz 50` plays it with the
app's own PolicyRunner and receiver.

Measured 2026-10-07, phone alone, trainer on the Mac (~4 min each):

| rate | recorded | dropped | pushes | phone sample→drive p50 / p99 / max |
|---|---|---|---|---|
| 25 Hz | 25.0 | 0 | 89 | 6.1 / 11.4 / 17.8 ms |
| 50 Hz | 50.0 | 0 | 88 | 4.0 / 5.7 / 15.0 ms |
| 100 Hz | 99.7 | 0 | 93 | 2.6 / 4.2 / 8.5 ms |
| 100 Hz, no pushes | 100.0 | 0 | 1 | 2.5 / 3.1 / 4.0 ms |

On Saion (gpu19, 50 Hz, same configs): 50.0 recorded, 0 dropped, a push every
~2 s taking 0.02 s to pack and ~0.5 s on the WiFi, phone sample→drive p99
5.6 ms. A push costs the phone ~1 ms at p99. Pinning the receiver thread to the little
cores made it worse (93 Hz, p99 6.6 ms), so it is not pinned.

## Troubleshooting

| symptom | fix |
|---|---|
| "Allow … to access Pico?" on the phone | `tools/phone_allow_usb.sh` (or tap OK) |
| "Robot not properly attached" | base not on the phone, or `source_power=false`: replug the base |
| Pico appears/disappears every second (`adb logcat \| grep UsbHostManager`) | reinstall the app (step 2.3) |
| phone shows `no trainer at …` | job not started yet, or wrong `trainer.json`; rerun `tools/robot_train.sh` |
| `cannot reach …: timed out` in logcat although the trainer listens | the phone is dozing (`adb shell dumpsys power \| grep mWakefulness`): `adb shell input keyevent KEYCODE_WAKEUP` |
| `adb devices` empty, phone pings | Wireless debugging switched itself off: turn it on, `tools/phone_adb_connect.sh` |
| a wheel spins at power-up | base has old firmware: flash it (step 1) |
| `RPI-RP2` drive won't mount on the Mac | ignore it, `tools/flash_base.sh` uses picotool instead |
| a job exits after ~30 s | stale `logdir/error_learner` in the run dir: delete it |
