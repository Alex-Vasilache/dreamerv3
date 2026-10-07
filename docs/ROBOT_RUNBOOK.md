# Robot runbook: flash a base, start a training

The phone runs the policy at 25 Hz; Saion trains it and pushes new weights to
the phone every ~3 s (`online_publish_every: 2` plus up to one train step).
Both ends move the socket I/O off the control path: the trainer keeps every
observation the phone sends (`log/dropped` 0, `log/backlog` ~0) and pushes
weights in 32 KB slices from a thread; the phone parses them on a receiver
thread and swaps them in between two steps. `log/policy_age_s` is how old the
phone's weights were at each step. The base (RP2040) runs the fast firmware, which answers a
wheel command in ~8 ms.

| repo (on the Mac) | branch | what |
|---|---|---|
| `~/work/research/dreamerv3` | `smartphone-robot-bridge` | trainer, these tools |
| `~/StudioProjects/smartphone-robot-android` | `dreamer-bridge` | phone app `dreamerBridge` |
| `~/StudioProjects/smartphone-robot-firmware` | `telemetry-cache` (fork `Alex-Vasilache/…`) | base firmware |

## 0. One-time setup (Mac)

```bash
brew install colima docker picotool   # build image runs in colima; picotool flashes
```

## 1. Flash a base

1. Turn the base over. Push the **External/Phone USB switch DOWN** (External).
2. Plug a data cable from the Mac into the base's **external USB port** (not the
   phone connector).
3. Run (from `~/work/research/dreamerv3`):
   ```bash
   tools/flash_base.sh            # add --build after changing the firmware
   ```
   It must end with `OK: base answers`. A running base is put into its
   bootloader automatically. If nothing is detected at all, unplug, hold
   **BOOTSEL** while plugging back in, and run it again.
4. Unplug, push the switch back **UP** (Phone Side).

All three bases run `1fd6c67`; check with `git -C ~/StudioProjects/smartphone-robot-firmware log --oneline -1`.

## 2. Connect the phone (each session)

1. Phone: Settings → Developer options → **Wireless debugging** → note IP:port.
   ```bash
   adb connect 10.13.64.55:<port>
   ```
2. Put the phone on the base. Check it powers the base:
   ```bash
   adb shell dumpsys usb | grep -E 'host_connected|source_power'   # both true
   ```
3. Only after changing the app:
   ```bash
   cd ~/StudioProjects/smartphone-robot-android
   JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
     ./gradlew :dreamerBridge:assembleDebug
   adb install -r apps/dreamerBridge/build/outputs/apk/debug/dreamerBridge-debug.apk
   ```

## 3. Start a training

1. Pick the next experiment number: highest `e<N>` in `EXPERIMENTS.md` on Saion
   (`/apps/unit/DoyaU/vasilache/apps/code/dreamerv3`), plus one.
2. Robot upright on the floor with room to move, then:
   ```bash
   tools/robot_train.sh e1224_robot_25hz
   ```
   It submits the job, waits for a node, writes the node's address to the
   phone (`trainer.json`), starts the app, taps the USB prompt, and waits until
   the phone is connected (~1-2 min).
3. Watch: the phone screen (reward on top, ~1.0 per step is perfect), and
   `ssh saion tail -f <run dir>/logdir/scores.jsonl` (max ~500 per 20 s
   episode). A healthy learner keeps changing the `policy` number on the phone.
4. Stop: `ssh saion scancel <job>` and
   `adb shell am force-stop jp.oist.abcvlib.dreamerBridge`. The base stops the
   wheels by itself 250 ms after the commands stop.
5. Log the run in `EXPERIMENTS.md` on Saion.

To resume an earlier run (same configs, same rate), pass its run dir:
`RUN_DIR=/work/DoyaU/vasilache/work/robot_v100_<…> tools/robot_train.sh e<N>_robot_25hz_resume`.

Settings: `CONFIGS` (default `robot_daydreamer robot_fast robot_25hz`),
`STEPS` (default 100000 ≈ 67 min at 25 Hz), `HZ` (default 25). For 50 Hz:
`HZ=50 tools/robot_train.sh e<N>_robot_50hz --agent.horizon 100 --env.robot.length 1000`.

## Test the link without a base

Put `"no_base": true` in `trainer.json`: the app then starts without USB, skips
the serial link and the wheels, and paces on `max_hz` alone. A local trainer:

```bash
echo '{"ip": "<Mac IP>", "port": 3000, "max_hz": 50, "no_base": true}' > /tmp/trainer.json
adb push /tmp/trainer.json /sdcard/Android/data/jp.oist.abcvlib.dreamerBridge/files/
.venv/bin/python -u dreamerv3/main.py --logdir ~/logdir/link50 --script train \
  --configs robot_daydreamer robot_fast robot_50hz --env.robot.onboard True
```

Without a phone at all, `tools/fake_onboard_phone.py --hz 50` plays it with the
app's own PolicyRunner and receiver.

## Troubleshooting

| symptom | fix |
|---|---|
| "Allow … to access Pico?" on the phone | `tools/phone_allow_usb.sh` (or tap OK) |
| "Robot not properly attached" | base not on the phone, or `source_power=false`: replug the base |
| Pico appears/disappears every second (`adb logcat \| grep UsbHostManager`) | reinstall the app (step 2.3) |
| phone shows `no trainer at …` | job not started yet, or wrong `trainer.json`; rerun `tools/robot_train.sh` |
| a wheel spins at power-up | base has old firmware: flash it (step 1) |
| `RPI-RP2` drive won't mount on the Mac | ignore it, `tools/flash_base.sh` uses picotool instead |
| a job exits after ~30 s | stale `logdir/error_learner` in the run dir: delete it |
