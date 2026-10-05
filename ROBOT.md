# Robot training quickstart

## 1. Phone

Set the Mac's IP in `~/StudioProjects/smartphone-robot-android/config.json`
(`ipconfig getifaddr en0`) -- it is baked in at build time, so it must be reset
and the app rebuilt whenever the Mac changes network, then, with the phone on adb (see below):

```bash
cd ~/StudioProjects/smartphone-robot-android
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
  ./app run --app dreamerBridge     # build + install + launch; `run`, not `install`
```

No hardware? `.venv/bin/python tools/fake_robot_phone.py --host 127.0.0.1 --port 3000`

### adb: connecting to the phone

**Over WiFi with a pairing code** (no cable). On the phone: Settings → System →
Developer options → **Wireless debugging** → *Pair device with pairing code*.
That dialog shows an IP:port and a 6-digit code; the main Wireless debugging
screen shows a **different** port.

```bash
adb pair 10.24.76.175:37535     # the PAIRING port, then paste the 6-digit code
adb connect 10.24.76.175:42761  # the port on the main Wireless debugging screen
adb devices
```

Pairing is one-time per machine; afterwards only `adb connect` is needed, but
that port changes on every reboot, so re-read it from the screen.

Do **not** run `adb tcpip` on a wireless connection — it restarts adbd and the
device goes `offline`. If you did, reconnect on 5555 instead:

```bash
adb disconnect 10.24.76.175:42761 && adb connect 10.24.76.175:5555
```

**Over USB:** plug in, accept the "Allow USB debugging" prompt, `adb devices`.
`adb tcpip 5555` then `adb connect <phone-ip>:5555` moves that session to WiFi
so the cable can come out.

## 2. Start training

```bash
tools/run_robot.sh                          # config robot_balance, learner GPU + actor CPU
CONFIGS=robot_daydreamer tools/run_robot.sh # other arm
tail -f ~/logdir/latest/train.log           # from any other terminal
```

The phone connects in; training starts once it does. Re-run with the same
`--logdir` to resume.

## 3. Stop

`Ctrl-C` in the `run_robot.sh` terminal (kills learner and actor), or
`pkill -f dreamerv3/main.py`.

## 4. Train on Saion instead (V100)

The actor stays here beside the robot at ~45Hz; only replay chunks go up and
policy weights come down. Needs the MacBook on the **OIST internal** network --
`saion-ext` through the jump host is policed to ~10 KB/s and cannot carry it
(`docs/SAION_BRIDGE_FINDINGS.md`). Put the phone on the internal WiFi too.

Start, in two steps -- `start_robot.sh` finds a running job, it does not submit
one:

```bash
# 1. the learner (8h wall limit; note the job id it prints)
ssh saion 'cd /work/DoyaU/vasilache/work/dreamerv3_robot && \
  CODE=$PWD SCRIPT=online_learner CONFIGS=robot_daydreamer \
  sbatch sbatch/run_robot_v100.sbatch'

# 2. once squeue shows it RUNNING: actor + bridge + TensorBoard here
MODE=saion tools/start_robot.sh
```

`MODE=saion` fails loudly if no learner is running; plain `tools/start_robot.sh`
silently falls back to a local learner instead, which is the right default when
the robot session matters more than the throughput.

Stop, also two steps -- **`stop_robot.sh` does not cancel the Saion job**, and a
learner left running holds a V100 for the full 8 hours:

```bash
tools/stop_robot.sh              # actor, bridge, TensorBoard here
ssh saion scancel <job-id>       # the learner
```

Watch it with `tail -f ~/logdir/latest/train.log` (actor),
`tail -f ~/logdir/.robot_session/bridge.log` (a `sent N/M` line per 8s cycle),
and http://localhost:6006.

The code the job runs is the staged copy at
`/work/DoyaU/vasilache/work/dreamerv3_robot`, not the shared checkout under
`/apps/unit/`, which does not carry the robot code at all. Sync it after
changing anything in `dreamerv3/` or `embodied/`:

```bash
rsync -a --exclude .git --exclude .venv --exclude logdir --exclude paper \
  ./ saion:/work/DoyaU/vasilache/work/dreamerv3_robot/
```

**Tunnelling the robot to the compute node instead** (`SCRIPT=train` plus
`tools/robot_tunnel.sh`) is the other shape, and it was measured at 0.45Hz with
95% of frames dropped over the jump host. Untested over the direct route; the
split above keeps the control loop off the network either way.

## 4b. Resume the 2026-09-03 balancing session

That session reached ~345 mean episode score (from 187) and was stopped
deliberately, not lost. Everything needed to continue lives on Saion:

| | |
|---|---|
| Run dir | `/work/DoyaU/vasilache/work/robot_v100_20260903_152855_xZnLLD` |
| Checkpoint | `logdir/ckpt/20260903T163052F054241` (32MB) |
| Replay | 1232 chunks in `logdir/online_shared/experience/` |
| Durable copy | `/bucket/DoyaU/vasilache/robot_checkpoints/20260903T164143_balancing` |

```bash
RUN=/work/DoyaU/vasilache/work/robot_v100_20260903_152855_xZnLLD

# The learner reloads ckpt/latest and the existing replay, so it does not
# re-prefill from nothing. train_ratio 384 rather than the block's 512: see
# docs/SAION_BRIDGE_FINDINGS.md, 512 sits exactly at what the V100 sustains.
ssh saion "cd /work/DoyaU/vasilache/work/dreamerv3_robot && \
  CODE=\$PWD SCRIPT=online_learner CONFIGS=robot_daydreamer STEPS=1000000 \
  RUN_DIR=$RUN sbatch sbatch/run_robot_v100.sbatch --run.train_ratio 384"

# once squeue shows it RUNNING
MODE=saion tools/start_robot.sh
```

**Before resuming, `$RUN/logdir/online_shared/actor_step` must not exist**
(already cleared). A fresh actor starts counting at 0, while `Ratio._prev` in
the learner is set from the first value it reads and never moves backwards -- a
leftover high `actor_step` makes the learner wait until the robot has climbed
all the way back past it, which looks exactly like the learner being hung.

Sync the code first if `dreamerv3/` or `embodied/` changed since; the job runs
the staged copy, not this working tree:

```bash
rsync -a --exclude .git --exclude .venv --exclude logdir --exclude paper \
  ./ saion:/work/DoyaU/vasilache/work/dreamerv3_robot/
```

**Health check while it runs:** `online_shared/policy/latest` must keep
advancing. Job state, GPU use and chunk flow all stay green while the learner
silently stops publishing weights, so they prove nothing on their own.

## 4bb. The learner can die silently -- watch `policy/latest`

On 2026-09-07 the learner crashed 1h32m into a session and nobody noticed for
75 minutes. The job disappeared from `squeue`, the actor kept driving the robot
at 49Hz, the bridge kept syncing, TensorBoard kept updating from the actor's own
metrics -- everything looked healthy while no gradient was being computed.

The crash was `KeyError` in `embodied/core/replay.py:_remove`. `load()` walks a
snapshot of `self.chunks` and calls `_insert()` as it goes; `_insert()` evicts
when the buffer is at capacity, so the first load that fills replay can retire a
chunk while items pointing into it are still queued. The next eviction of one of
those items then looked up a `refs` entry that was already gone. It is fixed by
discarding such an item instead of raising -- which is what `_sample()` already
does on the same condition. This only bites once replay reaches capacity, which
is why it takes an hour or two to show up.

**The only honest health check is that `online_shared/policy/latest` keeps
advancing.** Job state, GPU utilisation and chunk flow all stay green through
this. Something should be watching it for the length of every session:

```bash
watch -n60 'ssh saion "cat <RUN>/logdir/online_shared/policy/latest"'
```

**A crashed learner blocks its own restart.** On the way down it writes
`<RUN>/logdir/error_learner`, and the next job sees that file and exits after
~30s with `Shutting down due to error file`, which looks like a fresh crash
rather than a refusal. Delete it before resubmitting:

```bash
ssh saion 'rm -f <RUN>/logdir/error_learner'
```

Note the `actor_step` rule in §4b is the opposite way round here: if the actor
is **still running** when you restart the learner, leave `actor_step` alone --
it is live, not stale, and the learner should pace against it.

## 4c. On-device policy (in progress, 2026-09-07)

Work has started on running the policy on the phone and sending only weight
updates down. See `docs/ONDEVICE_POLICY.md` for the measured reaction-delay
budget that motivates it and for what is built so far.

**Heads up before the next `./app run --app dreamerBridge`:**
`apps/dreamerBridge/build.gradle.kts` now pulls in numpy via Chaquopy, which
required moving that module to Python 3.13 (Chaquopy 17 wants buildPython to
match, and 3.10 is not installed here) and cutting its ABIs to `arm64-v8a,
x86_64` (Chaquopy has no 3.13 for 32-bit). It is verified to **build**; it has
not yet been run on the robot, and the phone was left on the older APK. So the
next install is a bigger change than usual -- do it when you can watch it, not
mid-experiment.

The policy port itself is verified against the JAX modules to ~1e-6:

```bash
.venv/bin/python tools/export_policy.py --checkpoint <policy.pkl> --out policy.npz
JAX_PLATFORMS=cpu .venv/bin/python tools/test_numpy_policy.py --weights policy.npz
```

## 4d. The control rate is set by the RP2040, not by the phone (2026-09-14)

The microcontroller services one motor command per ~83 ms and reads USB only in
between, so the wheels update at ~12 Hz whatever the phone does. The phone now
paces one decision per reply (`PACE = 'serial'` in `main.py`), which applies
every action fresh instead of a random third of them, and it no longer sends
coast commands, which were stalling the microcontroller for 1 s at a time.
Numbers and the firmware fix that would lift the ceiling: `docs/ROBOT_BRIDGE_FINDINGS.md`.
On 2026-09-16 the phone side was taken to that floor: ~92 ms per decision in
both onboard and trainer-driven mode, ~15 ms sensor-to-torque onboard (see
§"Driving the phone side to the floor" in the findings). Three things to check
before every session:

```bash
adb shell dumpsys power | grep mWakefulness   # must be Awake: a dozing phone freezes the sensors
adb logcat -d | grep SerialLatency | tail -1  # max should be ~90 ms; 1090 means coast commands
adb shell dumpsys usb | grep source_power     # must be true: the phone powers the Pico
```

If the Pico is attaching and vanishing every second (`adb logcat | grep
UsbHostManager`), the phone has stopped sourcing power on the port; reinstall
the app (`./app run --app dreamerBridge`) to re-run the USB permission grant.

## 5. When the rate is below 50 Hz

`log/dropped` above zero means the phone produced observations we did not
consume. Find out whose fault it is without starting a training run:

```bash
.venv/bin/python tools/time_robot_steps.py                 # per-stage timing
.venv/bin/python tools/time_robot_steps.py --stages arrival
```

Measured 2026-09-03, the answer was the WiFi, not the code: the actor's whole
step path costs about 1 ms of the 20 ms budget. Check `ping <phone-ip>` (`adb shell ip route` gives it) -- an
average far above the ~6 ms minimum is the radio dozing.

Full numbers, and what was ruled out, in `docs/ROBOT_BRIDGE_FINDINGS.md`.

## 6. Check it

```bash
.venv/bin/python tools/did_it_learn.py ~/logdir/latest
tools/tensorboard.sh ~/logdir            # http://localhost:6006
```


Fixed. run.online_max_train_repeats: 32 caps the inner batch so weight publishing keeps a floor cadence regardless of how far behind the learner is; throughput is unchanged since the debt is still worked off, just in slices. Verifying rather than assuming caught a near-miss here: the online_* keys are passed into args explicitly, not via **config.run, so adding the config key alone would have crashed the learner with AttributeError on resume. dreamerv3/main.py:157 now wires it through, and both it and the --run.train_ratio 384 override are verified to resolve. Synced to Saion's staging copy.

Saved. Checkpoint 20260903T163052F054241 and all 1232 replay chunks are intact in the run dir, with the durable copy in /bucket/DoyaU/vasilache/robot_checkpoints/20260903T164143_balancing. I also cleared online_shared/actor_step, because a fresh actor counts from 0 while the learner's Ratio._prev never moves backwards — leaving it would have stalled the resume in a way that looks exactly like a hang.

Documented. ROBOT.md §4b has the resume procedure with the concrete paths and the one-command relaunch; docs/SAION_BRIDGE_FINDINGS.md has the train-ratio debt mechanism, including that online_shared/policy/latest is the health check that matters — job state, GPU use and chunk flow all stay green while the learner silently stops publishing.