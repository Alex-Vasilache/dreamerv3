# Smartphone robot bridge: measured findings

Everything here was measured on 2026-09-03, on the real rig (Pixel 3a on the
OIST WiFi, MacBook M5) unless it says otherwise. Numbers are from
`tools/time_robot_steps.py`, `tools/bench_learner.py` and the per-step
`$logdir/env0/timing.csv` that `embodied/envs/robot.py` writes.

## 1. The control loop was capped by the protocol, not by anything physical

The bridge was lock-step: the phone blocked waiting for an action before every
tick. A step therefore cost one control period **plus** the round trip plus
whatever the policy took, and the loop ran at 9.7Hz against a phone that ticks
at 50Hz.

Pipelining it -- the phone free-runs on its own clock, applies the newest action
it has received, and reports every tick without waiting -- moved the round trip
inside the control period.

| | Lock-step | Pipelined |
|---|---|---|
| Agent decisions | 4.8 Hz (repeat 2) | **~45 Hz** (repeat 1) |
| Round trip p50 / p90 | 63.7 / 223 ms | **19.0** / 46 ms |
| Phone blocked on the trainer | p50 35.8 ms, max 2370 ms | **p50 0.3 ms**, p99 5.4 ms |

The cost is a fixed one-tick action delay: an observation sampled at 18.4ms is
decided on at ~27ms and reaches the wheels at the next tick, 40ms. It is
consistent, so the world model can learn it. Protocol version 2; `pipeline`
False in the config restores lock-step without reinstalling the phone app.

## 2. Nothing in the actor's step path costs anything

Measured per step against a 20ms budget, with no JAX and no learner:

| | p50 |
|---|---|
| socket send | 0.06 ms |
| drain queued frames | 0.03 ms |
| reward | 0.02 ms |
| observation assembly | 0.06 ms |
| `replay.add` | 0.10 ms |
| **whole step path** | **0.5 - 1.1 ms** |

Adding the Driver, and then moving the env into its own subprocess, changed
nothing measurable. There is no optimisation to find here.

## 3. The remaining gap is the WiFi link

`--stages arrival` separates the two ends by comparing the phone's own
timestamp on each frame with when it reached us:

| | p50 | p90 | p99 |
|---|---|---|---|
| Phone **made** the frames | 20.00 ms | 21.6 | 40.8 |
| We **received** them | 20.1 ms | 45.9 | 176 |

Evenly produced, clumpily delivered: 1.47 frames per wake. ICMP to the phone
ran 6ms at best against a **129ms average**, which is the radio dozing between
beacons.

A `WifiLock(WIFI_MODE_FULL_HIGH_PERF)` held for the length of a run (in
`MainActivity.kt`) took the ping average to 19ms and the clumping to 1.29
frames per wake. What is left is the medium: the reverse direction is just as
ragged (phone to Mac, 5.6ms best against 38ms average). About 10-30% of frames
are still dropped depending on how busy the air is; `log/dropped` reports it
per run. Dropping costs less than it sounds -- the trainer always acts on the
**newest** reading, so a drop lengthens the control period rather than acting
on stale data.

A cable (`adb reverse` over USB) would fix it outright but the rig is mobile.
A quieter or dedicated access point is the only remaining lever.

## 4. Actor on CPU or on the GPU makes no difference

| Condition | CPU | Metal |
|---|---|---|
| Learner idled, back to back | 33.2 Hz | 33.3 Hz |
| Learner training, back to back | 44.7 Hz | 44.2 Hz |
| All 10 cores pegged | 41.9 Hz, 14.5% dropped | 34.4 Hz, 30.7% dropped |

Follows from (2): the policy is a fraction of a millisecond, so the device it
runs on cannot matter. **Compare only back to back** -- WiFi drift moved the
identical configuration between 34.9 and 44.7 Hz within one morning, which is
larger than any device effect, and the loaded row above is unpaired and so
should not be read as a difference. `ACTOR_PLATFORM=cpu` stays the default: it
leaves the GPU entirely to the learner and costs nothing. Pegging all ten cores
costs the actor only 44.7 -> 41.9 Hz, so ordinary MacBook use does not disturb
the robot.

## 5. Bigger batches do not help on the M5

`tools/bench_learner.py`, learner alone, deter 256 / units 256, batch_length 32:

| batch | samples/s | gradient steps/s |
|---|---|---|
| 16 | 1462 | 2.86 |
| 64 | 1521 | 0.74 |
| 256 | 1645 | 0.20 |

16x the batch for 12% more throughput: this learner is FLOP-bound, so batch
size trades step count against step size at par. Raising it does **not** buy
train ratio here. Whether a V100 behaves differently (it has far more parallel
headroom and may well be launch-bound at this size) has to be measured there,
not assumed.

## 6. Which leaves the train ratio as the open problem

At ~45Hz with repeat 1, `train_ratio: 512` asks for ~23,000 samples/s. The M5
delivers ~1,600, so the achieved ratio is nearer **35**. Nothing local closes
that gap -- not width, not batch size, not the actor's device. It is the reason
to move the learner to a V100 (`sbatch/run_robot_v100.sbatch`), and the first
thing to measure there is whether batch scaling behaves differently than it
does here.

Earlier network-size work, for reference (learner throttled to the 512 target,
so these are "does this shape keep up", not ceilings):

| shape | deter | params | achieved ratio |
|---|---|---|---|
| 3x64 | 256 | 0.36M | 511 |
| 3x256 | 512 | 2.64M | 489 |
| 3x512 | 512 | 6.45M | 391 |

## Tools

| | |
|---|---|
| `tools/time_robot_steps.py` | Per-stage step timing and frame arrival, no JAX. Seconds to start. |
| `tools/bench_learner.py` | Learner throughput by shape, in-process synthetic batches. `--smoke` validates it on CPU in 11s. |
| `tools/run_robot.sh` | `ACTOR_PLATFORM=cpu\|metal`, `SPLIT=0\|1`. |
| `$logdir/env0/timing.csv` | Per step: latency, phone wait, phone work, frames dropped. |

---

# The reaction-time audit, 2026-09-10

Everything above measures the *trainer* side of the loop. None of it measured
what happens after `drive()` returns, and that turned out to be where almost
all of the delay is.

The method mattered here, because most of the numbers this project trusted were
self-reported by the same code being measured. Three rules were applied:
a delay is only believed if it is recovered by **cross-correlating a signal we
injected against a signal we measured**, both sampled at the same instants on
one clock; the one timestamp that cannot be avoided (`SensorEvent.timestamp`)
is dated against **both** candidate clock bases so the base is measured rather
than assumed; and a filtered signal is raced against an unfiltered one, because
a filter's group delay appears in no timestamp. `tools/robot_latency_probe.py`
does all three.

## 1. The RP2040 is the reaction time

A `SET_MOTOR_LEVELS` command is a blocking request/response over USB. Split:

| stage | mean | p95 |
|---|---|---|
| mailbox wait (`setMotorLevels` → writer picks it up) | 18.4 ms | 39.9 ms |
| `prepareForCommand` | 0.08 ms | 0.12 ms |
| USB write | 1.0 ms | 1.5 ms |
| **waiting for the RP2040 to answer** | **81.5 ms** | 86.9 ms |
| first reply byte | 74.2 ms | 77.6 ms |

**Effective wheel command rate: 12.0 Hz against a 49.5 Hz control loop. 59% of
actions are overwritten in the writer's one-slot mailbox and never reach the
wheels at all.** Every timing number the robot reported before this stopped at
`drive()`, which only fills that mailbox, so none of this was visible.

Three hypotheses were tested and two were killed:

- *Are we catching the tail of a periodic cadence?* No. Throttling commands to
  3 Hz, so each write lands at a random phase with the link otherwise idle,
  left the reply at 75.8 ms first byte (p50 75.8, max 80.7). It is a fixed
  per-command cost, not a wait for the next slot.
- *Is it host-side USB contention?* No. The reader parks in a `bulkTransfer`
  with an infinite timeout, which by the library's own documentation can block
  a write from another thread -- but the write measured 0.86-1.0 ms throughout,
  and bounding the read timeout to 20 ms did not shorten the reply. It only
  lost bytes between transfers, so packets stopped assembling, `awaitPacketReceived`
  hit its 10 s ceiling and the effective rate fell to 2.3 Hz. **Do not set a
  read timeout**; there is a comment in `UsbSerial.kt` saying so.
- *Does not waiting for the reply help?* Not on its own. With
  `ControlLatencyTrace.useAsyncMotor(true)` the block simply moves into
  `port.write`, which then takes 81 ms because the microcontroller is not
  draining its USB input. Effective rate 14.1 Hz, barely better than 12.0.

That leaves the firmware: the RP2040 services roughly one command per 75-80 ms.
Its source is not in either repo, so this is the boundary of what can be fixed
from the phone.

**Still open, and it needs the wheels to turn:** whether the microcontroller
*applies* the PWM promptly and merely *replies* late, or does both at the end
of its cycle. The difference is between ~20 ms and ~100 ms of actuation delay
and it decides how much of this matters. Hold the robot or stand it so the
wheels spin free, then:

```bash
.venv/bin/python tools/robot_latency_probe.py --signal prbs --amp 0.4 \
  --period 0.4 --seconds 40 --out /tmp/step.csv
```

The probe cross-correlates the injected command against wheel speed (the
actuation path alone) and against the gyro (actuation plus sensing).

## 2. `Outputs.setWheelOutput` rate-limits every action, silently

It clamps each call to a change of **±0.4** (`maxChange`), logs a warning at a
level that is compiled out, and reports nothing. The clamp advances per *call*,
so a full −1 → +1 reversal takes five control ticks: **~100 ms at 50 Hz**,
~180 ms at the 27.5 Hz the onboard loop ran at -- and the 12 Hz serial path
then samples that ramp. Worse, replay records the action the policy
*requested*, not the clamped value the wheels received, so the world model is
being trained on a control signal the robot never applied. `req_l/req_r`
versus `sent_l/sent_r` in the probe's CSV shows the gap directly.

## 3. Two numbers we had were wrong, and one was misread

- `imu_stale_ms` was **negative** (−10.7 ms). It read `imu_stamp` when the
  message was built, ~17 ms after the tick it subtracted, so it dated each
  observation by a callback that had not happened when the observation was
  taken. `sample()` now snapshots the sensors and their stamp together;
  it reads +3.1 ms mean, +10.4 ms p95.
- The sensor dict was read field by field while the sensor thread wrote it, so
  one observation could mix two samples. Same fix.
- **The on-device policy was never the 16.7 ms.** That figure is
  `phone_work_ms`, which spans sample → policy → `drive()`. Measured on the
  Pixel 3a, the numpy policy costs **3.49 ms**: encode 0.44, core 1.52, stoch
  0.55, actor 0.74. `drive()` is 1.3 ms. The rest was the tick structure.
  The numpy port does not need optimising.

## 4. Two assumptions checked out

- **The sensor clock base is right.** `SensorEvent.timestamp` dated against
  `elapsedRealtimeNanos` gives 7-10 ms; against `nanoTime` it gives −225003 s,
  i.e. the accumulated deep-sleep offset. So `imu_age_ms` was always on the
  correct base. IMU age is 9.4 ms mean, 21.5 ms p95.
- **The rotation vector adds no hidden fusion lag.** Racing it against the raw
  gyroscope -- newly registered at the same 197 Hz, `SensorLatencyTrace` --
  the fused angle's derivative lags `gyro_x` by **−0.6 ms** (r=0.59). Note this
  was taken with the robot stationary, so the correlation is over sensor noise;
  it should be repeated while the robot is moving before the question is closed.

## 5. Where the budget actually goes

| stage | cost |
|---|---|
| IMU hardware → callback | 9.4 ms |
| callback → sample used | 3.1 ms |
| policy (on-device) | 3.5 ms |
| `drive()` → serial mailbox | 1.3 ms |
| **mailbox + RP2040** | **~100 ms**, and the wheels update at 12 Hz |

The on-device policy was built to remove a 20 ms tick delay from a budget that
is dominated by a 100 ms term nobody had measured. It is still the right
change, but it is not the change that matters.

## Tools added

- `tools/robot_latency_probe.py` -- trainer-side probe and analysis.
- `ControlLatencyTrace` / `SensorLatencyTrace` in abcvlib, both surfaced into
  the obs frame. Note they are read through `snapshotJson()`: Kotlin `var`s in
  an `object` are not reachable as Python attributes through Chaquopy, and
  reading one throws from inside the control loop.

---

# The wheel test, and the firmware, 2026-09-14

## 1. A lifecycle bug had the policy acting on 28-second-old IMU data

The first wheel run was unusable: `imu_age_ms` was **21,000-31,000**, the
control loop reported 50 ms of work per 20 ms tick, and `step` occasionally
went backwards. The process had **three `sensorThread`s, two `batteryThread`s,
two `wheelDataThread`s**. `UsbSerial` fires `onSerialReady` on every USB
attach; `MainActivity.initPython()` then calls `abcvlib.run()`, which ran
`main.setup()` again and started a second infinite `loop()` beside the first
-- the Kotlin `cancelAndJoin` cannot stop a coroutine that is inside Python.
Every sensor event then fanned out through duplicate handlers and Python
callbacks and the sensor queue fell behind by tens of seconds. Nothing logged
an error.

This is almost certainly the "8.5 second delivery backlog" of the earlier
commit too, which was attributed to too many sensors being registered. And an
RP2040 reset under motor load (§3) is exactly what re-enumerates USB and
triggers it mid-run.

Fixed in `abcvlib.py`: `run()` builds publishers and loop once per process; a
repeat call only lets the framework restart the serial writer and outputs.
The phone now also prints and shows `STALE SENSORS` when a sample is older
than 250 ms. After the fix: one thread of each, IMU age 9.4 ms, 197 Hz.

## 2. Command → wheel: 180-250 ms, and 1.1 s freezes under load

Robot on its back, wheels free, PRBS ±0.4 at 0.4 s dwell, 40 s:

| | lag behind the phone's `sent` | r |
|---|---|---|
| wheel speed (from the RP2040 reply) | **≈ 250 ms** | 0.48 |
| d(encoder count)/dt | ≈ 180 ms | 0.16 (8 Hz steps) |

Of that, ~80 ms is the serial round trip, the rest is motor reversal at 2 V
plus the ~120 ms encoder-speed window. The IMU side of the test is void in
this posture: free-spinning wheels put no torque into the body and every gyro
axis read 0.00. Repeat on the wheels to get command → gyro.

Two things only appear when the motors run:

- **Stalls: 7-12 per 40 s, each 1.09 s.** During one, the wheels hold the
  last command and the policy's newer ones pile into the mailbox. Not the
  driver: the DRV8830 fault register (bit 2, UVLO) was set *after* 2 of 7
  stalls and never before one. Not the battery: 3.98 V throughout.
- Reported wheel speed spikes to ±25,000 across a stall (the speed window
  straddles it). Clip before correlating.

## 3. The firmware explains both numbers

`oist/smartphone-robot-firmware` is public; cloned to
`~/StudioProjects/smartphone-robot-firmware`.

- **The 75 ms**: `handle_packet` → `SET_MOTOR_LEVEL` calls `set_motor_levels`
  (fast, two I2C writes) and then `get_state`, which re-reads the DRV8830
  faults, the MAX77976 charger, two ADCs, and five BQ27742 fuel-gauge values
  over I2C -- and along the way makes up to **77 `rp2040_log` calls**, several
  with 100+-character strings, each a double `vsnprintf` (with `%f` on a core
  without an FPU), or a synchronous `uart_puts` at 115200 baud if built with
  `LOGGER=UART`. Only after all that is the reply written. So the motors *are*
  set promptly; it is the next command that cannot be read until the dump is
  done.
- **The 1.09 s**: `include/robot.h` has `I2C_TIMEOUT _u(1000000)` -- one
  second -- with `MAX_RETRIES 3`. One NAK under motor noise costs exactly one
  timeout plus one service; three would `assert(false)` and reset the board.

A patch is prepared on branch `motor-command-latency` in that clone (commit
`a07253c`): charger/fuel-gauge reads cached and refreshed at most every
500 ms, packet layout unchanged; `I2C_TIMEOUT` 20 ms. **Not built and not
flashed** -- the build needs the project's Docker image, and flashing needs
the Pico on this Mac's USB with BOOTSEL held (README §Flashing). Expected
effect: round trip from ~82 ms to the order of 5 ms, wheel rate from 12 Hz to
the controller's 50 Hz, and no second-long freezes.

## 4. Still open

- Flash the firmware, then re-run `tools/robot_latency_probe.py --signal prbs`
  and confirm `ser_service_ms` and the stall count.
- Command → gyro lag with the robot on its wheels (hold it; the wheels need
  load for the body to react).
- Replay records the policy's requested action, not the ±0.4-per-call slewed
  value the wheels got (`Outputs.setWheelOutput`). Whether to record `sent`
  instead, or drop the slew for this app, is a training-design decision.

---

# Phone-side fixes, 2026-09-14 (no firmware change)

The firmware's 12 Hz ceiling cannot be moved from the phone. What could be
moved: which actions get applied, how stale they are, and -- it turned out --
the 1 s stalls entirely. All measured with `tools/robot_latency_probe.py`,
robot on its back with wheels free, PRBS ±0.4 at 0.4 s dwell.

## 1. Pace the control loop on the serial link, not a clock

`main.py` `PACE = 'serial'`: the phone blocks until the RP2040's reply has
landed (`ControlLatencyTrace.serialIdleUs()`), waits `WAKE_LEAD_MS` = 6 ms so
the write lands just before the firmware's next 10 ms USB poll, then samples,
decides and drives. One decision per microcontroller cycle. The trainer can
send `pace: 'clock'` in the handshake for the old behaviour; `--pace` on the
probe.

| | clock paced (before) | serial paced |
|---|---|---|
| actions applied | 29-41% | **100%** |
| mailbox wait | 14-18 ms mean, 40 ms p95 | **0.4 ms** |
| sample → drive (`work_ms`) | 16 ms (spans the tick sleep) | **3 ms** |
| drive-to-drive | 20 ms tick, wheels at 12 Hz at random | 112 ms trainer-driven, ~93 ms onboard |

Trainer-driven, the cycle is reply → obs (+8 ms) → action back (+17 ms) →
drive; the action is applied at the firmware's next poll, so sensor-to-torque
is ~25-35 ms. Onboard, the policy replaces the 17 ms network hop with 3.5 ms.
The trainer echoes the phone's `seq` in every action so a late reply to the
previous observation is never mistaken for the current one.

**Trap, and its symptom:** the first version signalled "idle" from a single
pending flag that the *reply* cleared. When a command was queued while the
previous one was in flight, the previous reply cleared it with ours still
waiting, so every command ran exactly one round trip behind -- visible as a
50 ms mailbox wait on a link that was supposedly free, and dequeue stamps
identical to the previous reply's. Idle is now "mailbox empty AND nothing in
flight". Lining events up on one clock (`t_queued`/`t_dequeued`/`t_reply`
against Python's `time.monotonic`, which is the same `CLOCK_MONOTONIC`) is what
found it; the summary statistics could not have.

## 2. The 1 s stalls are coast → drive, and they are gone

With every action now applied, the stalls got *more* frequent (14-15 per
30 s), and every one followed a `sent = 0.0` step. Two things put a zero on the
wire: abcvlib's `setWheelOutput` slews by at most ±0.4 per call, so a
reversal passes through 0.0; and its scaling maps |cmd| < 0.097 (0.49 V) to
coast, H-bridge off. Re-energising the bridge is the trigger: the DRV8830
drops out (its fault register shows UVLO afterwards), stops ACKing I2C, and
the firmware waits its full 1 s timeout.

| reversal ±0.4 | stalls / cycles | faults |
|---|---|---|
| via 0.0 (slew 0.4, abcvlib default) | 15 / 139 | 19 |
| direct (slew 2.0) | **0 / 268** | 0 |

| PRBS between 0.4 and 0 | stalls / cycles | faults |
|---|---|---|
| zero = coast (abcvlib) | 9 / 188 | 9 |
| zero = brake | **0 / 266** | 0 |
| zero = min drive | **0 / 268** | 0 |

Defaults now: `SLEW_MAX = 2.0` (no ramp) and `ZERO_MODE = 'min'` -- inside the
dead zone the phone sends ±`MIN_DRIVE` = 0.13 with the sign of the last
command, i.e. ~0.66 V on a 5 V motor: below stiction, no torque, bridge never
drops. 0.13 rather than 0.10 because abcvlib's scaling gives DRV8830 register
0x05 for anything under 0.111, and 0x05 is reserved; 0x06 is the floor.
`'brake'` (short the motor) is the alternative and also gave zero stalls; it
adds damping at zero, which `'min'` does not. Both are per-action overridable
by the trainer (`slew`, `zero`) and on the probe (`--slew`, `--zero`).

A balancing policy lives in the dead zone, so under the old defaults every
small correction followed by a larger one was a coast → drive step. That is
worth remembering when reading the earlier training curves.

Validation with pure defaults, 30 s each: PRBS ±0.4 -- 0 stalls, 0 faults,
0 overwritten, reply p95 87 ms; PRBS 0.4↔0 -- the same.

## 3. Operational

- **The phone must stay awake and foreground.** A run today died silently
  because the screen went off: `onPause` stops the serial writer and the
  publishers, the IMU dict freezes (staleness climbed to 400 s), and Android
  throttles the socket. `keepScreenOn` is set, so this was the power button.
  `adb shell dumpsys power | grep mWakefulness` says `Dozing` when it has
  happened; `input keyevent KEYCODE_WAKEUP` + `am start` brings it back, but
  the process should be restarted after.
- Episode `length` in the config is in steps; at ~9-11 Hz a 1000-step episode
  is 90-110 s, not 20 s.

## 4. What the firmware patch would still add

Serial pacing makes the 83 ms cycle clean; it does not shorten it. The branch
`motor-command-latency` in the firmware clone would take the round trip to a
few ms and let the loop run at the controller's rate. The I2C timeout change
there is now less urgent, since the phone no longer provokes the UVLO.

---

# Driving the phone side to the floor, 2026-09-16

Question: with the firmware fixed at one command per ~83 ms, how low can the
phone get sensor-to-torque, and how close to the firmware's rate? Answer: both
modes now sit on the floor, and the floor is set by one detail of the
firmware's USB polling that no phone-side change can move.

## 1. Where the policy's time was really going

`policy_ms` in the live onboard loop read **20-25 ms** against 3.5 ms in the
startup benchmark. Three causes, separated one experiment at a time:

| change | policy p50 / p95 | hot benchmark, sensors live |
|---|---|---|
| as found | 21-25 / 32 | 10.8 ms |
| sensor subscribers moved to Kotlin (`SensorSnapshot.kt`) | 10.0 / 24 | **3.35 ms** |
| + control thread pinned to the big cores (6-7) | 9.9 / **10.3** | |
| + GRU half precomputed after the previous action | **6.3 / 6.5** | |

- The ~200 orientation callbacks a second each crossed Chaquopy and took the
  GIL from the policy; that alone was 3.5 → 10.8 ms hot. In the other
  direction the callbacks queued behind the policy, which is why IMU age was
  9 ms p50 / 21 p95. Now **3.8 / 5.6**. No sensor callback enters Python any
  more; `sample()` reads one JSON string.
- Pinning removed the tail, not the mean: the remaining 10 → 3.4 gap is the
  core being cold after ~80 ms idle. Neither a busy-wait nor 1-5 throwaway
  steps before the reply fixed it (the last warm step reaches 5.5 ms, the real
  one after a 1 ms sleep is back to 8), so the hot number is only reachable
  by never sleeping, which is not worth the power.
- `NumpyPolicy.step` is now `precompute(carry)` -- the GRU, half the weights
  streamed -- plus `finish(obs, ...)`. `PolicyRunner.prepare()` runs the first
  half in the slack after each action; the parity test covers both paths and
  `step == precompute+finish` is asserted exactly.

Also: a busy-wait of any length starved the sensors (IMU age → 1 s) while the
callbacks were in Python. Chaquopy does not release the GIL on Java calls.

## 2. The RP2040's poll phase, measured, and why 92.5 ms is the floor

Sweeping when the command is written (relative to Android seeing the reply)
and reading `ser_first_byte_ms`:

| written at | first byte | cycle |
|---|---|---|
| +5.5 ms | 78.8 | 92.4 |
| +7.4 | 78.6 | 92.6 |
| +12.4 | 73.4 | 92.8 |
| +16.3 | 79.3 | 102.3 |
| +25.3 | 72.3 | 105 |

A sawtooth with a 10 ms period: the firmware sleeps 10 ms between USB polls,
and **the first poll a command can catch is ~13.5 ms after the reply is seen
on the phone**. Writing earlier -- even deciding *before* the reply lands and
writing at +1.5 -- catches the same poll. So the cycle is 81 (round trip) +
~11.5 = **92.5 ms, 10.8 Hz**, and the last 10 ms is the microcontroller's,
not ours.

What the sweep does buy is freshness: any write before +13 catches the poll,
so the observation can be taken as late as +3..4 ms (`WAKE_LEAD_MS = 3`)
instead of at +0. Onboard, sensor-to-torque is now roughly IMU age 3.8 +
sample-to-poll 10 + parse/set 1.5 ≈ **15 ms**, at 10.8 Hz, every action
applied. Before this work it was ~50-60 ms with the wheels taking a random
third of the actions at 12 Hz and freezing for a second at a time.

Trainer-driven mode gets the same treatment from the other side: the
observation is shipped `TRAINER_PRESAMPLE_MS = 12` before the reply is due,
the action is back in ~10 ms, and the write lands at +5 -- cycle **91.7 ms**,
down from 112. Its observation is ~10 ms older at apply than onboard's.

## 3. Also found on the way

- **The phone powers the Pico.** `dumpsys usb` shows `source_power=true`
  when healthy. On 09-14 the port dropped to `no-power` after
  "Queueing USB request failed", and the Pico then re-enumerated and died
  150 ms later, once a second, **for two days** -- through app restarts, with
  the app force-stopped, no firmware watchdog involved. Reinstalling the app
  re-ran the USB permission grant, the port re-negotiated, and it stopped
  within a second. If the Pico is bouncing: `adb shell dumpsys usb | grep
  source_power`; reinstall (or revoke and re-grant USB permission).
- `wait_for_serial()` could hang forever on a dead link with the trainer
  connected and no observation ever sent. Bounded at `SERIAL_WAIT_MAX_MS` =
  250; frames then carry `ser_ok = false`.
- abcvlib's `Logger.i/d/v` are live in debug builds and hex-dump every USB
  chunk on the reader thread. `Logger.setQuiet(true)` in `MainActivity`.

## 4. Defaults now (phone `main.py`)

`PACE='serial'`, `WAKE_LEAD_MS=3`, `PREPARE_AHEAD=True`, `PIN_CPUS=(6,7)`,
`TRAINER_PRESAMPLE_MS=12`, `SLEW_MAX=2.0`, `ZERO_MODE='min'`,
`SERIAL_WAIT_MAX_MS=250`. `WARM_MS`, `SPIN_BEFORE_MS`, `PRESAMPLE_MS` exist
for experiments and are off. All are overridable per frame by the trainer
(`tools/robot_latency_probe.py` exposes each as a flag).

Validation, pure defaults, 30 s each: onboard cycle 92.8 p95 94.6, first byte
p90 74.6, 0 stalls; trainer-driven PRBS ±0.4 cycle 91.7, 334/334 applied,
0 stalls.

---

# "A colleague balances at 200 Hz with PID" -- what that means, 2026-10-05

Checked from the code, without the robot (it was off the base). Short answer:
our 10.8 Hz ceiling is real **for the firmware we run**, it is not the motors,
and the fix exists upstream but is not on `main` of the firmware.

## 1. The 200 Hz is the controller's loop rate

`apps/pidBalancer` on the `tutorial` branch runs `BalancePIDController` with
`setTimestep(5)` ms -- 200 Hz -- and drives through `setWheelOutput`, i.e. the
same one-slot `SerialCommManager` mailbox measured above. The controller can
*compute* at 200 Hz on any firmware; how many of those commands reach the
wheels is set by the RP2040 round trip. On our firmware that is one per
~92 ms, so ~95% of a 200 Hz PID's outputs would be overwritten. A PID tolerates
that far better than our policy did: every command that does go out is the
freshest one, and its gains can be tuned inside the delay margin. A sim run at
200 Hz with no actuation delay says nothing about this either way.

## 2. The firmware fix exists: `RTT-LoopReduction`

The ~75 ms is the firmware doing a full `get_state()` dump (charger, fuel
gauge, ADCs, faults, dozens of log calls) on every `SET_MOTOR_LEVEL`, plus a
10 ms idle sleep in the main loop (§"The firmware explains both numbers").
The maintainer fixed exactly that on 2026-06-22, on firmware branch
`RTT-LoopReduction` (commit `7bb9193` "Reduce motor command response latency":
`SET_MOTOR_LEVEL` replies with encoder counts only, up to 8 packets drained per
pass, 1 ms sleep only when idle). It is **not merged to firmware `main`**; it
has since been rebased as `split/rtt-loop-reduction` and, with a cached
telemetry refresh, `split/telemetry-cache` (2026-07-30) on the `tekkura/`
remote. The Android half of the same change (no background `GET_STATE`
polling, commit `9373a1ae` on `tutorial`) is already in `dreamer-bridge`.

The firmware repo's own RTT benchmark quotes 8-26 ms, 14.8 ms mean, host to
Pico -- i.e. ~50-120 Hz deliverable, not 200, but 5-10x what we have.

## 3. So, were we wrong?

The measurement was right and the attribution (firmware, not motors) was
right. What was wrong was treating it as a fixed ceiling: a newer firmware
build than ours removes most of it. To confirm, ask the colleague which
firmware is flashed and look at `BasicAssemblerTrace SERIAL_RX ... rttMs=` in
their logcat. Then: build `split/telemetry-cache` (Docker image
`topher217/smartphone-robot-firmware`, `make firmware`), flash it on the Mac
(BOOTSEL), and re-run `tools/robot_latency_probe.py`; `PACE='serial'` on the
phone will follow the faster cycle automatically.
