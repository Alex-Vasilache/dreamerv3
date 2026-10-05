# Running the policy on the phone

Status 2026-09-07: the policy itself is built and verified against the JAX
modules; the phone-side wiring is not done yet. Nothing here has been installed
on the robot, and the live training run was not disturbed.

## 1. Why: the reaction delay, measured

The question that started this was "if the robot is balanced and something
pushes it, when do we notice and when do we act?" Measured on the live run
(60,000 consecutive steps, `~/logdir/robot_local/env0/timing.csv` plus the
`epstats/log/*` series in `metrics.jsonl`):

| stage | mean | max | source |
|---|---|---|---|
| IMU hardware timestamp → Android callback | **7.4 ms** | 34.8 ms | `log/imu_age_ms` |
| callback → observation put on the wire | **2.9 ms** | 33.6 ms | `log/imu_stale_ms` |
| observation shipped → action applied | **20 ms** | — | structural, see below |
| `setWheelOutput` → USB serial → Pico → PWM | not measured | | |
| **total, IMU to torque** | **≈30 ms + actuation** | | |

The network is **not** in the critical path. RTT to the phone is 5.8 ms mean
(`ping`, 15 packets), the actor sustains 49.0 Hz against the phone's 49.4 Hz
tick, and only 0.5% of steps discard a stale frame. The pipelined protocol is
doing its job: the round trip hides inside the control period.

### Where the 20 ms comes from

`main.py:loop()` on the phone runs, in this order:

```
read_nowait()      -> newest action, if one has arrived
drive(action)                       <- action applied here
wait_for_tick()    -> sleeps out the rest of the 20 ms period
sample sensors, write obs           <- observation shipped here
(loop)
read_nowait()                       <- only ~0.1 ms after shipping
```

The observation goes out at the end of a tick, and the very next `read_nowait`
happens a fraction of a millisecond later -- far too soon for the reply, which
needs ~6 ms. So the action prompted by that observation is not picked up until
the *following* tick: a full 20 ms later. This is visible in the data as
`latency_ms` (time from the actor sending an action to it receiving the next
observation) sitting at 16.5 ms mean, i.e. very nearly a whole period.

Note `phone work_ms` (17.7 ms mean) is **not** the phone being saturated --
it spans `wait_for_tick()`, so it is mostly sleep. The phone is idle.

### Is 30 ms too slow?

Yes, probably. From a replay chunk, tilt excursions in the 0.3°--3° band grow
at a 90th-percentile rate of λ ≈ 20 s⁻¹, i.e. a time constant of **τ ≈ 50 ms**,
and theta covers the full bumper range (−7.8° to +10.2°) within a 4-second
chunk. A 30 ms delay is ~0.6 τ: by the time the first corrective torque lands,
the disturbance has grown by e^0.6 ≈ 1.8×.

Caveat on that number: it is measured closed-loop, with a policy that is itself
injecting noise, so it mixes plant divergence with policy excitation. A clean
figure needs an open-loop test -- wheels disabled, release from ~2°, log theta
-- which is worth doing before tuning anything against it.

### What actually helps, in order

1. **Apply the action in the tick that produced the observation.** This is the
   20 ms term and it dwarfs everything else.
   - With the policy on-device: sample → compute (~2-4 ms) → apply, all inside
     one tick. Total delay ≈ 7.4 + 2.9 + 3 ≈ **13 ms**.
   - Without it: after shipping the observation, block up to ~10 ms for the
     reply instead of looping immediately. Total ≈ **16 ms**, for about five
     lines. But it re-couples the control loop to the WiFi, which is exactly
     what the pipelining was built to avoid -- and we watched the link drop the
     robot twice today.
2. **The IMU.** 7.4 ms hardware→callback is the largest remaining term once the
   tick delay is gone. Check what sensor delay the orientation publisher
   registers with; if it is not `SENSOR_DELAY_FASTEST`, that is free latency.
3. **A faster control rate.** 100 Hz halves the structural tick term, but after
   fix 1 that term is already gone, and it doubles what the learner must
   consume. Third-order lever, not first.

So: the on-device policy is worth doing, but *not* because the network is slow.
It is worth doing because it is what lets the loop close within a single tick,
and because it removes the WiFi as a thing that can stop the robot.

## 2. What is built

### Why not TFLite

The obvious path (`jax2tf` → LiteRT) fails on the requirement that matters:
**TFLite bakes weights into the model file at conversion time.** There is no
supported runtime weight swap, so every update from the learner would mean
re-running the converter and shipping a new model. Since the whole point is a
continuous stream of weight updates, that is the wrong shape.

The app already embeds **Chaquopy**, and the control loop is already Python
(`apps/dreamerBridge/src/main/python/main.py`). So the policy can just be numpy,
and a weight update is a plain `.npz` that gets swapped in place.

### Pieces

| file | what |
|---|---|
| `dreamerv3/deploy/numpy_policy.py` | the acting path -- encoder, RSSM observe, actor head -- in pure numpy, no repo imports, meant to be copied to the phone |
| `tools/export_policy.py` | policy `.pkl` or checkpoint dir → compact `.npz` + manifest |
| `tools/test_numpy_policy.py` | parity against the real `rssm.Encoder` / `rssm.RSSM` / `MLPHead` |

Verified:

```
$ .venv/bin/python tools/export_policy.py --checkpoint <policy.pkl> --out policy.npz
wrote policy.npz  (5.43 MB, 1,353,860 params, 41 arrays)

$ JAX_PLATFORMS=cpu .venv/bin/python tools/test_numpy_policy.py --weights policy.npz
tensor      max rel err
tokens         1.27e-06
deter          8.43e-07
logit          6.45e-07
mean           9.81e-07
stddev         1.42e-06
OK
```

The test drives the actual ninjax modules with the exported weights and
compares every deterministic tensor. The stochastic sample is taken from the
JAX side and injected into the numpy side, so the actor comparison is exact
rather than a distribution check that would pass with the wrong weights.

Cost, measured on this Mac: **0.204 ms** per policy step, **2.2 ms** to reload a
full weight set. Measured on the Pixel 3a itself (2026-09-10): **3.49 ms** --
encode 0.44, core 1.52, stoch 0.55, actor 0.74 -- against a 20 ms budget, so
the prediction held and the port needs no optimising. The 16.7 ms that
`phone_work_ms` reported is the whole sample → policy → `drive()` span, not the
policy; see `docs/ROBOT_BRIDGE_FINDINGS.md`.

Side benefit: the export is **5.4 MB against the 33 MB** the bridge currently
ships, because four fifths of a policy pickle is Adam moment state for parts the
actor never touches.

### Android build

`apps/dreamerBridge/build.gradle.kts` now asks Chaquopy for numpy. Two things
were needed to make that resolve, both done and verified by a successful
`:dreamerBridge:assembleDebug`:

- `version = "3.13"` -- Chaquopy 17 requires buildPython to match the app's
  Python minor version exactly, and 3.13 is what this Mac has (the default is
  3.10, not installed).
- ABIs cut to `arm64-v8a, x86_64` -- Chaquopy offers Python 3.13 for those two
  only. The Pixel 3a is arm64.

numpy 1.x is confirmed present in `assets/chaquopy/requirements-arm64-v8a.imy`.
The APK was built but **not installed**, so the running robot is untouched.

## 3. Wiring (built 2026-09-07, not yet installed)

Protocol bumped to 3. Onboard mode is negotiated, not assumed: the phone offers
it only if weights actually loaded, the trainer accepts only if
`env.robot.onboard` is set. A phone with no policy, or a trainer with the flag
off, behaves exactly as before.

**The reorder is the point.** `loop_onboard()` paces first, then samples,
decides and drives back to back:

```
wait_for_tick()          # pace at 50Hz
obs, act = runner.act(sensors)
drive(act)               # applied in the SAME tick
write(obs + act)         # off the critical path
drain_control()          # resets, weight updates
```

Everything not on the sensor-to-torque path happens afterwards, in the slack of
the same tick.

**Who owns what.** The trainer keeps episode boundaries and reward -- it is the
side that knows the length and the termination rule -- and sends them as a
`ctrl` frame. Only the timing is relaxed: a reset one tick late costs nothing,
unlike an action one tick late. The phone owns the policy and the RSSM carry.

**Recording the executed action.** The actor still runs its own policy (the
driver requires one) but that action must not reach replay: the world model has
to learn what the wheels did. Envs run in separate processes, so the phone's
action comes back through the obs dict on a new generic `executed/<key>`
channel that `embodied/core/driver.py` lifts over `acts` when it assembles the
transition, and masks on `is_last` exactly as it masks `acts`.

**Weight delivery** rides the blob the wire format already had; `parse()` now
returns it instead of skipping it, and `read_nowait` drains the socket rather
than taking one 64KB bite per tick (which would stretch a 5MB update over
nearly two seconds).

Verified:

| check | how |
|---|---|
| numpy port == JAX modules | `tools/test_numpy_policy.py`, ~1e-6 on every deterministic tensor |
| `executed/` overrides the actor, is masked, and is inert when absent | `embodied/tests/test_driver_executed.py`, 4 tests |
| handshake, `ctrl`-only downstream, phone's action reaches the transition | `tools/test_onboard_protocol.py`, real env vs a socket playing the phone |
| nothing else regressed | full suite: 1199 passed (5 pre-existing `ManagerRingHead ... nu` failures in the HRL arm, unrelated) |

## 4. Weight push and standalone control

**Push.** The env exports the newest learner policy to the on-device `.npz` and
sends it down the existing blob channel, at most every `env.robot.weights_every`
seconds (default 30). It is `dreamerv3/deploy/export.py` doing the packing, the
same code `tools/export_policy.py` uses, so the manifest cannot drift between
the offline and online paths. A multi-megabyte `sendall` here blocks the
*trainer*, not the robot -- the phone is free-running on its own policy and is
not waiting for us, which is precisely the property that made moving the policy
onto the phone worth doing.

**Bootstrap.** A phone that boots with no weights negotiates `onboard=False` and
the trainer drives as before. The push is keyed on the *configured* flag rather
than the negotiated one, so weights still go down; when they land the phone
installs them, drops the link, and the next handshake enables onboard. No adb
step is needed to bring a fresh phone up -- though for this first install the
policy was also pre-exported so it takes over immediately.

**Standalone.** Once a trainer has agreed the phone holds the policy, that is
sticky across disconnects. If the link dies the phone keeps sampling, deciding
and driving at 50Hz and retries the connection between ticks; the experience it
collects meanwhile is simply not recorded. Stopping the wheels is what makes a
balancing robot fall over, so the old `stop_wheels()` on disconnect is now
conditional on *not* being able to control. The GUI shows `standalone (no
trainer)` while this is happening.

This is a real change in unattended behaviour: the robot will now keep
balancing indefinitely with nothing watching it, until its battery runs down or
someone stops the app.

## 5. What is left

**Read `docs/ROBOT_BRIDGE_FINDINGS.md` first (2026-09-10).** The premise of
this document -- that removing a 20 ms tick delay from a ~30 ms budget is the
first-order fix -- is wrong. The budget is ~130 ms, and ~100 ms of it is the
serial round trip to the RP2040, which caps the wheels at 12 Hz: against a
50 Hz control loop, three ticks in four change nothing. The on-device policy is still worth
having; it is not the lever that matters.


- **Install and measure.** None of this has run on the robot. The first install
  should confirm `work_ms` in the onboard path (now the real decide cost, not a
  sleep) is the ~2-4ms the Mac benchmark predicts, and that the sensor-to-torque
  budget actually fell.
- **Push the weights.** The trainer accepts onboard mode and records actions,
  but nothing yet sends a `weights` frame down; the phone side to receive one is
  written and the export is a `tools/export_policy.py` call away.
- **Decide the trainer-absent behaviour.** The phone can now keep balancing when
  the trainer goes quiet, instead of coasting to a stop. That is a real change
  in what the robot does unattended, and it is currently still governed by the
  legacy `ACTION_TIMEOUT` path.
- **The IMU handler priority** was raised to `THREAD_PRIORITY_URGENT_DISPLAY`;
  worth re-measuring `imu_age_ms` against the 7.4ms baseline once installed.
