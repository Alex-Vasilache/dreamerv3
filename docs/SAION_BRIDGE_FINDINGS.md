# Training the robot with the learner on Saion: what works and what does not

Measured 2026-09-03 from the MacBook, on both OIST networks. Reproduce with
`tools/test_saion_link.sh` (about two minutes, every step guarded so a stall is
a result rather than a hang).

## The conclusion first

**Which network the MacBook is on decides everything.** The link is not slow and
it is not flaky; the `login.oist.jp` bastion path is *policed* to about 10 KB/s,
and the direct internal route is 200x faster. The same bridge fails on one and
has 7-13x headroom on the other.

| | oist-public via `saion-ext` | OIST internal via `saion` |
|---|---|---|
| Raw ssh throughput | **~10 KB/s** | **2.3 MB/s up, 2.1 MB/s down** |
| One 390KB chunk per rsync | under 6 KB/s | 56-68 KB/s (setup-bound) |
| 12 chunks in one rsync (4.7MB) | under 38 KB/s | **683 KB/s** |
| 25 chunks as tar (9.8MB) | under 79 KB/s | **1225 KB/s** |
| ssh session setup | 6-11 s | 4-6 s |
| Requirement (390KB / 4.1s at 48Hz) | 93 KB/s | 93 KB/s |

So: **use `saion`, never `saion-ext`, for the bridge.** The scripts now pick it
automatically -- if `dig +short saion.oist.jp` answers, the MacBook is on a
network with the direct route and `REMOTE` defaults to `saion`.

## The bastion path is policed, not congested

Worth recording because every instinct here is wrong. From oist-public the
WiFi itself did 3.3 MB/s down and 1.4 MB/s up to the internet, RTT to
`login.oist.jp` was 4 ms, and a 300KB ssh upload that averaged 3 KB/s produced
**zero TCP retransmissions**. No loss, no latency, no MTU problem. And 1, 4 and
8 concurrent ssh uploads all summed to the same ~10 KB/s:

```
1 parallel x200KB   21.95s   aggregate    8.9 KB/s
4 parallel x200KB   90.01s   aggregate    8.7 KB/s   (did not finish)
8 parallel x200KB  110.02s   aggregate   14.2 KB/s   (did not finish)
```

That is a token bucket on the path, so no amount of batching, compression,
parallelism or protocol choice recovers it. `ssh -o IPQoS=none` and alternate
ciphers changed nothing. Ports 80, 443 and 8443 are closed on `login.oist.jp`,
so there is nothing to tunnel over either.

## The direct route exists from both networks

`saion.oist.jp` is `10.210.28.18`, and it is *routable even from oist-public*
(5 hops, 4 ms, 0% loss) -- only TCP/22 is firewalled there, which is what forces
the bastion. On the internal network port 22 is open and the DNS name resolves,
so plain `ssh saion` works. Its host key is byte-identical to the one already
trusted as `saion-ext`.

## Session setup is Saion-side, so the bridge must multiplex

Direct setup is still 4-6 s, and that is not the network: `GSSAPIAuthentication=no`
and `PreferredAuthentications=publickey` make no difference. It is the login
node's own auth path. One chunk per rsync therefore lands at 56-68 KB/s -- below
requirement -- purely because 390KB is dwarfed by a 5 s handshake. Batched, the
same link does 683-1225 KB/s. `tools/robot_bridge_sync.sh` already holds its own
multiplexed connection, which is what makes the difference.

## Two architectures, both measured

**Tunnel the robot to the cluster** (`sbatch/run_robot_v100.sbatch` with
`SCRIPT=train`, `tools/robot_tunnel.sh`): the phone reaches the compute node
through an SSH forward and the whole loop runs on Saion. Measured over the
bastion path it was dead on arrival -- 0.45 Hz consumed, 95.1% of observations
dropped, step gaps up to 203 s. Untested over the direct route; the control loop
would still inherit whatever tail the link has, so the split below stays the
safer shape.

**Actor local, learner on Saion** (`SCRIPT=online_learner` plus
`tools/robot_bridge_sync.sh`): the robot talks to the MacBook at 45Hz and only
replay chunks go up, policy weights come down. The control loop is untouched and
the slow link carries only things that tolerate seconds of delay. Verified end to
end with both halves on one machine (`REMOTE=local`, plain file copies):

```
actor    39.1 Hz, 21.6% dropped, latency p50 18ms
learner  training at 2296 samples/s on chunks delivered only by the bridge
bridge   12 chunks up, 3 policy files down, newest chunk correctly held back
```

Over the direct link the transport now clears requirement by 7-13x, so the
subsetting and float16 workarounds below are no longer needed to make it run.

## The phone: resolved, put it on the internal network too

Do not chase the oist-public firewall. The phone (Pixel 3a, no MDM lock) joins
the internal WPA2-Enterprise network directly, which puts both halves on the
fast side and makes the whole question moot:

| Field | Value |
|---|---|
| EAP method | PEAP |
| Phase 2 | MSCHAPV2 |
| CA certificate | Use system certificates |
| Domain | `radius.oist.jp` |
| Identity | OIST username |
| Anonymous identity | blank |

The domain is the trap. Android 11+ demands one whenever a CA is selected, and
the RADIUS server cert (`CN=radius.oist.jp`) is issued by **GeoTrust/DigiCert**,
a public CA -- so "use system certificates" is correct. The `oistCS-CA` /
`ROOTCS-CA` pair in the Mac's System keychain is OIST's internal AD CA and has
nothing to do with WiFi; installing it as the WiFi CA will fail. Read the real
answer off the Mac with:

```
security find-certificate -a -p ~/Library/Keychains/login.keychain-db |
  openssl x509 -noout -subject -ext subjectAltName
```

Verified 2026-09-03 with both on the internal network: phone `10.13.70.3` to
MacBook `10.13.67.125` at **5.9 ms**, and the app reaching `:3000` -- the failure
before the learner starts is `[Errno 111] Connection refused`, an RST from the
Mac, not a timeout, which is what proves the path is open rather than filtered.

The MacBook's address changes when it moves networks, so
`~/StudioProjects/smartphone-robot-android/config.json` needs the new one and the
app must be rebuilt (`./app run --app dreamerBridge`) -- the IP is baked in at
build time, not read at runtime.

## The learner falls behind, and the failure is silent

Measured 2026-09-03 on a V100. `train_ratio: 512` at 50Hz demands
`512 * 50 / 8192 = 3.125` train steps/s; the V100 delivers about **3.1**. It
runs precisely at capacity, so any hiccup turns into permanent debt --
`elements.when.Ratio` never forgives it.

That debt would be tolerable on its own. What makes it damaging is that
`should_sync_policy`, `should_log` and `should_save` all sit *after* the
`for _ in range(repeats)` loop in `run_learner`. So as the backlog grows, the
learner spends longer inside one uninterrupted batch, and the weight-refresh
cadence collapses non-linearly:

```
last logged learner step   73,334,784
on-ratio it would be      118,988,800
shortfall                  45,654,016 step-units = ~5573 queued train steps
=> ~30 minutes to the next policy publish, and widening
```

Every local signal looks healthy while this happens -- job RUNNING, GPU busy,
chunks flowing, `actor_step` climbing, the learner genuinely training at full
speed. The only symptom is that `online_shared/policy/latest` stops advancing,
so **that stamp is the health check that matters**, not job state or throughput.
The robot meanwhile drives on increasingly stale weights, which is the one thing
the split architecture exists to avoid.

Two fixes, both applied:

1. `run.online_max_train_repeats` (default 32) caps the inner batch. Throughput
   is unchanged -- the debt is still owed and still worked off, just in slices --
   but the publish/log/save cadence gets a floor.
2. Run the robot learner at `--run.train_ratio 384` for ~20% headroom instead of
   sitting exactly at capacity. The `robot_daydreamer` block still says 512; that
   value has a documented rationale, so it is overridden at launch rather than
   rewritten.

## Why the payload is so large

390KB per 200 steps is **1.95 KB per step**, and the observation is four
floats. The bulk is the RSSM carry stored as replay extras:

| | per step |
|---|---|
| `dyn/deter` (256 float32) | 1024 B |
| `dyn/stoch` (32x4 float32) | 512 B |
| observation, action, flags, stepid | ~400 B |

So ~77% of what crosses the link is recurrent state, not sensor data. It cannot
simply be dropped: `Agent.train` asserts the batch keys equal `agent.spaces`,
which includes those extras, and `Consec` uses them for replay context.

## What would have made the bastion path work (superseded)

Kept for the record; the direct route removed the need for all of it.

Ordered by how much they change:

1. **Upload a subset of chunks.** DreamerV3 replays each sample hundreds of
   times (`train_ratio` 512), so a learner does not need every transition -- it
   needs a representative buffer. Keeping 1 chunk in 2 halves the requirement
   to 46 KB/s, which is inside the measured envelope; 1 in 3 gives margin. This
   is the smallest change that fits the measurement and costs only replay
   diversity.
2. **Shrink the stored carry.** float16 for `dyn/deter` and `dyn/stoch` would
   cut the payload ~40% for a small change in `embodied/`, but it touches the
   agent's data contract.
3. **A faster path to the cluster.** The 6-11s per session and the throughput
   ceiling are both the `login.oist.jp` hop; a direct route, VPN, or running
   the MacBook on a wired campus network would remove the problem rather than
   work around it. Worth checking before engineering around a link that might
   simply be misconfigured for this use.

Until one of those is in place, `tools/start_robot.sh` falls back to a local
learner, which sustains ~1600 samples/s (train ratio ~35 against a 512 target):
slower learning, but the robot session is not wasted.

## Traps that cost time here, all fixed

- **SSH multiplexing.** A `ControlMaster` block covering `oist-ext` -- which
  `saion-ext` proxies through -- meant one wedged socket silently blocked every
  later connection, and `ConnectTimeout` does not apply while waiting on a
  control socket. It looked exactly like the cluster being down. Removed from
  `~/.ssh/config`; the bridge and tunnel now force `ControlPath=none`.
- **Watchdogs that kill the wrapper, not the transfer.** Without `set -m` a
  background job shares the shell's process group, so `kill $pid` leaves the
  rsync running; hung transfers stacked three deep while the loop believed it
  had timed them out. `guard()` now uses `kill -- -$pid` with job control on.
- **Stale endpoint files.** A cancelled job's endpoint outlives it when SLURM
  kills the script with SIGKILL, so the EXIT trap never fires. The bridge
  latched onto a dead job's directory and every chunk went somewhere nobody was
  reading -- the robot looked perfect while nothing learned. `start_robot.sh`
  now checks `squeue` for RUNNING and deletes stale files.
- **`find` is `bfs` in the interactive shell** and rejects
  `-newermt '-5 seconds'`, so hand-testing the bridge's file list silently
  produced nothing. Scripts run under bash and get BSD find; use
  `/usr/bin/find` when testing by hand.
