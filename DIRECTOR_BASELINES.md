# Director baselines — seed tracker

Target: **5 seeds of pure Director** per environment. All BIG scale
(`director_match`, batch 16x64, train_ratio 64, replay 1e6).

Buffer: official Director sets `replay_size: 1e6` in `defaults` with **no
per-environment override** (`code/director/embodied/agents/director/configs.yaml`
line 10; only `debug` overrides it, to 500). So every environment below —
including the 8M and 50M ones — uses 1e6, which is what our runs already use.
Nothing to adapt.

Priority is top to bottom.

| # | Environment | Task string | Steps | Seeds | Have | Missing |
|---|---|---|---|---|---|---|
| 1 | Pinpad Five | `pinpad_five` | 4M | **0/5** | ~~e632 e633~~ void (10k episodes) | s4 — e652-e655 running s0-s3 |
| 2 | Pinpad Six | `pinpad_six` | 4M | **0/5** | ~~e624 e625~~ void (10k episodes) | s0-s4 — none launched |
| 3 | Antmaze L | `loconav_ant_maze_l` | 4M | 0/5 | — | s0–s4 |
| 4 | Antmaze XL | `loconav_ant_maze_xl` | 8M | 0/5 | e620:s0 e621:s1 (dead at 1.8M, and 6M-configured) | s0–s4 |
| 5 | Crafter | `crafter` | 4M | 0/5 | — | s0–s4 |
| 6 | Breakout | `atari_breakout` | 50M | 0/5 | — | s0–s4 |
| 7 | Cheetah Run | `dmc_cheetah_run` | 4M | 4/5 | e554:s0 e555:s1 e556:s2 e557:s3 | s4 |
| 8 | Hopper Hop | `dmc_hopper_hop` | 4M | 4/5 | e566:s0 e567:s1 e568:s2 e569:s3 | s4 |
| 9 | Cartpole Swingup | `dmc_cartpole_swingup` | 4M | 4/5 | e502:s0 e503:s1 e504:s2 e505:s3 | s4 |
| 10 | Pendulum Swingup | `dmc_pendulum_swingup` | 4M | 0/5 | — | s0–s4 |
| 11 | Quadruped Run | `dmc_quadruped_run` | 4M | 0/5 | — | s0–s4 |
| 12 | Walker Run | `dmc_walker_run` | 4M | 0/5 | — | s0–s4 |
| 13 | Montezuma's Revenge | `atari_montezuma_revenge` | 50M | 0/5 | — | s0–s4 |

A seed counts as **done** at >= 3.9M steps (runs stop at 3,995,992–3,998,592,
never exactly 4M). Pinpad Five e632 is still running; e633 finished at 3.9M.
Antmaze XL e620/e621 stopped at 1.8M of 8M and do not count.

**Totals: 8 of 65 seeds done. 57 missing.**

## Pinpad episode length — read this before adding pinpad seeds

The Director paper (Sec. 3.1) states *"Episodes last for 2000 steps"*. Both our
`pinpad.py` and the released Director repo default the constructor to
`length=10000`, and neither ships an env override, so every pinpad run before
2026-08-26 used a **5x-too-long episode** (measured `episode/length` = 10,001).
Reward is +10 per completed pad sequence with the position reset after, so score
scales roughly linearly with episode length — those numbers do not compare to
the paper's Figure 5 at any budget.

Fixed by adding `env.pinpad: {length: 2000}` to the defaults. Verified: episodes
now end at exactly 2000. **e624/e625 and e632/e633 are void as pinpad results**;
e644-e647 were cancelled mid-flight and relaunched as e652-e655.

Budgets also checked against the paper's figure axes, and yours were right:
Pin Pad spans **0-6M** (Fig. 5) and Ant Maze **0-9M** (Fig. 4) — not the `1e8`
config default, which those blocks simply never override. Ant Maze uses
**3000-step** episodes and 5 seeds, matching the target here.

## Antmaze config — checked 2026-08-26, no pinpad-style bug

Verified against the Director paper on the four e620–e623 runs:

| Paper says | Ours | |
|---|---|---|
| episodes end after a 3000-step time limit | `episode/length` = **3001**, min = max, ~600 episodes/run | ok |
| 50 Hz control | `time_limit=60` s, `repeat: 1` -> 60 x 50 = 3000 | ok |
| no early termination that could leak task information | `termination=False` -> `aliveness_threshold=-1.0`, `contact_termination=False` | ok |
| first-person camera plus proprioception | `walker_egocentric_camera (64,64,3)` + joints/vel/accelerometer/gyro/touch/velocimeter/appendages/bodies/quats/height/end-effectors/zaxis | ok |
| no privileged information | **zero** global-position or top-down keys in the observation space | ok |

The 3001 is not a coincidence the way pinpad's 10,001 was: it falls out of
`time_limit` x control rate, the same mechanism the paper describes, rather than
a hardcoded step count nobody overrode.

One structural difference, not a defect: Director selects proprio vs image with
DreamerV2-style `mlp_keys`/`cnn_keys` regexes, while we use the DreamerV3
`simple` encoder with no key filters, which consumes every observation key.
Different mechanism, same set of inputs — nothing is dropped.

**The one real deviation is the budget.** e620–e623 were configured
`run.steps: 6e6`; the paper's Fig. 4 x-axis spans **0–9M** and the table above
targets 8M. Those runs died at 1.8M so it is moot, but any relaunch must set 8M
or 9M rather than inherit the 6M.

Task strings confirmed valid: `LocoNav` splits on the first `_`, so
`loconav_ant_maze_l` and `loconav_ant_maze_xl` resolve to walker `ant` plus the
`maze_l` / `maze_xl` entries in `embodied/envs/loconav.py` `MAPS`. Note that
e620–e623 were all **maze_xl**; Antmaze L has never been run.

Antmaze also runs `run.train_ratio: 256` (config block `loconav`), not the 64
used everywhere else. That does not affect the count-novelty decay, which is
derived from `config.run.train_ratio` at construction time and so re-derives
itself, but it does mean 4 env steps per train step instead of 16.

## Capacity

BIG does not fit on one 16 GB card. Batch 16x64 needs ~59 GB on A100; split
across 4 devices it is ~12 GB each, so a V100/P100 run needs **4 GPUs**
(`run_v3_e124_director_baseline_multiv100_routeA.sbatch` line 21, fit validated
by e119–e122). At NGPU=2 each device would need ~24 GB and OOM.

Our quota is 8 GPUs on `gpu-v100` and 8 on `gpu-p100`, so at 4 GPUs per job that
is **2 concurrent jobs per partition**. 80% occupancy is not reachable at this
granularity — per partition it is 50% (1 job) or 100% (2).

Both partitions in use (2026-08-25): **4 concurrent jobs**, 8 V100 + 8 P100.
Measured throughput on the 4-GPU probes: **V100 26.2 env fps** (~43 h per 4M
seed), **P100 18.2** (~61 h).

## Wall-clock, measured

e294 (cartpole, BIG, 4xV100) ran 46.5 h and hit the 48 h wall without finishing
4M. So on 4xV100 a 4M run is roughly **50–70 h**, i.e. 2 requeues. P100 is
slower still.

That makes the 50M rows impractical here: at V100 rates Breakout or Montezuma is
**25–35 days of wall time per seed**, 5 seeds each. They need A100s, a smaller
scale, or to be dropped. Antmaze XL at 8M is ~5 days per seed on 4xV100.

Any script used for these must be requeue-safe (job-id RUN_DIR + USR1
self-requeue). `run_v3_e124_director_baseline_multiv100_routeA.sbatch` is **not**
— it builds RUN_DIR with `mktemp` plus a timestamp, so a requeue restarts from
zero, and it has no SEED argument. Use
`run_v3_director_baseline_multigpu.sbatch` instead.

## Conv backend — do not skip this

The A100 scripts `unset DREAMERV3_CONV_IMPL` for native cuDNN conv (~2x). On
V100/P100 that fails outright in the stock env:

```
INTERNAL: All algorithms tried for ... custom-call(__cudnn$convForward) failed
Profiling failure on cuDNN engine ...: CUDNN_STATUS_INTERNAL_ERROR
```

It killed jobs 4700184/4700185 (V100) and 4700186 (P100) inside a minute.
`run_v3_director_baseline_multigpu.sbatch` now branches on
`SLURM_JOB_PARTITION`:

- **V100** — Route-A clone (`/work/.../dreamerv3_env_routeA`: jax 0.4.33,
  cuDNN 9.1.0.70, CUDA 12.2, pinned against the drift that broke sm_70 native
  conv), `--xla_gpu_strict_conv_algorithm_picker=false`, native conv.
- **P100** — same clone, also native conv. Reference conv is NOT a usable
  fallback: forcing it there (job 4700192) OOM'd at 13.4 GB/device, because the
  reference path costs more memory than cuDNN and 4x16 GB only fits BIG at the
  ~12 GB/device native uses. On a 16 GB card it is native conv or nothing.
  P100 emits `slow_operation_alarm` lines during autotuning; those are benign.
- **A100** — unchanged, stock env and native conv.

The clone is activated directly and never sources `source_dreamerv3_env.sh`, so
`WANDB_API_KEY` must be exported explicitly or the run silently degrades to
jsonl-only. Probes 4700191 (V100) and 4700192 (P100) both cleared the failure
with zero cuDNN errors.

## In flight

Submitted 2026-08-25 with `sbatch/run_v3_director_baseline_multigpu.sbatch`
(4 GPUs each, requeue-safe):

| Exp | Env | Seed | Partition | Job | State |
|---|---|---|---|---|---|
| e652 | Pinpad Five | s0 | gpu-v100 x4 | 4700541 | launched 2026-08-25, length 2000 |
| e653 | Pinpad Five | s1 | gpu-v100 x4 | 4700542 | launched 2026-08-25, length 2000 |
| e654 | Pinpad Five | s2 | gpu-p100 x4 | 4700543 | launched 2026-08-25, length 2000 |
| e655 | Pinpad Five | s3 | gpu-p100 x4 | 4700544 | launched 2026-08-25, length 2000 |

These replace e644–e647, which were cancelled mid-flight because they still ran
the 10k-step pinpad episode. All four slots (8 V100 + 8 P100) are full, so
**Pinpad Five s4 and all five Pinpad Six seeds wait for a free slot** — roughly
two days. The V100 pair clears 4M inside one 48h wall; the P100 pair needs one
requeue, which the script handles (job-id RUN_DIR + USR1).

Dead dirs left in `/work` (~24 GB, disk at 23%, not urgent): the two 849 MB
native-conv crash artifacts `e644_..._j4700184` / `e645_..._j4700185`, plus the
four cancelled `e644_..._j4700195`, `e645_..._j4700196`, `e646_..._j4700201`,
`e647_..._j4700202`. All are void as pinpad results, so archiving them to the
bucket buys nothing; they only need deleting.

**e632 finished 2026-08-25 at 4.000M**, last-15 mean **0.0**, peak 1270. With
e633 at **158.0** (peak 1350) that is the full 2-seed picture, and the two seeds
disagree completely: both find reward, only one holds it. The spread dwarfs any
effect worth measuring, which is exactly why 5 seeds are needed here. Pinpad Five
reaches **5/5 once e644–e646 land**, in roughly two days.

Dead run dirs `e644_..._j4700184` and `e645_..._j4700185` (849 MB each) are
crash artifacts from the pre-fix native-conv failure: 0 steps, 24 fatal errors.
Safe to delete whenever; left in place since /work is only at 21%.

All four slots full (8 V100 + 8 P100). e644/e645 were resubmitted as
4700195/4700196 after the conv fix; the first pair (4700184/4700185) died on the
native-conv failure above. When e644-e646 finish, **Pinpad Five reaches 5/5**.

Move rows into the tracker above when a seed passes 3.9M.

## Next, in priority order

Once a slot frees: Pinpad Five s4 -> Pinpad Six s2, s3, s4 -> Antmaze L s0-s4.
At ~50-70 h per 4M seed on 4xV100 and 4 concurrent slots, the 41 missing 4M
seeds are roughly 3 weeks of continuous V100+P100 occupancy. The 8M and 50M rows
are not reachable this way at all.
