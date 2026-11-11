# Handoff 2026-11-11: why TCE does not help, and the next plan (goal holding + surprise)

Start here in the next session. Previous handoff: `docs/HANDOFF_TCE_2026-11-05.md`.

## 0. State at handoff (2026-11-11 ~23:00 JST)

- **Nothing of ours is running.** Cancelled by the user:
  - TCE critic h0314-h0343 (array 4809330): 16 of 30 runs finished.
  - The worker-lr ablation h0344-h0373 (array 4809246).
  - The greedy/sampled eval h0313 (array 4812471).

  Every run dir that existed is archived in the bucket (`SKIP_REPLAY=1`) and deleted from `/work`, along with its SLURM logs.
- **New default: worker lr 1e-4** (`defaults.agent.wkr_opt.lr`; it only takes effect with `wkr_own_opt`, which `wkr_v3` sets).
  - The user chose it as the minimal change from pure Director, whose shared optimizer runs at 1e-4.
  - Every earlier V run (e19xx, e20xx, e95xx, h0001-h0343) used v3's 4e-5. Add the block `wkr_lr4e5` to reproduce those.
  - **The new V base (V at lr 1e-4) has no runs yet.** Its baseline must be run before any new arm is compared to it.
- **Next free experiment number: h0374.** HRL work uses the h series; the robot session uses e. Never take a number from the other series.
- **Investigation files:** scripts and write-ups copied to `experiments/tce_investigation_20261111/` (durable); rollout data in `/work/DoyaU/vasilache/work/meetings/tce_investigation/` (93 MB)
  - `FINDINGS.md`, the short summary;
  - `literature.md`;
  - `analysis.txt`;
  - scripts: `seeds.py`, `fromto_traj.py`, `pinsim.py`, `probe.py`, `probe_kl.py`, `handgoal.py`, `analyze.py`, `reach2.py`, `kl_analyze.py`;
  - rollout data: `probes/`, `hand/`, `kl/`.
- **Meeting page** (claude.ai artifact, private): https://claude.ai/artifact/UrJ1P4yjKWRz11h8DzZd4q. Sources are in `/work/DoyaU/vasilache/work/meetings/HRL-weekly-11-10/`.

## 1. Where the arms stand

5 seeds each, late window 3.5-4M, episode return.

| Task | OG | V | V + US-SOM + TCE sqrt | V + TCE critic (n=3) | DreamerV3 |
|---|---|---|---|---|---|
| Pin pad four | 270 | 311 | 317 | 293 | 382 |
| Pin pad five | 210 | 278 | 260 | 259 | ~200, seeds vary |
| Pin pad six | 70 | 40 | 127 | 135 | 0 (0/5) |
| Cartpole | 691 | 743 | 755 | 686 | — |
| Cheetah | 509 | 536 | 577 | 537 | — |
| Hopper | 335 | 293 | 260 | 283 (n=2) | — |

- Score as % of max: clear 4 tasks (pin pad four/five, cartpole, cheetah) V 70.7, US-SOM 72.3, V + TCE 70.9. All 6 tasks: V 54.1, US-SOM 51.2, V + TCE 58.7. None of these differences is significant.
- Pin pad six is all-or-nothing per seed. Seeds solved, pooled over the finished runs:

| Arm | Solved |
|---|---|
| OG | 2/5 |
| DreamerV3 | 0/5 |
| OG + fixed worker entropy | 3/5 |
| V | 1/5 |
| US-SOM | ~4/13 |
| TCE sqrt | 4/12 |
| TCE-KL | ~5/12 |
| TCE critic | 2/3 |

- Plot: `/work/DoyaU/vasilache/work/collapse_figs/tce/tcecrit_matched.png`, made with `tools/collapse_study/compare_matched.py tcecrit`.

## 2. What the investigation found

Offline only: logged metrics, a numpy simulation of pin pad, and checkpoint rollouts on CPU.

1. **Timing caps the score once a task is solved.**
   - Simulation: an oracle that knows the sequence, with goals held K steps.

     | K | Pin pad four | Pin pad five | Pin pad six |
     |---|---|---|---|
     | 1 | 415 | 353 | 291 |
     | 2 | 392 | 338 | 283 |
     | 4 | 368 | 324 | 262 |
     | 8 | 357 | 313 | 220 |
     | 16 | 296 | 249 | 173 |

   - Solved pin pad six seeds already score 205-220, i.e. the K=8 cap.
   - On pin pad four, DreamerV3 (382) beats the K=8 cap (357), and our worker's imprecision costs a further ~47 (310 vs 357).
2. **Exploration fails because the manager does not pick far pad goals and stick with them.**
   - In normal play, failed seeds aim 3-31% of their goals at pads (solved runs: 60-67%). They press 2-20 pads per 1000 steps (solved: 106-136) and never get past prefix 2.
   - **Hand test (point 4):** with the manager overridden and a fixed goal code for each pad, the worker of a failed seed reached the pad within 48 steps 46-100% of the time (within 8 steps: 25-67%). In normal play the same seeds reach only 5-15% of their pad goals, because the manager re-decides every 8 steps and abandons distant goals (Go-Explore's "derailment").
   - So for 4 of 5 checkpoints, goal space and worker are fine; the manager doesn't commit.
   - Only TCE sqrt s2 shows the goal space shrinking away from the pads: 16% of random codes decode to a pad (pads are 31% of the cells), and encode→decode lands back on the same pad 0% of the time on 4 of 6 pads. This is the problem EDL (Campos 2020) describes.
3. **Representation is fine.** A linear probe on the world-model state recovers the last pad and the sequence prefix with 95-100% accuracy. The 16-bit hash cell predicts the last pad 89-95% of the time.
4. **TCE barely acts.**
   - The lean controller (target 1.0 = "as novel as uniform") drives the TCE weight to 0.05-0.2 by 1-2M on every pin pad six seed. It only rises after a seed has solved.
   - The bonus is about 0.8 almost everywhere, so pads don't stand out.
   - Random goals in the simulation find the first reward in a median 0.5M steps; real agents take 1-2M or never.
5. **Decay is not the limit.** The mean count lifetime is 0.72M steps, and failed seeds barely press pads anyway.
6. **Presses on the way to a goal are mostly useful.** In solved runs, 66-75% of pad presses advance the sequence, 16-24% break it, and the rest leave it unchanged.
7. **World-model surprise marks the key events.** Surprise here is the KL between posterior and prior, i.e. what was observed vs what was predicted.
   - Before solving, pad entries are 1-2σ above ordinary steps and 8-30× over-represented among the top 1% most surprising steps.
   - After solving, the surprise fades: an advancing press scores 0.51 vs 0.78 for a floor step, while a press that breaks the sequence stays high at 1.57.
   - Raw change in the world-model state (RIDE-style impact) rises on pad entry too, but the top 1% of changes are almost never pad entries. Surprise is the better signal.
8. **US-SOM is not needed now.** It is only read where the TO side is the goal code (direct TCE sqrt). V vs US-SOM is not significant.
9. **Novelty of TCE.** The hash version counts world-model hash pairs, not goal codes. That is essentially #Exploration's SimHash (Tang 2017) applied to state pairs one hold apart, used as a manager reward with its own critic. It is a modest contribution and is not a claim to build the paper on.

## 3. Decisions taken by the user

- No per-episode memory (NGU-style), because it would be task-specific.
- The exploration signal is **surprise**. For now it is used **only as a manager reward for goals that lead to surprising real states**. Do not yet do surprise-weighted goal-AE replay (Skew-Fit-like).
- Impact (RIDE) is noted as a possible relevance filter later. It is not used now.
- TCE is paused.
- Base = V with worker lr 1e-4 (see section 0).

## 4. Plan: step by step

### Step 1: calibrate "goal reached" (offline, CPU, no training)

The worker reward is cosine_max(goal, deter) (`hrl/goals.py:_wkr_goal_reward`, `hrl/tensors.goal_reward_cosine_max`). Define reached = cos_max ≥ τ.

- Extend `probe.py` to also record the decoded goal deter: `log/mgr_diag_goal` needs `policy_mgr_diag`, which is read through `getattr` and is **not** a config key, so the flag parse fails. Either add the key to `configs.yaml` (default False) or monkeypatch it in the probe.
- Compare the distribution of cos_max with the player on the goal cell, within one cell, and elsewhere. Do this for 2-3 checkpoints (V, OG, a solved and a failed seed).
- Pick τ with few false "reached". Also check a DMC task (cartpole), where "reached" is less well defined.

### Step 2: answer the open code questions (read code and tests)

- **Discounting across variable-length segments.** Is the manager λ-return discounted per env step (γ^duration) or per decision? Look at the block-pooled path in `agent.py` around `downsample_at_switch_mask` (~L1720-1760) and `hrl/manager.py`. It needs per-step discounting. Use `mgr_reward_agg: sum`, not `mean`, which under-credits long holds.
- **Replay.** Where do the replay-side manager and worker losses (`agent.py` ~L2080-2220, `valid_mgr`) get their switch steps under `variable_goal_length`? Either store the reached-based switch flag in replay, or recompute it from replay deter + goal with the same τ.
- **Worker return.** Check that reaching the goal ends the worker's return for that goal (continuation 0 at the switch). The worker λ-return is already reset at switches.
- Earlier var-K bugs are documented in memory and tests: the block-pooled padded-axis bug, the duration-head REINFORCE noise and the replay trailing-state asymmetry. Re-run `embodied/tests` before and after the change.

### Step 3: implement reached-or-cap goal holding (new config block, e.g. `goal_hold_reach`)

- Reuse `variable_goal_length` with a fixed duration = cap (`goal_duration_fixed: 32`) and no duration head (it is bypassed when fixed).
- Add an early switch: switch_t = countdown expired OR cos_max(goal, deter_t) ≥ τ.
  - Real env: in `_manager_skill_step` (`hrl/manager.py` ~L258), `update = mgr_step <= 0` becomes `| reached`.
  - Imagination: `_imagine_with_manager`, same rule on imagined deter.
- Cap = 32. In the hand test the worker needs a median of 6-15 steps, and often more than 16, so 16 would cut pad trips. Consider 16 as a second arm.
- `imag_length` 16 → 32, so at least one full hold fits; this roughly doubles imagination cost. Batch/replay `batch_length` 64 holds ≥ 2 decisions.
- `worker_timed_goals: False`, since there is no fixed deadline.
- Unit tests:
  - the switch fires on reach and at the cap;
  - manager credit equals fixed-K when τ = ∞ and cap = 8 (regression to Director);
  - the replay switch flags match imagination.
- Smoke on CPU (`debug` last in `--configs`, pin pad six, ~600 steps), then an A100 smoke.
- Log: mean hold length, fraction of holds ending by reach vs cap, worker reach rate.

### Step 4: runs

1. New V base (lr 1e-4), 6 tasks × 3-5 seeds, as the reference.
2. V + goal holding (cap 32), 6 tasks × 3 seeds (maybe cap 16 too).

Watch hopper: earlier variable-length runs hurt it badly, though those used a learned, short duration. Report pin pad as the sustained-50 crossing plus the late mean, never the final last-15 episodes.

### Step 5: surprise reward for the manager (separate arm, after step 4)

The KL exists only on real steps; imagination has no posterior. So, as with Director's reward head:

- Train a small **surprise head** on replay: regress the per-step real KL(post‖prior), with stop-gradient on the target, from the world-model feat. The real KL is already computed in `rssm.RSSM.loss` (`dyn`).
- In imagination, the manager's exploration reward is the head's prediction on the imagined states, with its own critic. It **replaces** Director's reconstruction reward (and TCE) so the total exploration pull stays at one weight (0.1).
- Normalise so the pull can fade. A fixed scale or a slow EMA, not per-channel meanstd, which keeps the share constant forever (the TCE critic problem).
- Offline check first: does the head's prediction rank pad entries high on the saved checkpoints?

Not now (user): surprise-weighted goal-AE replay; impact filter; per-episode memory.

## 5. Useful tools from this session

- `sbatch/run_eval_final.sbatch` + `sbatch/h0313_eval_params.txt`: greedy-vs-sampled eval of 4M milestone checkpoints (`--agent.eval_greedy`). It never ran on GPU. The CPU smoke died silently at step 3390, so re-smoke it before use.
- `tools/collapse_study/compare_matched.py <set>`: matched-window curves + table.
  - Sets include `tcecrit`, `ogustce_all`, `wkrstory`, `ogdv3v`.
  - `FINISHED_ONLY` restricts a set to runs ≥ 3.98M.
  - The `MARKERS` dict selects solid lines + markers.
- `/work/DoyaU/vasilache/work/meetings/tce_investigation/probe.py`: CPU rollouts of a checkpoint, logging position, the pad sequence, the decoded goal image and deter.
  - `handgoal.py` overrides the manager (monkeypatched `_emit_manager` that re-emits the carried code), so a goal code can be fixed by hand.
  - `probe_kl.py` adds the per-step world-model KL (monkeypatched `RSSM.observe` / `Agent.policy`).
  - Run them under `source_dreamerv3_env.sh` on the login node: about 3 min to compile, about 1 min per 2000 steps.
- `pinsim.py`: pin pad grid sim (oracle with K-step holds; random/episodic goal policies).

## 6. House rules and peculiarities (from this and earlier sessions)

**Communication**
- Short replies, 1-3 sentences by default. No headers or bullets unless a comparison is asked for. Plain language, no jargon.
- When asked for a table, give a narrow plain markdown table in chat with short arm names: "V", "V + TCE", "V + US-SOM", "OG", "DreamerV3".
- Status, plots and tables go in chat with the PNG path. No claude.ai artifacts unless asked. Answer first, housekeeping after.
- Any claude.ai artifact: white background, light theme only.
- The user reviews claims critically ("are we sure?"), so state n, p and caveats. Never quote pin pad final last-15 alone.

**Plotting** (memory `figure-style.md`)
- Panels on 2 rows (6 tasks = 2×3, figsize 13×8.2). Fonts: titles 17, axis labels 15, ticks 13, legend 14. lw 2.4; ±1 std bands at alpha 0.13-0.15.
- Shared legend centred below. 200 dpi PNG + PDF. Python: `/work/DoyaU/vasilache/work/plotenv/bin/python`.
- Okabe-Ito colours: OG #E69F00, V #56B4E9, US-SOM #000000, TCE #009E73, others #0072B2 / #D55E00 / #CC79A7. Never pair green with purple.
- **Choose the legend style for every plot.** Run the deuteranopia check (`cvd.py`: Machado 2009, OKLab ×100) on that figure's exact colours.
  - If every pair is ≥ 15, use solid lines + distinct markers (o, s, ^, D; markevery 5).
  - Only use dash patterns if a pair is < 15 or there are > 4 arms.
  - `handlelength` ≥ 3.5. State the minimum pair distance when reporting.
  - The user complained when a 3-arm plot was all dashed.
- Goal-code blocks are always drawn **vertically**: one row per block, classes left to right.
- Plots and tables must use the same seeds (use `FINISHED_ONLY` when pooling extra seeds).
- TCE comparisons always include V + US-SOM (no TCE) next to OG.

**Cluster and repo**
- **Never touch the robot session's jobs or files:** e1910+ robot names, e2xxx, e30xxx, e5030, e6031, e8xxx simrobot, e9660-9671, robot_*, sim_videos*, sysid, emotion, realcmd, spr_meshes, robot_endpoints, robot_policies_r6, linktest_20261102. The V100 jobs are the robot's. The robot session holds all 8 V100 and 8 P100 slots, so HRL runs are A100 only (`gpu-a100`, 8 GPUs, 128 CPUs).
- `short-a100` is low priority and tends to sit pending. `intel` preempts instantly.
- `squeue` hides other users; read real free capacity from `scontrol` AllocTRES.
- **Never write into `apps/clipboard/`**: it's the user's personal scp folder. Meeting material goes to `/work/DoyaU/vasilache/work/meetings/`.
- Archive only after a terminal `sacct` state: `SKIP_REPLAY=1 copy_dreamerv3_runs_to_bucket.sh <dir>`, verify, then delete the run dir and its SLURM logs.
- Log every launch in `EXPERIMENTS.md`, with the outcome later. Commit and push periodically. Never force-push.
- Update the paper (`code/26_04_HRL-paper`) when results materially change. `code/dreamerv3/paper/` is stale; don't touch it.
- **Gotcha:** `pkill -f <pattern>` / `pgrep -f` in the same shell also matches your own command line and kills the tool shell (exit 144). Kill by PID from `ps -eo pid,args | grep "[p]attern"`.
- **Gotcha:** importing a helper script without an `if __name__ == '__main__'` guard reruns its whole analysis (that cost 10 min).
- **Gotcha:** `elements.Flags` rejects keys that aren't in `configs.yaml`; `getattr(config, ...)`-only flags such as `policy_mgr_diag` can't be set from the CLI.
- Background `nohup` loops survive a killed tool shell, so check for duplicates before relaunching.
