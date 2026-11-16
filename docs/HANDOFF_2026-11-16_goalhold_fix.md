# Handoff 2026-11-16: goal hold works after two fixes; surprise helps; runs still live

Start here. Previous: `docs/HANDOFF_2026-11-13_goalhold_results.md` (its negative result is
VOID, see section 1). Full log: `EXPERIMENTS.md`, sections h0410-h0452. Progress page (private
artifact): https://claude.ai/artifact/Tms5v6DyAqd1y9bAij36QZ, rebuilt with
`plotenv python tools/goalhold/plot_progress.py && python3 tools/goalhold/build_page.py <out.html>`.
Paper: `code/26_04_HRL-paper/sections/motivation.tex`, subsection "Holding a goal until it is
reached" (figure script `figures/motivation/make_goal_hold.py`).

## 0. State at handoff

See the table at the end of this file (section 6) for live jobs; it is updated last.

- **Next free experiment number: h0459.**
- Autonomy given by the user ended Monday 2026-11-16 20:00 JST. Live jobs were left running.
- The supervision cron was session-only. Recreate with `tools/goalhold/supervise.sh <array id>`
  (status) and `tools/goalhold/archive_done.sh <array id>` (archives terminal runs; skips run dirs
  that contain a `.keep` file).
- A development clone lives at `/work/DoyaU/vasilache/work/dev_dreamerv3` (same remote). Live jobs
  import from the main tree when they start or requeue, so edit and test in the clone, push, then
  `git pull --ff-only` in the main tree. Delete the clone when no longer wanted.

## 1. Two errors in the variable-length path (both fixed, both tested)

1. **Bootstrap of the horizon-cut last hold** (`goal_duration_trailing_boot`, default True;
   commits 5e745f3 + 2069ea8). In imagination the last hold of a rollout bootstrapped its manager
   return from its own start state, V(s) <- R + g^h V(s). Harmless with fixed holds dividing the
   horizon and with dense rewards; on pin pads the manager return range was 30-120 instead of 9.
   **Every variable-length run before 2026-11-14 23:07 carried it** (h0374-h0409 and all the old
   duration-head runs). The complete fix sets EVERY padding slot of the packed decision buffer to
   the final imagined state (`hrl/tensors.fill_trailing_final_state`); patching only the first
   slot leaves 0.95 of the error.
2. **Counterfactual switch points on replay** (`goal_replay_real_switch`, block
   `goal_hold_realsw`, default False; commit e588598). The replay-side manager returns pooled
   rewards over switch points the current manager would make on old data, where "reached" rarely
   fires: replay holds 22-29 steps vs 10-11 real at cap 32. The option stores the real switch flag
   per step in replay (`mgr_sw`). Effect: removes the late start at cap 32 on pin pad five (first
   score 0.26M vs 1.93M); final level unchanged; no effect at cap 8; dips seen in several
   real-switch runs (not yet shown to be more frequent than without).

Tests: `embodied/tests/test_vargoal_path_stepwise.py` (real agent, every stage recomputed by hand:
env switch rule, imagination, manager and worker tensors, bootstrap state, replay, the return
number itself, the other switch rules, real switch points), `test_mgr_surprise.py`. The older
var-K suites are repaired (a test stub returned a Python float). Probing tips: critics are zero at
init, so identify states by policy logits; `nj.grad` traces the loss twice, so `jax.debug.callback`
captures appear twice (use the last); never stash traced values on `self` across the grad.

Still open, shared with fixed-hold V (not variable-length bugs): the replay manager lambda-return
uses one final-state bootstrap for every decision instead of a per-decision one; the replay worker
path evaluates the current manager's goals on old trajectories; 5 ring-head manager tests fail on
an unrelated argument (`nu`).

## 2. What was built (all behind config blocks, layered on V = `director director_og wkr_ent_fixed wkr_v3 ... pubmatch`)

| Block | What |
|---|---|
| `goal_hold_reach` (+ `goal_hold_cap8` / `cap16` / `cap4` / `cap48`) | hold until reached (cos_max >= 0.8 x ceiling) or cap (32 default) |
| `goal_hold_frac07` / `frac09` | threshold 0.7 / 0.9 |
| `goal_hold_done` | worker "done": (1-g) V_wkr(s,g) <= cos_max(g,s), floor 0.5 x ceiling (`goal_reach_mode: done`) |
| `goal_hold_mse` / `mse4` | reached by squared error <= 2x / 4x running self-reconstruction error |
| `goal_hold_noreach` | reached test off (fixed holds through the same code) |
| `goal_hold_realsw` | real switch points on replay |
| `goal_hold_cost` | price per manager decision (untested in runs) |
| `goal_dur_head` (+ `goal_dur_head8`) | manager picks the length 1..32 (1..8), same SMDP training path, countdown shown to the worker; use INSTEAD of `goal_hold_reach` |
| `mgr_surprise` (+ `mgr_surprise_k2`) | surprise reward: head regresses max(0, KL - mean - k std) on replay, paid in imagination via the third critic; replaces TCE, reconstruction reward stays |
| `steps6m` | run to 6M (put after `pubmatch`) |

New metrics: `episode/log/hold_switch/avg` (real switch rate; 1 / it = real hold),
`surprise/*`, `goal/reach_self_mse`. `goal/hold_len_repl` is counterfactual unless
`goal_hold_realsw` is on.

## 3. Results (late 3.5-4M; first trailing-10 >= 50 in M steps)

Pin pad five (old V, 5 seeds: 278 on 4 of 5; first 1.11-2.59):

| Arm | Real hold | Late | First |
|---|---|---|---|
| reach or 8 | 6-7 | 277, 293 | 0.30, 0.31 |
| reach or 8 + surprise (k=0) | 7 | 295, 296 | 0.22, 0.34 |
| reach or 8 + surprise (k=2) | 7 | 304 | 0.76 |
| reach or 8, squared error x2 / x4 | - | 293 / 298 | 0.74 / 0.98 |
| reach or 8 + real switches | - | 292 | 0.27 |
| worker done, cap 32 | 3-6 | 293, 294 | 0.36, 0.47 |
| worker done + real switches, cap 32 | 4-5 | 298 | 0.26 |
| worker done + surprise, cap 32 | - | 262 | 0.26 |
| reach or 16 | ~9 | 258 | 0.46 |
| reach 0.7, cap 32 | 7-8 | 210 | 0.81 |
| reach or 32 | ~10 | 201 | 1.93 |
| reach or 32 + real switches | ~10 | 209, 214 | 0.26, 0.83 |
| reach 0.9, cap 32 | 20-22 | 0 (stopped 2.2M) | - |
| manager picks 1..32 | 11-18 | 0 (stopped 2.6M) | - |
| manager picks 1..8 | 4-5 | 1 | 1.86, then lost |
| fixed 4, no reached test (control) | 4 | 0 (stopped 3.0M) | - |

Pin pad four (old V 311): reach or 8 332 / 350; reach or 32 284 / 303; + real switches 271;
worker done 287. Before the bootstrap fix: 181 (cap 8), 237 (cap 32).
Cheetah: reach or 8 616 (old V 536); reach or 32 + real switches ~330 at 3.9M (h0449).
Pin pad six: 0 in 7 runs (cap 8, cap 32, worker done, surprise k=0 x2, surprise k=2 seed 0,
cap 32 + real switches); **surprise k=2 seed 1 (h0439) first scores at 3.63M, ~225 through 5.7M**.

Reading:
- Ending a hold when the goal is reached is what helps, not a shorter hold: fixed 4 is worse than
  fixed 8.
- Shorter realised holds are better; the final level follows the real hold (timing cap).
- Cap 32 is fine when holds end early for another reason: worker done at cap 32 matches the best.
  For "cap 32" use `goal_hold_reach goal_hold_done` (optionally + `goal_hold_realsw`).
- Manager-chosen length fails at 1..32 (entropy regulariser keeps durations near 15, nothing
  favours short holds before the first reward) and at 1..8 (holds of 4-5 steps, scored once at
  1.86M and lost it). Neither ran with real switch points; there the replay bias goes the other
  way (replay holds SHORTER than real: ~2 vs 4-5 and ~12 vs 15-18), and the user suspects that is
  what breaks it. h0454 / h0455 test this.
- Surprise adds ~10-20 points on pin pad five; one pin pad six seed solved with k=2. All n <= 2.

## 4. Suggested next steps

0. **Read h0454 / h0455 first** (duration head with real switch points; the user's open question).
1. Read the live runs (section 6). Pin pad six seeds h0450-h0452 decide whether surprise k=2 is a
   real effect (need >= 3 of 5 against V's 1 of 5 to say anything).
2. More seeds of the three best arms on all six tasks: reach or 8; reach or 8 + surprise k=2;
   worker done (cap 32) + real switches. Hopper and cartpole are untested with the fixes.
3. Decide on real switch points: count dips with and without it at matched seeds.
4. Duration head: if 1..8 learns, add a prior toward short holds instead of the range limit.
5. The two replay imprecisions shared with V (section 1) as a separate arm against V.
6. Update the paper table with the final numbers of the live runs; rebuild the figure.

## 5. House rules that mattered this weekend

- A100 only (cap 8 GPUs / 128 CPUs); the partition is often full and can preempt. Shorter time
  limits (10 h, self-requeue) get backfilled much sooner than 16 h.
- Cancelling a run to free a slot only helps when we are at our own cap (pending reason
  `AssocGrpCpuLimit`); otherwise other users take the GPU.
- Never touch the robot session's jobs (non-`goalhold` names, `e` numbers).
- Pin pad: sustained-50 crossing plus late-window mean, never the last-15 alone. Runs dip to ~0
  for 0.1-0.3M and recover; a dip is not a failure.

## 6. Live jobs at handoff

State at 2026-11-16 18:40 JST. Check with `tools/goalhold/supervise.sh <array id>`; archive with
`tools/goalhold/archive_done.sh <array id>` once a task is COMPLETED.

| Run | Array_task | What | State |
|---|---|---|---|
| h0450 / h0451 / h0452 | 4824792_0 / _1 / _2 | pin pad six, reach or 8 + surprise k=2, seeds 2 / 3 / 4, to 6M | running: 4.25M / 4.06M / 3.39M, all 0 (h0439 first scored at 3.63M) |
| h0445 | 4824246_1 | pin pad six, reach or 8 + TCE critic, seed 1 | running: 3.10M, 0 (seed 0, h0444: 0 at 4M, archived) |
| h0449 | 4824385_3 | cheetah, reach or 32 + real switches | PENDING at 3.93M (requeued after an interruption; resumes from its checkpoint and finishes in minutes); 330 over 3.5-3.9M |
| h0453 | 4824840_0 | pin pad five, worker done + real switches, cap 32, seed 1 | pending, estimated start 11-17 18:19 |
| h0454 / h0455 | 4827526_0 / _1 | pin pad five, duration head 1..8 / 1..32 + real switches | pending, estimated start 11-18 23:08 |
| h0456-h0458 | 4827569_0-2 | pin pad five real-switch panel: reach 0.7 cap 8 / cap 32, done cap 8 | pending (Priority) |

Everything else of h0410-h0452 is finished or stopped and archived in the bucket; nothing else of
ours is in `/work` except the dev clone and `/work/DoyaU/vasilache/work/meetings/goalhold/`
(figures). Arrays use a 10 h limit with self-requeue; h0450-h0452 will requeue once to reach 6M.

To start the next session, paste:

> Continue the HRL work in code/dreamerv3. Read docs/HANDOFF_2026-11-16_goalhold_fix.md in full,
> then check the live arrays (4824792, 4824246, 4824385, 4824840, 4827526, 4827569) with
> tools/goalhold/supervise.sh and archive what is finished.
