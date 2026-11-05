# Handoff: TCE exploration experiments (2026-11-05, 21:30 JST)

Read this first, then `EXPERIMENTS.md` (top entries) and the memory index. Keep replies
short and plain (CLAUDE.md). Answer status questions in chat with a table and the PNG
paths; do not build web-page artifacts unless asked; answer first, archive afterwards.

## What is running

| Array | Runs | What | Status at handoff |
|---|---|---|---|
| 4795008 | e19123-e19129 | TCE, 4x12-bit WM hash, adaptive (lean) weight; seed 3 hopper + all of seed 4 | 7 running, finish overnight / Fri morning |
| 4798833 | h0073-h0102 | **TCE-KL, 1x16 hash** (`tce_kl`) | queued, 6 tasks x 5 seeds, `%4` |
| 4798834 | h0103-h0132 | **TCE-KL, 4x12 hash** (`tce_kl4`) | queued, 6 tasks x 5 seeds, `%4` |

Both TCE-KL arms use `director director_og wkr_ent_fixed wkr_v3 goal_som_line goal_even
fromto_dir {tce_kl|tce_kl4} pubmatch` via `sbatch/run_benchmark.sbatch` with params files
`sbatch/pubtcekl_params.txt`, `sbatch/pubtcekl4_params.txt` (A100, RUN_ENVS=4,
DMC_PROPRIO=False). Expected: seed 0 of both arms done ~Fri evening, all done ~Sun midday
if the A100 partition stays free (on 3 Nov other users filled it and we got 1 GPU for 20 h).

**No extra seeds.** The user does not want extra pin pad six seeds (or extra seeds in general
unless asked). Every batch is 6 tasks x 5 seeds.

## What TCE-KL is (the current mechanism)

TCE = transition-count exploration. At every real env step the host counts (start cell ->
goal cell) for states 1..8 steps back and for each manager proposal; counts decay with a
0.5M-step half-life. Start cell = SimHash of the per-state standardised world-model deter
(1x16 bits, or 4 independent 12-bit hashes whose bonus is averaged); goal cell = goal code
binned to 2 bins per block (US-SOM codebook). In the manager loss:

    entropy + TCE = alpha * H(pi) - alpha' * E_pi[log(n+1)]  =  -alpha * KL(pi || q),  q(g|s) ~ 1/(n(s,g)+1)

alpha comes from Director's manager entropy controller (target 50%); alpha' = alpha divided
by the per-block entropy range (log 8), so units match. No separate TCE weight, no lean
controller, band or floor. With equal/zero counts q is uniform and this is plain Director.
Code: `dreamerv3/hrl/fromto.py` (`expected_novelty(..., bonus='log')`, `FromToCounter`),
`dreamerv3/agent.py` (search `fbonus`, `_start_hash`), `embodied/run/train.py` (counter hook),
config blocks `tce_kl`, `tce_kl4` in `dreamerv3/configs.yaml`. Tests:
`embodied/tests/test_fromto.py` (15 pass; run with `JAX_PLATFORMS=cpu`).

**Main risk to watch:** alpha is small (~0.002 in the smoke test), so the push may be too weak.
Check `train/fromto/weight` (should be ~alpha/2.08), `train/fromto/gap` (E_pi - E_uniform of
the log bonus; > 0 means the manager prefers rarely tried goals) and whether pin pad five
finds its first reward by ~0.8M and pin pad six solves at all.

## Results so far (late 3-4M mean, % of max; pin pad six seeds solved)

| Arm | Overall | vs TCE | Pin pad six |
|---|---|---|---|
| OG Director | 50% | -7.9 (p=0.03) | 2/5 |
| V + US-SOM (no TCE) | 51% | -7.1 | 0/5 |
| **TCE**, codebook start, 0.5M half-life, adaptive w (e6200) | 58% | - | 4/5 |
| TCE-anneal, no decay, w=2 (e7000) | 52% | -6.5 | 2/4 |
| TCE-grow, h=t/2, w=2 (e8000) | 51% | -7.3 | 1/5 |
| TCE on plain SOM (e8600) | 50% | -8.3 (p=0.01) | 1/5 |
| TCE, expl reward off, w=2 (e9630) | 40% | -18.7 (p<0.01) | 0/3 |
| **TCE, 1x16 WM hash**, adaptive w (e9500) | **59%** | +0.4 (p=0.95) | 3/5 |
| TCE, 4x12 hash, w=2 (e10000, stopped) | 48% | -14.7 | 0/2 |
| TCE, 4x12 hash, adaptive w (e19100, finishing) | ~51% (n=4) | -6.9 | 2/4 |
| 4x12 hash, smoothed lean + floor 0.3 (h0001-h0072, cancelled after 2 seeds) | ~52% | - | 0/3 |

Lessons: 0.5M half-life is the forgetting rate that keeps pin pad six; pin pad six fails when
lean climbs well above ~1.04 (too much push) or the TCE weight collapses before the first
reward (too little); US-SOM beats plain SOM with TCE; Director's exploration reward is still
needed; the WM hash start is at least as good as the codebook. The per-batch lean controller
(AutoAdapt default band +-10%) ratcheted w down on hash runs (only lean > 1.1 batches crossed
the band); the floor fixed pin pad five but over-pushed pin pad six. Those problems are why
TCE-KL has no separate weight at all.

Full table: `/work/DoyaU/vasilache/work/collapse_figs/tce_weekend_summary.md` (regenerate:
`/work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/tce_weekend_summary.py`).
Report page with every step: https://claude.ai/artifact/AvCTQpGvqwscfeFbF7z3uw (do not
republish unless the user asks).

## Monitoring routine (every ~4 h; set it up with `/loop supervize the tasks every 4h`)

1. `squeue -u $USER` and `sacct -j <array> -X -o JobID,State`; grep
   `/work/DoyaU/vasilache/work/slurm_logs/*<array>*.err` for `Traceback`.
2. Per-run progress: last score and `train/fromto/{weight,gap,novelty}` from
   `<run>/logdir/metrics.jsonl` and `scores.jsonl`.
3. Archive finished runs: only when `sacct` says COMPLETED/CANCELLED/FAILED, then
   `SKIP_REPLAY=1 /work/DoyaU/vasilache/work/copy_dreamerv3_runs_to_bucket.sh <run_dir>`,
   verify `logdir/scores.jsonl` and `slurm_logs/` in
   `/bucket/DoyaU/vasilache/bucket/results/dreamerv3/<run_dir>`, then delete the run dir and its
   `slurm_logs/*<jobid>_<task>.{out,err}`. Never touch a running job's dir.
4. Compare: `tce_weekend_summary.py` (table, includes the two TCE-KL arms) and
   `tools/collapse_study/compare_matched.py <set>` (curves; add a `tcekl` set with OG, TCE
   e6200, 1x16 hash e9500, and the two h-arms: patterns
   `h00[7-9]?_{t}_...-tce_kl-pubmatch_s*` / `h010?..`, `h01[0-3]?_{t}_...-tce_kl4-pubmatch_s*`).
   Diagnostics: `tce_lean_weight_plot.py`, `pp5_failure_diag.py`, `tce_vs_anneal_diag.py`.
   Figures go to `/work/DoyaU/vasilache/work/collapse_figs/`.
5. Log outcomes in `EXPERIMENTS.md` (top entry h0073-h0132), commit and push
   (`code/dreamerv3`, branch main, never force-push). Ask before changing anything the user
   did not request.

## Conventions that bit us

- **Experiment numbers:** HRL runs use the `h{NNNN}` series (next free: h0133). The robot
  session uses `e{N}` and kept colliding with my e-ranges. Never use `e` numbers again.
- **A100 only** (8 GPUs, `-p gpu-a100 --gres=gpu:a100:1 -c 16 --mem=128G`). V100 OOMs with the
  stock env and gives NaNs with the Route-A env.
- **Archive deletion** only after a confirmed Slurm terminal state (a failed scancel once
  deleted live run dirs).
- `pkill -f <pattern>` kills your own shell if the pattern appears in the command line; use
  the PID.
- Figures: 2 rows, large fonts, Okabe-Ito colours plus line styles (colour-blind safe).
- Names: US-SOM = SOM-line + evenness loss; TCE = transition-count exploration; OG = base
  Director; V = OG + fixed worker entropy + worker v3.

## Open questions for the user

1. Is TCE-KL's push strong enough with alpha ~0.002? If not, the fallback is one fixed
   multiplier on the TCE part (same for every task), still without a controller.
2. 1x16 vs 4x12 hash under TCE-KL (this batch answers it).
3. Later: why fixed worker entropy alone worked so well on pin pads (memory note
   "revisit fixed worker entropy").
