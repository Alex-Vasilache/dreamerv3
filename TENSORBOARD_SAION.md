# TensorBoard (Saion login node)

Run on the **login node**, not inside Slurm. Use `rl_env` so `tensorboard` matches the stack used for training.

## Single run (example v4 run)

```bash
source "/apps/unit/DoyaU/vasilache/rl_env/bin/activate" && tensorboard \
  --logdir "/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_v4_20260513_134842_d000kC/" \
  --port 6006 --host 0.0.0.0 --load_fast=false
```

Then open `http://127.0.0.1:6006/` (or SSH port-forward that port).

`--logdir` may be the run directory (as above) or `.../<RUN_ID>/logdir/`; TensorBoard recurses under the path you give.

## Wrapper scripts (same flags)

- One run, defaults: `./tensorboard_login.sh` (override `LOGDIR`, `PORT`, `BACKGROUND`).
- **All baseline (v3 defaults) + v4 + single-flag A/Bs** under `dreamerv3_online_vis1m32_1env_*` (sorted newest-first, every run that has a `logdir/`):

  ```bash
  cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
  ./tensorboard_all_last_runs.sh
  ```

  Newest run **per** variant only (one baseline, one `ab_pmpo`, one `ab_rms`, one `ab_rollout`, one `v4`):

  ```bash
  TB_ONE_PER_VARIANT=1 ./tensorboard_all_last_runs.sh
  ```

  Background: `PORT=6007 BACKGROUND=1 ./tensorboard_all_last_runs.sh`

## Common mistake

Use **`--load_fast=false`** exactly once. Avoid pasted typos like `load_fast=falset` or duplicated `--port` / `--host`.
