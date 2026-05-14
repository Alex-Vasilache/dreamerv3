# Director-style HRL on DreamerV3 (JAX) — plan and progress

Reference repos (Task 1):

- **DreamerV2 (upstream baseline):** shallow-cloned at `/apps/unit/DoyaU/vasilache/apps/code/dreamerv2` for reference (`git clone --depth 1 https://github.com/danijar/dreamerv2`).
- **Director (TensorFlow, DreamerV2-era stack):** `../director` — `embodied/agents/director/{agent.py,hierarchy.py,nets.py,configs.yaml}`.

## Task 1 — Architectural / algorithmic diff (Director vs flat Dreamer)

| Area | DreamerV2-style world model | Director additions (`hierarchy.py`) |
|------|-----------------------------|-------------------------------------|
| **State for control** | RSSM latent → flat actor | Same RSSM; **worker** conditions on **goal** (decoded vector in **deter space**); **manager** outputs **discrete skills** every **K** steps. |
| **Goal autoencoder** | — | `enc` maps **(goal, context)** → categorical **skill**; `dec` maps **(skill, context)** → **goal** vector; trained on replay (and optionally imagination) with **reconstruction + KL** to prior (`train_vae_replay`, `train_vae_imag`). |
| **Manager** | — | `ImagActorCritic` in **skill space**; rewards mix **extrinsic** (WM reward head), **exploration** (`elbo_reward` / `adver` MSE on `dec.mode()-feat`), **goal channel** (often 0 weight). Steps every `train_skill_duration` / `env_skill_duration`. |
| **Worker** | Standard actor | `ImagActorCritic` with **inputs** `[deter, stoch, goal]` (plus `delta = goal - feat`); optimizes **goal reward** (`goal_reward` in config; default `cosine_max`). |
| **Goal reward (`cosine_max`)** | — | `norm = max(||g||,||s||)` per vector, then dot-product of **g/norm** and **s/norm** (see `goal_reward` branch `cosine_max` in `hierarchy.py`). |
| **Imagination training** | `imagine(actor)` | **`imagine_carry`**: joint rollout (`train_jointly`) so **manager + worker** share one imagined trajectory; `split_traj` / `abstract_traj` reshape for worker vs manager updates. |

DreamerV3 (`dreamerv3/agent.py`) differs from DreamerV2 (RSSM API, `imag_loss`, PMPO / single-rollout options). The JAX port **aligns training routing** with Director (`split_traj` / `abstract_traj`, `worker_rews` / `manager_rews`, `expl_rew` adver vs disag) while still using **one shared critic** (`val` / `slowval`) and **one** `retnorm` / `advnorm` across worker and manager imagination steps (Director uses separate `ImagActorCritic` stacks and per-stream norms).

---

## Checklist (Tasks 3–7)

- [x] **Task 3:** `configs.yaml` — `use_director_hrl: False` default; nested `director_hrl:` hyperparameters (`manager_step_K`, `train_skill_duration`, `env_skill_duration`, `jointly`, `goal_reward`, `worker_rews`, `manager_rews`, `expl_rew`, `adver_impl`, `disag_*`, `vae_replay`, loss scales including `director_disag`).
- [x] **Task 4:** Goal autoencoder on **replay** deter trajectories (ELBO-style: MSE reconstruction + categorical KL toward uniform), gated by `vae_replay`.
- [x] **Task 5:** **Manager** `imag_loss` on **`abstract_traj`** with **weighted** `reward_extr` / `reward_expl` / `reward_goal` (exploration = `elbo_adver_reward` or `disag_reward` on full imagined deter).
- [x] **Task 6:** **Worker** `imag_loss` on **`split_traj`** with the same three reward channels and **configurable** `goal_reward` string (`director_rewards.goal_reward`, many branches from `hierarchy.py`).
- [x] **Task 7:** Hierarchical **`nj.scan`** imagination when enabled; **flat** path unchanged; assert `(imag_length + 1) % train_skill_duration == 1` (default `imag_length: 16`, `K=8` → 17 steps).

### Still not a full TF port (differences from `hierarchy.py`)

- [ ] **`jointly`:** only **`new`** is implemented (`train_jointly`); `old` / `off` raise or are absent.
- [ ] **Separate critic per reward stream** (Director has three value nets per actor with per-stream `retnorm` / `scorenorm`); here **one** `val` and **combined** `worker_rews` / `manager_rews` scalars on pre-split rewards.
- [ ] **`explorer`**, **`vae_imag`**, **`explorer_repeat`**, **`train_vae_imag`**, **`abstract_traj`/`split_traj` parity** on all edge keys (`delta`, decoder `context` branches, `vae_span`).
- [ ] **Goal decoder / encoder** `inputs` dict routing like `nets.Input` (only concat order used here).
- [ ] **`goal_reward`** branches that need the encoder (`enclogprob`, …) are left as `NotImplementedError` in `director_rewards.py`.

---

## File map (implementation)

| File | Role |
|------|------|
| `dreamerv3/director_hrl.py` | AE replay loss, exploration MSE, `elbo_adver_reward`, `disag_reward`, `disag_replay_loss`, `reinforce_manager`, carry helpers. |
| `dreamerv3/director_rewards.py` | `goal_reward` branches (ported from `Hierarchy.goal_reward`), `broadcast_context`. |
| `dreamerv3/director_trajectories.py` | `split_traj`, `abstract_traj` (batch-major; dict **action** supported in `split_traj`). |
| `dreamerv3/director_full.py` | `build_imag_traj_for_hierarchy`, `split_and_abstract`, `feat2tensor`. |
| `dreamerv3/agent.py` | Conditional modules, hierarchical scan, **split + abstract** AC losses, `env_skill_duration` in `policy()`. |
| `dreamerv3/configs.yaml` | Defaults; `imag_length: 16` for `K=8`; `director_disag` in top-level `loss_scales`. |
