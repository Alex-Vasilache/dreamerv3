# DreamerV4-style upgrades (GRU RSSM, togglable)

Checklist — complete in order:

- [x] **Task 0:** Progress file created (this document).
- [x] **Task 1:** Configuration toggles: `dreamerv3/configs.yaml` under `agent:` — `use_single_rollout`, `use_rms_loss_norm`, `loss_rms_rate`; under `agent.imag_loss` — `use_pmpo_actor`, `pmpo_beta`, `pmpo_alpha` (defaults preserve DreamerV3).
- [x] **Task 2:** Single imagination rollout when `use_single_rollout` is `True` (`K = 1`); multi-branch when `False`.
- [x] **Task 3:** PMPO actor when `use_pmpo_actor` is `True`: weighted **−log π** on **D⁺** and **D⁻** with **the same `pmpo_alpha`** on both, **KL(π_prior ‖ π)** with frozen prior (`policy_behavior_kl_reverse`); **no** DreamerV3 `actent` in the PMPO branch. Standard actor when `False`.
- [x] **Task 4:** Per-term running RMS loss normalization when `use_rms_loss_norm` is `True` (`embodied.jax.RmsTracker`); off when `False`.

## Notes

- With **`use_single_rollout`**, imagination uses **`K_imag = 1`**, but **`repval_loss`** still needs at least two replay timesteps for `lambda_return`, so the repval window uses **`K_repl = min(max(K_cap, 2), T)`** and the imagination bootstrap **`boot`** is broadcast when **`K_repl > K_imag`**. If the replay slice has fewer than two steps, **`repl_loss`** returns zero **`repval`**.
- All behavior defaults to **DreamerV3** when toggles are off.
- RSSM remains **GRU-based** (no transformer).
- Unofficial PyTorch [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) targets tokenizer/dynamics; actor and λ-return logic here follow this JAX codebase.
