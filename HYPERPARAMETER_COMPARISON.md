# Hyperparameter comparison table (2026-07-15 snapshot)

Split out of `EXPERIMENTS.md` §2 (2026-07-23) to keep that file condensed per its own
stated convention (§1 "Style"). This is a single dense cross-cell table from the
2026-07-15 live campaign + controls, covering e46/e124 through e249 — kept verbatim for
reference. It predates the differentiable-reuse campaign (e250 onward, F19/F20) and the
single-head board's later declines noted in `EXPERIMENTS.md` §2/§3; read it as a
historical snapshot, not current best-configs. Current state, live findings, and the
experiment ledger all live in `EXPERIMENTS.md`.

---

### Interim hyperparameter comparison table (2026-07-15, live campaign + controls)

All current-campaign rows (no `†`) are **still running** (8–89% of a 4M-step budget) —
read as a snapshot, not a landed result. `†` rows are finished/archived comparators pulled
in for contrast; `‡` rows use the **single-head (joint) masked manager**
(`agent.mask_joint_edit`, EXPERIMENTS.md §2/§3 F17) — a one-categorical-per-block design
that replaced the dual-head skill+mask design used by every other row in this table.
`works?`: ✅ alive/on-reference, ❌ dead or collapsed, ~ mixed/partial/still resolving.
`score` = last-15 (peak-15) episode return. `blk/step` = realized `mask_frac×8/duration`
(masking off ⇒ whole code re-written every switch, so it reduces to `8/duration`;
Director's own fixed-K8 baseline = 1.00; for `‡` rows K is always fixed at 8, so blk/step
reduces to `mask_frac` directly). `goal length` = `fix 8` (Director's fixed switch
interval) or `var τ{4,8} (reg|lagr)` (soft fixed-prior vs. duration-Lagrangian control
mode). Struct `+adapt` = `goal_struct_adapt` Lagrangian on top of the listed init weight;
`(dual, targetX)` = two-sided dual-ascent toward raw struct-loss setpoint `X`. `goal mask`
for `‡` rows: `joint free` = no sparsity penalty (`mask_sparsity_mode=none`, edit fraction
shaped only by task-return REINFORCE); `joint ratchet→X` = `mask_sparsity_mode=prob` with
the target ratcheted from 1.0 down to `X` over ~1M env-steps (§sec:ratchet). `§` rows
(e230–e249) are a **third, distinct design** from either `†`/plain or `‡`/single-head:
plain pure Director (no mask of any kind, whole goal code always redrawn) with
`mgr_cond_goalcode=True` so the manager can *choose* to reproduce blocks, plus the
**implicit-sparsity controller** (EXPERIMENTS.md §2, F18) — a REINFORCE cost on
`1 − kept_frac` where `kept_frac` is *measured*, not masked. `blk/step` is n/a for `§`
rows (there is no edit mask; `goal mask` column instead reports the controller config and
`kept` = `goal/implicit_sparsity_block` at the final checkpoint). `impl struct-only` = no
controller, struct-adapt only; `impl ratchet→0.7kept` = controller target ramped 0→0.7 kept
over training, one-sided; `impl direct 0.7kept` = controller target fixed at 0.7 kept from
step 0; `impl struct+ratchet→0.7kept` = both. All 16 `§` rows completed the full 1M-step
budget (not interim). **Correction (2026-07-15):** e226/e227 — read as
"H confirmed"/decisively alive in the 07-14 interim readout at 65% — have since **declined
substantially** in large-window trend (e226: peak ~290 @1.9M → ~75 in the last ~270-episode
window @3.2M; e227: peak ~150 @1–2.4M → ~24 @3.2M), while internal signals
(`mgr_extr_adv`, `wkr_goal_rew`) show no classic collapse signature (advantage stays small,
worker reward is still rising) — this does not match F11's collapse anatomy and is
unexplained; treat e226 as "was alive, now declining" rather than a settled rescue until
investigated. Cheetah `‡` cells (e222/e223) show no equivalent decline over the same
window.

| exp | env | size | works? | score | blk/step | struct | goal mask | goal length | reward agg | wkr countdown | mask ratchet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e46† | cartpole | small | ✅ | 724 (755) | 1.00 | 200 | off | var τ8 (reg) | mean | no | no |
| e124† | hopper | BIG | ✅ | 196 (326) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e162† | cartpole | small | ~ | 270 (565) | 0.31 | 200 | prob | fix 8 | mean | no | no |
| e163† | cartpole | BIG | ~ | 364 (412) | 0.31 | 200 | prob | fix 8 | mean | no | no |
| e164† | hopper | small | ❌ | 0.0 (3.7) | 0.30 | 200 | prob | fix 8 | mean | no | no |
| e165† | hopper | BIG | ❌ | 0.3 (3.9) | 0.31 | 200 | prob | fix 8 | mean | no | no |
| e166† | cartpole | small | ❌ | 129 (185) | 2.00 | 0 | off | var τ4 (reg) | mean | no | no |
| e167† | cartpole | BIG | ❌ | 98 (650) | 2.00 | 0 | off | var τ4 (reg) | mean | no | no |
| e168† | hopper | small | ❌ | 0.3 (3.1) | 2.02 | 0 | off | var τ4 (reg) | mean | no | no |
| e169† | hopper | BIG | ❌ | 0.0 (2.4) | 2.01 | 0 | off | var τ4 (reg) | mean | no | no |
| e170† | cartpole | small | ✅ | 653 (785) | 0.61 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e171† | cartpole | BIG | ✅ | 776 (843) | 1.13 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e172† | hopper | small | ❌ | 0.7 (4.3) | 1.03 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e173† | hopper | BIG | ❌ | 0.0 (3.9) | 1.33 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e174† | cheetah | small | ~ | 165 (171) | 1.05 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e175† | cheetah | BIG | ✅ | 463 (502) | 1.05 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e176† | acrobot | small | ❌ | 1.9 (12.3) | 0.54 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e177† | acrobot | BIG | ❌ | 2.5 (25.9) | 0.75 | 200 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e161† | hopper | BIG | ❌ cancelled | 0.4 (1.5) | 1.17 | 0 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e181† | hopper | BIG | ❌ cancelled | 0.0 (3.0) | 1.00 | 0 | off | var τ8 (reg) | **sum** | no | no |
| e182† | hopper | BIG | ❌ cancelled | 0.1 (1.6) | 0.50 | 200 | prob_entropy | var τ8 (lagr) | mean | no | no |
| e183† | hopper | BIG | ❌ cancelled | 0.0 (0.4) | 0.94 | 200 | prob_entropy | var τ4 (lagr) | mean, expl_w=1.0 (10×) | no | no |
| e184† | acrobot | BIG | ❌ cancelled | 0.4 (17.6) | 0.30 | 200 | prob_entropy | var τ8 (lagr) | mean | no | no |
| e160 | cartpole | BIG | ❌ | 22 (182) | 1.04 | 0 | prob_entropy | var τ4 (lagr) | mean | no | no |
| e178 | hopper | BIG | ❌ | 0.03 (3.1) | 1.00 | 0 | off | var τ8 (reg) | mean | no | no |
| e179 | cartpole | BIG | ✅ | 740 (755) | 1.00 | 0 | off | var τ8 (reg) | mean | no | no |
| e180 | acrobot | BIG | ~ (climbing) | 177 (262) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e185 | hopper | BIG | ❌ | 0.03 (0.9) | 1.00 | 0 | off | var τ8 (reg) | mean | **yes** | no |
| e186 | cartpole | small | ✅  | 695 (766) | 0.72 | 200 | prob_entropy | var τ4 (lagr) | mean | **yes** | no |
| e187 | cartpole | small | ✅ | 697 (757) | 2.00 | 0 | off | var τ4 (reg) | mean | **yes** | no |
| e188 | cartpole | small | ~ | 254 (347) | 0.29 | 200 | prob | fix 8 | mean | **yes** | no |
| e189 | hopper | BIG | ❌ | 0.0 (1.7) | 1.00 | **200** | off | var τ8 (reg) | mean | no | no |
| e190 | cartpole | BIG | ✅ (climbing) | 714 (747) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e191 | cheetah | BIG | ~ (non-monotonic) | 293 (499) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e192 | acrobot | BIG | ❌ | 8.1 (19.4) | 1.00 | 0 | off | var τ8 (reg) | mean | **yes** | no |
| e193 | cartpole | small | ❌ | 42 (203) | 2.00 | 200 | off | var τ4 (reg) | mean | no | no |
| e194 | cartpole | small | ❌ | 152 (285) | 1.00 | 0 | off | var τ8 (reg) | mean | no | no |
| e195 | cartpole | small | ~ | 175 (352) | 2.00 | 200+adapt | off | var τ4 (reg) | mean | no | no |
| e196 | cartpole | small | ✅ | 648 (683) | 0.71 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** | no |
| e197 | hopper | small | ❌ | 0.23 (3.1) | 1.07 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** | no |
| e198 | cheetah | small | ❌ | 100 (122) | 1.17 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** | no |
| e199 | acrobot | small | ❌ | 3.4 (21.9) | 0.60 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** | no |
| e200 | cartpole | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | no |
| e201 | cheetah | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | no |
| e202 | hopper | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | no |
| e203 | acrobot | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | no |
| e204 | cartpole | small | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e205 | cartpole | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e206 | cheetah | small | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e207 | cheetah | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e208 | hopper | small | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e209 | hopper | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e210 | acrobot | small | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e211 | acrobot | BIG | ❌ cancelled, no results (freed for e212+) | - | - | 200+adapt(dual, target0.006) | prob_entropy | var τ8 (lagr) | mean | **yes** | **yes** |
| e212 | cheetah | small | ~ (noisy, windowed trend rising ~140) | 71 (193) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e213 | hopper | small | ❌ | 2 (40) | 1.00 | 0 | off | fix 8 | mean | no | no |
| e214‡ | cheetah | small | ✅ | 151 (169) | 0.83 | 0 | joint free | fix 8 | mean | no | no |
| e215‡ | cheetah | small | ✅ | 159 (197) | 0.82 | 200+adapt(dual, target0.006) | joint free | fix 8 | mean | no | no |
| e216‡ | cheetah | small | ~ | 124 (239) | 0.30 | 0 | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e217‡ | cheetah | small | ❌ | 44 (66) | 0.31 | 200+adapt(dual, target0.006) | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e218‡ | hopper | small | ❌ | 0.6 (125) | 0.93 | 0 | joint free | fix 8 | mean | no | no |
| e219‡ | hopper | small | ~ | 60 (117) | 0.94 | 200+adapt(dual, target0.006) | joint free | fix 8 | mean | no | no |
| e220‡ | hopper | small | ❌ | 20 (81) | 0.30 | 0 | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e221‡ | hopper | small | ❌ | 8 (21) | 0.30 | 200+adapt(dual, target0.006) | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e222‡ | cheetah | BIG | ✅ | 425 (438) | 0.96 | 0 | joint free | fix 8 | mean | no | no |
| e223‡ | cheetah | BIG | ✅ | 360 (455) | 0.95 | 200+adapt(dual, target0.006) | joint free | fix 8 | mean | no | no |
| e224‡ | cheetah | BIG | ❌ cancelled 07-15 | 134 (143) | 0.32 | 0 | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e225‡ | cheetah | BIG | ❌ cancelled 07-15 | 79 (201) | 0.30 | 200+adapt(dual, target0.006) | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e226‡ | hopper | BIG | ~ (was ✅ @65%, now declining — see note above) | 87 (302) | 0.96 | 0 | joint free | fix 8 | mean | no | no |
| e227‡ | hopper | BIG | ❌ (was ~ @65%, now declined) | 19 (165) | 0.94 | 200+adapt(dual, target0.006) | joint free | fix 8 | mean | no | no |
| e228‡ | hopper | BIG | ❌ cancelled 07-15 | 22 (52) | 0.30 | 0 | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e229‡ | hopper | BIG | ❌ cancelled 07-15 (collapsed) | 4 (268) | ~0.31 | 200+adapt(dual, target0.006) | joint ratchet→0.3 | fix 8 | mean | no | **yes** |
| e230‡ | cartpole | BIG | ~ cancelled @17%, still climbing | 435 (479) | 0.68 (ramping to 0.5, cut early) | 200+adapt(dual, target0.01) | joint ratchet→0.5 | fix 8 | mean | no | **yes** |
| e231‡ | acrobot | BIG | ~ cancelled @15%, weak but alive | 54 (84) | 0.71 (ramping to 0.5, cut early) | 200+adapt(dual, target0.01) | joint ratchet→0.5 | fix 8 | mean | no | **yes** |
| e232‡ | cheetah | BIG | ~ cancelled @16%, > e224/e225 already | 173 (178) | 0.69 (ramping to 0.5, cut early) | 200+adapt(dual, target0.01) | joint ratchet→0.5 | fix 8 | mean | no | **yes** |
| e233‡ | hopper | BIG | ~ cancelled @17%, ≫ e228 at 1/5 budget | **215 (216)** | 0.68 (ramping to 0.5, cut early) | 200+adapt(dual, target0.01) | joint ratchet→0.5 | fix 8 | mean | no | **yes** |
| e234§ | hopper | small | ~ (F15 floor, uninformative) | 1.9 (46.5) | n/a (no mask; kept 0.36) | 200+adapt(dual, target0.01) | impl struct-only | fix 8 | mean | no | no |
| e235§ | hopper | small | ❌ | 0.0 (0.7) | n/a (kept 0.56, λ railed 5.0) | 200+adapt(dual, target0.01) | impl struct+ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e236§ | hopper | small | ❌ | 0.0 (0.6) | n/a (kept 0.56, λ railed 5.0) | 0 | impl ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e237§ | hopper | small | ❌ | 0.0 (0.5) | n/a (kept 0.56, λ railed 5.0) | 0 | impl direct 0.7kept | fix 8 | mean | no | no |
| e238§ | cheetah | small | ~ (below Director-at-1M) | 117.8 (159.0) | n/a (no mask; kept 0.26) | 200+adapt(dual, target0.01) | impl struct-only | fix 8 | mean | no | no |
| e239§ | cheetah | small | ❌ | 3.0 (5.0) | n/a (kept 0.56, λ railed 5.0) | 200+adapt(dual, target0.01) | impl struct+ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e240§ | cheetah | small | ❌ | 2.2 (4.6) | n/a (kept 0.51, λ railed 5.0) | 0 | impl ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e241§ | cheetah | small | ❌ | 1.4 (4.2) | n/a (kept 0.56, λ railed 5.0) | 0 | impl direct 0.7kept | fix 8 | mean | no | no |
| e242§ | hopper | BIG | ~ (partial, still climbing @1M) | 115.1 (115.7) | n/a (no mask; kept 0.38) | 200+adapt(dual, target0.01) | impl struct-only | fix 8 | mean | no | no |
| e243§ | hopper | BIG | ❌ | 0.0 (0.2) | n/a (kept 0.53, λ railed 5.0) | 200+adapt(dual, target0.01) | impl struct+ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e244§ | hopper | BIG | ❌ | 0.0 (1.0) | n/a (kept 0.49, λ railed 5.0) | 0 | impl ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e245§ | hopper | BIG | ❌ | 0.0 (0.6) | n/a (kept 0.51, λ railed 5.0) | 0 | impl direct 0.7kept | fix 8 | mean | no | no |
| e246§ | cheetah | BIG | ✅ (≈ Director-at-1M control) | **329.8 (335.4)** | n/a (no mask; kept 0.20) | 200+adapt(dual, target0.01) | impl struct-only | fix 8 | mean | no | no |
| e247§ | cheetah | BIG | ❌ | 3.4 (6.1) | n/a (kept 0.52, λ railed 5.0) | 200+adapt(dual, target0.01) | impl struct+ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e248§ | cheetah | BIG | ❌ | 4.1 (8.6) | n/a (kept 0.53, λ railed 5.0) | 0 | impl ratchet→0.7kept | fix 8 | mean | no | **yes** |
| e249§ | cheetah | BIG | ❌ | 3.4 (7.9) | n/a (kept 0.55, λ railed 5.0) | 0 | impl direct 0.7kept | fix 8 | mean | no | no |

Reading the columns together: every ✅ dense-task row sits at blk/step ≥1.0 by the end of
training (Director-dense or denser) except e170/e186/e196 (0.61–0.72) — high final blk/step
is **not** on its own a collapse signature (e171/e175 are campaign champions at 1.05–1.13);
what distinguishes e160/e197/e198 is that mask_frac is already inflated mid-run while score
is still low, not just a late-training drift after success. The τ4-vs-τ8 columns line up
cleanly with the struct column: every working τ4 var-K cell (e46 doesn't apply — that's τ8;
e170/e171/e174/e175/e187/e196) either has struct on or has countdown standing in for it
(e187); every dead-or-weak τ4 cell without either (e166/e167/e169/e176/e193/e195) stays low.
τ8 cells (e178/e179/e181/e185/e189/e192/e194) are struct-independent on cartpole (e179/e194
split by scale, see §6) but never rescue hopper/acrobot regardless of struct or countdown.

The mask-only fixed-K8 cluster (e162–e165) sharpens the sparse-task question further. On
cartpole, masking alone (no var-K at all) gets a real but modest score (270–412) — below
the full combined recipe (653–843) but clearly non-zero, confirming masking is an
independent, if weaker, dense-task mechanism (blk/step 0.31, properly sparse). On hopper,
the *same* fixed-K8-plus-masking recipe (e164/e165) is dead (0.0–0.3), even though it uses
the exact fixed duration (K=8) that keeps e124/e180 alive. Combined with e169/e168/e173/e172
(masking off or on, var-K on, also all dead on hopper) and e178/e179's split (var-K alone is
fine on cartpole, dead on hopper), the honest reading is: **masking alone and variable
duration alone are each independently sufficient to kill hopper/acrobot**, while vanilla
fixed-K8 whole-code Director (neither mechanism) is the only thing that survives. This is a
stronger claim than "variable duration is the culprit" (§6, e160 interim note) — sparse
tasks look fragile to *any* deviation from vanilla Director, not to one specific mechanism.
