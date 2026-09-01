# Why the UCB selection bonus never annealed (e713-e717)

Reads the three count tables straight out of the archived milestone checkpoints
-- no GPU, no environment -- and answers three questions about them.

| script | what it does | env |
|---|---|---|
| `extract.py` | pulls `select_tab` / `fine_tab` / `coarse_tab` out of every `ckpt_milestones/*/agent.pkl` into `results/tables.npz` | `dreamerv3_env` (the pickles are numpy 2.x) |
| `analyse.py` | coverage, concentration and bonus statistics per milestone | plotenv |
| `gapsim.py` | **the mechanism check**: freeze the manager's policy at a real measured per-state distribution, advance only the table, and see whether the candidate gain still rises | plotenv |
| `counterfactual.py` | sweeps grid resolution `b` and kernel width `h` on a simulated decision stream, to ask what *would* anneal | plotenv |
| `figdata.py` | merges everything into `results/figdata.json` for the paper figure | plotenv |

```bash
bash -lc 'source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh && python extract.py results/tables.npz'
P=/work/DoyaU/vasilache/work/plotenv/bin/python
$P gapsim.py results/tables.npz ../manager_rao/results results/gap.json
$P counterfactual.py ../manager_rao/results results/counterfactual.json
$P figdata.py results/tables.npz results/gap.json results/counterfactual.json results/figdata.json
$P ../../../26_04_HRL-paper/figures/motivation/make_no_anneal.py results/figdata.json
```

## What it found

1. **Smearing works, for coverage.** Under a hard deposit only 3-30% of the
   65,536 cells are ever touched in a whole run; smeared, **every** cell holds
   mass from the first milestone on. It also cuts the tilt by ~25% and turns a
   rising curve flat under a stationary policy.
2. **The level anneals, the spread does not.** Mean bonus 0.52 -> 0.43 over 4M
   steps while the chosen candidate's advantage climbs to 1.14-1.29.
3. **Why.** `1/sqrt(n+1)` is nearly flat below one count. Rare and busy cells
   grow their counts by a similar factor, but only the busy ones lose bonus, so
   the best-available-over-typical ratio widens 1.6x -> 3.8x.
4. **Proof it is the table and not the policy.** With the policy frozen at a
   real measured distribution the rise still appears on all five runs.
5. **What would fix it.** `b=3` (6,561 cells, 76 counts/cell) genuinely anneals;
   `b=4` is flat; `b>=5` climbs. The alternative is an explicit schedule on `c`.

Caveat on `counterfactual.py`: a stationary policy piles counts into fewer cells
than a real run does, so its absolute gains (1.5-2.2) sit above the measured
ones (1.03-1.29). Read it for the ordering between grids, not for the level.
