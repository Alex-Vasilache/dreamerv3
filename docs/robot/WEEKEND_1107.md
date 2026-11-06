# Weekend 2026-11-07/09: fastest reliable live learning, balance and commands

Ask (Alex, Fri 11-06 evening): keep the GPUs busy over the weekend with
structured sweeps that find how to make live learning as fast and as reliable
as possible, with little babysitting and little movement, for balancing and
for following commands. Results by Monday morning, German time.

Everything runs in the fitted sim on the Germany floor (sim table schedule +
drive-train randomisation), through the phone's own loop (`sim_phone`), with
the live demo's settings (e40006: publish every sync, drive home capped at
10 s, phone policy age ~1.3 s). Params come from `sbatch/wk/gen.py`.
Experiments e42001 onward.

## Scores

Balance, per run (n seeds per arm, mean ± SD):
- *fast*: first minute with upright (|tilt| < 3°) ≥ 0.9;
- *steady*: bumper crossings a minute, minutes 15-29;
- *reliable*: fraction of seeds up by minute 6 that stay below 10
  crossings a minute after minute 10;
- *movement*: |wheel speed| and roaming after minute 10;
- *compute-invariant*: the same at the slow (P100, ratio ~280) and fast
  (IAS A5000/3090, ~570) learner.

Commands: minutes to forward and turn tracking correlation 0.7, final gain,
falls a minute, and |speed| at a zero stick.

## Stages

| stage | when (German time) | question | runs |
|---|---|---|---|
| IA1/IA2 | Fri → Fri ~22:00 | which drift control keeps the fast start at every learner speed | 108 |
| K1 | Fri 18:00 → Sat ~00:00 | command learning on the Germany floor: base and seven one-factor changes | 32 × 60 min |
| W1 | Fri ~20:00 → Sat ~06:00 | balance speed-up screen on the IA winner, one factor at a time, slow and fast learner | 112 × 30 min |
| W2/K2 | Sat | best factors combined (factorial), n = 6 per speed | ~100 |
| W3/K3 | Sun → Mon 06:00 | confirmation: best two against the live recipe, n = 12, 60 min, three learner speeds, carpet and pushes as holdouts | ~150 |

Each stage is analysed before the next is designed; the stage log below
records what was found and why the next one looks as it does.

## Log
