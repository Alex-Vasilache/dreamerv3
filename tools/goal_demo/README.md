# Goal code explorer

Edit a goal code by hand and watch what the world model decodes it into, for
Director and for our SOM-line arm side by side, on the same environment.

```bash
./tools/goal_demo/run_demo.sh          # then open http://saion-login2:8899
```

Runs on the login node, same as `tensorboard_login.sh`. `--port` to move it.

## What you are looking at

Each panel is one 3M-step milestone checkpoint. Rows of the grid are the 8
blocks of the goal code, columns are the classes for that block. Click a cell
to set a block, drag along a row to sweep it, hover to preview without
committing. The frame underneath is

```
code --goal_dec--> deter --dyn._prior--> feat --dec--> 64x64 image
```

which is the path `Agent.report` renders goals with. `render_core.py` builds the
real `dreamerv3.agent.Agent` module tree and runs `nj.init` over exactly that
path, so nothing about the decode is reimplemented — it creates parameters only
for the three modules involved (~55M of the checkpoint's 232M) and loads them
from `ckpt_milestones/000003000000/agent.pkl`.

The active row is decoded ahead of time in the background to fill the cache, so
dragging a row or hovering a cell lands instantly instead of waiting on a
render.

## What the agent thinks of the state

Under the frame are five numbers read off the same `feat`, using the
checkpoint's own heads (`render_core.values`). The three critics are kept apart
because they answer different questions, and pooling them would hide exactly the
disagreement worth seeing:

| Shown | Head | What it is |
|---|---|---|
| manager · task | `mgr_extr_val` | discounted task return expected from here |
| manager · explore | `mgr_expl_val` | exploration return, whose reward is the goal autoencoder's own squared reconstruction error — it scores how *unfamiliar* a state is, not how good |
| worker · on its goal | `wkr_goal_val` | worker return under the cosine goal reward |
| reward here | `rew` | one-step reward |
| keeps going | `con` | probability the episode continues |

`wkr_goal_val` is the only critic whose input is a *pair* — `[deter, stoch,
goal]`, the same layout `_feat_goal2tensor` builds in training. On its own it is
read with the state as its own goal, the worker's best case; the saved-state
comparison also asks it across the two codes, which is the one reading that says
whether the worker thinks it could get from one to the other.

Values are un-normalized with each head's `valnorm` stats. That is a no-op on
the pinned runs (`valnorm.impl: none`, so the heads predict raw returns) and
stays correct if a later pin trains with a real one.

Same caveat as the frame: all of these were trained on features of *real*
states, and a hand-edited code is only as real as the goal decoder makes it.
`check_render.py` asserts what can be asserted — that the labels sit on the
heads they name, that the task critic ranks states the way the reward head does
(`+0.6` on cartpole, against `−0.9` for the exploration critic), and that the
worker critic actually consumes the goal it is handed.

## A new state at a given distance

Two distances, and they are not interchangeable.

**Class steps** is the index distance the jump uses, `Σ|Δc_l|`, exact by
counting: `codeAtDistance` is the second half of the jump with `d` handed in
instead of drawn, so the jump is now just `p(d)` in front of it. A fractional
code is rounded first.

**Goal space** is the distance between the decoded goals, which no amount of
counting gets you — but the demo's class indices are real-valued, so the goal
moves *continuously* along a ray in code space. `code_at_goal_distance` draws a
random direction, scales it so its largest block step is one class, cuts it off
where the first block would leave the displayed columns, then brackets the first
crossing of the target on a 48-point grid and refines it twice. Bracketing the
first crossing rather than bisecting means it does not assume the distance grows
monotonically along the ray — it does not have to, and outside the trained range
it does not: on Director a ray often peaks mid-way and comes back. It lands
within ~0.4% of the requested distance, and reports the distance re-measured
from the code it returns rather than from the batch it was found on.

`goal_scale` gives the number a scale to sit on, and the two arms differ in a
way that is the ordered-index argument from a third direction. On cartpole at
3M, moving one block by one class against two unrelated codes:

| | one class step | unrelated pair | ratio |
|---|---|---|---|
| Director | 4.32 | 9.32 | 2.2× |
| Ours | 1.53 | 13.52 | 8.9× |

One class step on Director already covers nearly half the distance between two
unrelated codes; on ours it covers a ninth of a much larger span.

## Save and compare

**save this state** freezes the current frame beside the live one and turns the
panel into a two-state comparison: class steps apart, blocks changed, goal
distance and cosine, mean absolute pixel difference between the two frames, and
every critic for both states with the change between them. The last two rows are
the pair reading only the worker critic can give — what it makes of the saved
state while aiming at the current goal, and the reverse. A large index gap with
a small goal distance, or a big goal distance the pictures barely reflect, is
the case worth looking at.

The comparison costs one request per edit, not two: the same `/api/values` call
carries the current code, the two cross-goal readings, and the saved code as the
reference to measure distance from.

Each grid is filled in its arm's colour, the same pair as Figure 10 of the
HRL paper (`26_04_HRL-paper`, `figures/motivation/block_distributions.png`):
`#74736f` grey for Director's independent categorical per block, `#498cdc` blue
for our discretized Gaussian. The rule under columns 0–7 brackets the range the
model was trained on; a selected cell outside it keeps the arm colour and gains
an orange ring.

## Extrapolation

Columns outside 0–7 are class indices the model never saw. The two arms reach
them differently, because only one of them has a geometry to extend.

**Ours** works in the codebook (`class_embeddings`). Inside the range the block
walks the piecewise-linear path through its `C` embeddings, so every integer
still lands exactly on a trained entry. Outside, it leaves the end entry along
the block's **fitted line** `d_l` — the first principal direction of all `C`
entries, oriented toward increasing class — one **mean segment length** `s_l`
per class:

```
z(t) = e_0       + t         * s_l * d_l          t < 0
     = (1-u) e_i + u e_{i+1}                      0 <= t <= C-1
     = e_{C-1}   + (t-(C-1)) * s_l * d_l          t > C-1
```

Anchoring on the real end entries keeps `z` continuous at 0 and `C-1`.

This replaced an earlier rule that simply repeated the end segment. That made
extrapolation inherit whichever end gap happened to be there, and measured
across all four environments the two end gaps are always the **shortest** in the
block (~0.4 against an interior mean of ~0.6), so extrapolation crawled. The
mean length and the overall direction fix both halves of that.

**Director** has no embedding, and its `C` one-hots are mutually equidistant, so
there is no line to fit. It extends the one-hot family instead
(`class_weights`):

```
i = clip(floor(t), 0, C-2);  u = t - i;  w[i] = 1-u;  w[i+1] = u
```

giving `[2, -1, 0, ...]` at `t = -1`. Those 64 numbers go into the decoder MLP as
input activations — defined, but with no learned metric behind it. That
asymmetry is the comparison worth making.

`spacing.py` measures the step sizes both arms actually take, in the codebook
and in goal space.

## Distant jump

The **distant jump** button is the paper's ε-greedy exploration move, triggered
on demand rather than firing with probability ε. From the current code `c`:

```
d_max(c) = sum_l max(c_l, C-1-c_l)          # how far the code can move at all
p(d)     = 2d / (d_max (d_max + 1))         # bigger jumps are the likely ones
```

`d` is then spread over the blocks within each block's capacity — uniformly over
the assignments that respect it — and each moved block goes up or down, whichever
stays on the line, or a coin flip when both do. The result sits at exactly index
distance `d` from where it started, and the readout under the buttons reports
`d`, `d_max` and the expected `E[d] = (2/3)(d_max + 1/2)`.

`d_max` is a property of the code, not a constant: over uniformly drawn codes it
averages 44, and only a code with every block already at 0 or `C-1` reaches 56.
So the rule draws `d ≈ 30` in practice. `test_ui.js` checks all of that against
the paper's formulas.

The button works on both panels on purpose. The move is only worth making
because our index is ordered — on Director's code a large index gap does not
imply a distant goal — and putting the same jump on both sides is the way to
see it.

## Which checkpoints

`runs.py` pins, per environment, the better-scoring seed of each arm, scored as
mean episode return over env steps 2.7M–3.0M. `pick_runs.py` recomputes those
from `metrics.jsonl` and reports whether a pin has gone stale — worth running
when new seeds finish.

## Speed

Frames are rendered live on CPU: ~40ms for one, ~200ms for a batch of 24. The
page hides that by prefetching, after every edit, every single-block variation
of the current code (8 blocks × 16 columns), active row first, in chunks small
enough that an interactive frame is never stuck behind a sweep. So a click or
drag onto anything one block away from the current code is served from the
browser cache with no request at all.

Three things mattered for that:

* `DREAMERV3_CONV_IMPL=reference`, which `source_dreamerv3_env.sh` exports for
  the V100 cuDNN bug, is 5–10× slower. `render_core` drops it before importing
  `embodied.jax.nets`.
* `nj.pure` also returns the state; returning it from the jit made every call
  re-materialize all 220MB of weights. Dropping it inside the jit cut a single
  render from 94ms to 29ms.
* Checkpoints load in the background but only while no render has arrived for
  3s — a build saturates the same cores and turned a 155ms batch into 920ms.

Rendering uses float32 rather than the bfloat16 the runs trained in: ~1.5×
faster and at most 4/255 different per pixel.

Memory is ~300MB per loaded checkpoint, ~2.5GB with all eight up.

## Checks

```bash
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
python tools/goal_demo/check_render.py [task]     # decode path is faithful
node tools/goal_demo/test_ui.js                   # page logic (no browser needed)
```

`check_render.py` confirms the native and reference convolutions agree (≤1/255),
that float32 and bfloat16 agree (≤4/255), that each block's codebook is close to
a line, that the critic readout is wired to the heads it names, that the
goal-distance walk lands on the distance it was asked for, and that `goal_enc`
and `goal_dec` round-trip a code far above chance — which is what catches
loading same-shaped weights from the wrong run. `test_ui.js` lifts the script
block out of `index.html` and runs it against a stub DOM, so it tests the code
that ships, including that a generated code sits at *exactly* the requested
class distance.
