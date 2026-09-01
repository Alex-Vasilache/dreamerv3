"""Serves the interactive goal-code demo from the login node.

    source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
    python tools/goal_demo/server.py --port 8899

then open ``http://<login-node>:8899``, the same way ``tensorboard_login.sh``
is reached.

Renderers are built lazily (~25s each: module tree, jit, checkpoint) and then
kept, so the first environment is usable while the rest warm up in the
background. Frames come back as raw RGB rather than PNG -- 12KB each, no encode
step, and the page writes them straight into a canvas.

Latency, measured on the 32-core login node: ~30ms for one frame, ~155ms for a
batch of 16. The page hides even that by prefetching, after every edit, every
single-block variation of the current code; see ``index.html``.

Three endpoints answer about codes rather than draw them. ``/api/values`` reads
the agent's critics at the state a code decodes to, optionally against a second
code as the worker's goal and a third as a reference to measure distance from;
it is ~10ms, an order below a frame, because it skips the convolutional
decoder. ``/api/at_goal_distance`` walks out along a ray in code space until the
goal is a requested distance away. Both take the render lock, so they queue
behind frames rather than fighting them for the same cores.
"""
import argparse
import collections
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)

import runs as runsmod  # noqa: E402

# Four classes of headroom on each side of the eight the model was trained on.
EXTRA = 4
CACHE_PER_MODEL = 6000  # ~74MB of frames each
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 128)

# What "8 points per block lie on a line" is worth measuring against: the share
# of spread a single axis captures for 8 points drawn at random in R^8, which is
# already 0.41 on average. Reproduced by check_render.py's random_pc1_baseline.
PC1_RANDOM = {'pc1_random_mean': 0.405, 'pc1_random_p95': 0.510}


def _bucketed(batch):
  """Pad a batch up to a bucket size by repeating its last row.

  jit recompiles per batch shape, so rounding up to one of a few sizes and
  throwing the padding away beats paying a compile for every odd chunk length.
  """
  size = next((b for b in BUCKETS if b >= len(batch)), len(batch))
  if size == len(batch):
    return batch
  return np.concatenate([batch, np.repeat(batch[-1:], size - len(batch), 0)])


class Registry:
  """Lazily built renderers plus their frame caches."""

  def __init__(self, dtype='float32'):
    self.dtype = dtype
    self._models = {}
    self._caches = {}
    self._errors = {}
    self._scales = {}
    self._build_lock = threading.Lock()
    self._render_lock = threading.Lock()
    self._preparing = False
    self.last_request = 0.0

  def status(self):
    keys = [(t, a) for t in runsmod.TASKS for a in runsmod.ARMS]
    return {f'{t}/{a}': (
        'error' if (t, a) in self._errors else
        'ready' if (t, a) in self._models else 'pending') for t, a in keys}

  def get(self, task, arm):
    key = (task, arm)
    if key in self._models:
      return self._models[key]        # never wait on a build we do not need
    if key in self._errors:
      raise RuntimeError(self._errors[key])
    # One build at a time: two 15s builds in parallel would fight over the same
    # 32 cores and neither would finish sooner.
    with self._build_lock:
      if key not in self._models:
        import render_core
        try:
          self._models[key] = render_core.GoalRenderer(
              task, arm, compute_dtype=self.dtype)
        except Exception as e:
          self._errors[key] = f'{type(e).__name__}: {e}'
          traceback.print_exc()
          raise
        self._caches[key] = collections.OrderedDict()
        # Part of the build, not of the first request that needs it: the page
        # asks for it as soon as a panel appears, and a checkpoint that reports
        # itself ready should have nothing left to compute.
        self._scales[key] = self._models[key].goal_scale()
        print(f'[goal-demo] ready: {task}/{arm} '
              f'({runsmod.info(task, arm)["exp"]})', flush=True)
    return self._models[key]

  def prepare(self, task):
    """Start building both arms of ``task`` if they are not up yet."""
    todo = [a for a in runsmod.ARMS if (task, a) not in self._models]
    if not todo or self._preparing:
      return
    def run():
      try:
        for arm in todo:
          try:
            self.get(task, arm)
          except Exception:
            pass
      finally:
        self._preparing = False
    self._preparing = True
    threading.Thread(target=run, daemon=True).start()

  def render(self, task, arm, codes):
    """``codes`` is a list of per-block class-index lists. Returns raw RGB."""
    self.last_request = time.time()
    model = self.get(task, arm)
    cache = self._caches[(task, arm)]
    codes = np.asarray(codes, np.float32).reshape(-1, model.blocks)
    keys = [c.round(3).tobytes() for c in codes]

    todo, seen = [], set()
    for i, k in enumerate(keys):
      if k not in cache and k not in seen:
        seen.add(k)
        todo.append(i)
    if todo:
      with self._render_lock:
        imgs, _ = model.render(_bucketed(codes[todo]))
      for i, img in zip(todo, imgs[:len(todo)]):
        cache[keys[i]] = img.tobytes()
      while len(cache) > CACHE_PER_MODEL:
        cache.popitem(last=False)
    for k in keys:
      cache.move_to_end(k)
    return b''.join(cache[k] for k in keys)

  def values(self, task, arm, codes, goals=None, ref=None):
    """Critic readings for each code, and its goal's distance to ``ref``.

    ``goals`` (one code per entry) is what the worker critic aims at, defaulting
    to the code itself. ``ref`` is a single code every entry is measured against
    in goal space -- the saved state, for the demo's comparison -- and is sent
    back as a distance rather than as the 1024 numbers it is computed from.
    """
    self.last_request = time.time()
    model = self.get(task, arm)
    if not model.has_values:
      return dict(values=None)
    codes = np.asarray(codes, np.float32).reshape(-1, model.blocks)
    goals = codes if goals is None else np.asarray(
        goals, np.float32).reshape(-1, model.blocks)
    assert goals.shape == codes.shape, (codes.shape, goals.shape)
    n = len(codes)
    with self._render_lock:
      vals = model.values(_bucketed(codes), _bucketed(goals))
      out = [{k: float(v[i]) for k, v in vals.items()} for i in range(n)]
      if ref is not None:
        vectors = model.goal_vectors(_bucketed(codes))[:n]
        base = model.goal_vectors(np.asarray(ref, np.float32))[0]
        for row, vec in zip(out, vectors):
          denom = np.linalg.norm(vec) * np.linalg.norm(base)
          row['dist'] = float(np.linalg.norm(vec - base))
          row['cos'] = float(vec @ base / max(denom, 1e-8))
    return dict(values=out)

  def at_goal_distance(self, task, arm, code, target, span, seed=None):
    """A code sitting ``target`` away from ``code`` in goal space."""
    self.last_request = time.time()
    model = self.get(task, arm)
    rng = np.random.default_rng(seed)
    with self._render_lock:
      return model.code_at_goal_distance(code, target, span, rng)

  def meta(self, task, arm):
    model = self.get(task, arm)
    table = model.codebook()
    line = None
    if table is not None:
      fracs = []
      for block in table:
        centered = block - block.mean(0)
        sv = np.linalg.svd(centered, compute_uv=False)
        fracs.append(float(sv[0] ** 2 / (sv ** 2).sum()))
      line = dict(pc1_min=min(fracs), pc1_mean=float(np.mean(fracs)),
                  dim=int(table.shape[-1]), **PC1_RANDOM)
    return dict(
        blocks=model.blocks, classes=model.classes, values=model.has_values,
        codebook=line, scale=self._scales[(task, arm)],
        **runsmod.info(task, arm))


def warm(registry, order, idle=3.0):
  """Load the remaining checkpoints in the background, but only while idle.

  A build saturates the same 32 cores a render needs; letting one start
  underneath an active session turned a 155ms batch into 920ms. Waiting for a
  few seconds of quiet costs nothing -- the user is looking at one environment
  at a time anyway.
  """
  def run():
    for task, arm in order:
      while time.time() - registry.last_request < idle:
        time.sleep(0.5)
      try:
        registry.get(task, arm)
      except Exception:
        pass
    print('[goal-demo] all checkpoints loaded', flush=True)
  threading.Thread(target=run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):

  registry = None
  protocol_version = 'HTTP/1.1'

  def log_message(self, fmt, *args):
    pass  # The default logs every request; the prefetch makes that unreadable.

  def _send(self, code, body, ctype):
    self.send_response(code)
    self.send_header('Content-Type', ctype)
    self.send_header('Content-Length', str(len(body)))
    self.send_header('Cache-Control', 'no-store')
    self.end_headers()
    self.wfile.write(body)

  def _json(self, obj, code=200):
    self._send(code, json.dumps(obj).encode(), 'application/json')

  def do_GET(self):
    path = self.path.split('?')[0]
    try:
      if path in ('/', '/index.html'):
        with open(os.path.join(_HERE, 'index.html'), 'rb') as f:
          self._send(200, f.read(), 'text/html; charset=utf-8')
      elif path == '/api/meta':
        self._json(dict(
            tasks=[dict(id=t, label=runsmod.RUNS[t]['label'])
                   for t in runsmod.TASKS],
            arms=[dict(id=a, label=runsmod.ARM_LABELS[a])
                  for a in runsmod.ARMS],
            step=int(runsmod.STEP), extra=EXTRA,
            runs={t: {a: runsmod.info(t, a) for a in runsmod.ARMS}
                  for t in runsmod.TASKS}))
      elif path == '/api/status':
        self._json(self.registry.status())
      elif path == '/api/prepare':
        # Put the environment the user just picked at the front of the build
        # queue, ahead of whatever the idle warmer would have done next.
        args = dict(p.split('=') for p in self.path.split('?')[1].split('&'))
        self.registry.prepare(args['task'])
        self._json(self.registry.status())
      elif path == '/api/model':
        args = dict(p.split('=') for p in self.path.split('?')[1].split('&'))
        self._json(self.registry.meta(args['task'], args['arm']))
      else:
        self._json({'error': 'not found'}, 404)
    except Exception as e:
      traceback.print_exc()
      self._json({'error': f'{type(e).__name__}: {e}'}, 500)

  def do_POST(self):
    try:
      n = int(self.headers.get('Content-Length', 0))
      req = json.loads(self.rfile.read(n) or b'{}')
      path = self.path.split('?')[0]
      if path == '/api/render':
        body = self.registry.render(req['task'], req['arm'], req['codes'])
        self._send(200, body, 'application/octet-stream')
      elif path == '/api/values':
        self._json(self.registry.values(
            req['task'], req['arm'], req['codes'],
            req.get('goals'), req.get('ref')))
      elif path == '/api/at_goal_distance':
        self._json(self.registry.at_goal_distance(
            req['task'], req['arm'], req['code'], req['target'],
            req['span'], req.get('seed')))
      else:
        self._json({'error': 'not found'}, 404)
    except Exception as e:
      traceback.print_exc()
      self._json({'error': f'{type(e).__name__}: {e}'}, 500)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--host', default='0.0.0.0')
  ap.add_argument('--port', type=int, default=8899)
  ap.add_argument('--task', default=runsmod.TASKS[0], choices=runsmod.TASKS,
                  help='environment to load first')
  ap.add_argument('--dtype', default='float32',
                  help='float32 is ~1.5x faster than the bfloat16 the runs '
                       'trained in and differs by <=4/255; see check_render.py')
  ap.add_argument('--no-warm', action='store_true',
                  help='load each checkpoint on first use instead of upfront')
  args = ap.parse_args()

  registry = Registry(args.dtype)
  Handler.registry = registry
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  host = os.uname().nodename
  print(f'[goal-demo] http://{host}:{args.port}  '
        f'(ctrl-c to stop)', flush=True)
  first = [(args.task, a) for a in runsmod.ARMS]
  rest = [(t, a) for t in runsmod.TASKS for a in runsmod.ARMS
          if (t, a) not in first]
  if not args.no_warm:
    warm(registry, first + rest)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print('\n[goal-demo] stopped')


if __name__ == '__main__':
  main()
