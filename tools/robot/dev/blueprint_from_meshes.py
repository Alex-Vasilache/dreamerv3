"""Line drawings of the robot from its STL meshes, for the phone's dashboard.

Projects the rlSim2Real meshes (tools/robot/dev/fetch_spr_meshes.sh) in three
orthographic views -- side, front, top -- and keeps the edges a draughtsman
would draw: outlines (silhouette and open edges) and creases sharper than
--crease degrees. Hidden lines are removed with a depth buffer. Writes a JS
object of SVG path data, in mm with the axle at the origin, that
dashboard.html (dreamerBridge app) embeds between its BEGIN/END BLUEPRINT
markers:

  python tools/robot/dev/blueprint_from_meshes.py ~/.cache/spr_meshes OUT.js
  python tools/robot/dev/blueprint_from_meshes.py ~/.cache/spr_meshes \
      --html $ROBOT_ANDROID_DIR/apps/dreamerBridge/src/main/python/webpages/dashboard.html

Parts are kept apart so the page can move them: the body and phone tilt about
the axle, and the wheel turns in the side view.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from embodied.envs import robot_mujoco  # noqa: E402

HALF_TRACK_MM = robot_mujoco.HALF_TRACK * 1e3


def mesh_parts(meshdir):
  """World-frame triangles (mm, axle at the origin, upright) of each mesh geom
  as MuJoCo places it -- MuJoCo re-centres every mesh on its own centre of
  mass, so the STL coordinates alone do not say where a part sits."""
  import mujoco
  model = mujoco.MjModel.from_xml_string(robot_mujoco.model_xml(
      0.161, 0.120, visual=(0, 0, 0), meshdir=str(meshdir)))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  axle = data.xpos[model.body('robot').id].copy()
  out = {}
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    m = model.geom_dataid[g]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m)
    v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
    f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
    verts = model.mesh_vert[v0:v0 + nv].astype(np.float64)
    faces = model.mesh_face[f0:f0 + nf]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    out[name] = (world - axle)[faces] * 1e3
  return out


def disc(radius, x0, x1, n=180):
  """Triangles of a closed cylinder along x (mm): the region a spinning
  wheel sweeps, whatever its angle."""
  a = np.linspace(0, 2 * np.pi, n + 1)
  ring = np.stack([np.zeros_like(a), radius * np.cos(a), radius * np.sin(a)], -1)
  tris = []
  for x in (x0, x1):
    c = np.array([x, 0, 0])
    for i in range(n):
      tris.append([c, ring[i] + [x, 0, 0], ring[i + 1] + [x, 0, 0]])
  for i in range(n):
    p, q = ring[i], ring[i + 1]
    tris.append([p + [x0, 0, 0], q + [x0, 0, 0], q + [x1, 0, 0]])
    tris.append([p + [x0, 0, 0], q + [x1, 0, 0], p + [x1, 0, 0]])
  return np.array(tris)


# view: (screen x axis, screen y axis, towards-viewer axis), all in body frame.
# Side: front (+y) to the right, seen from the robot's right (+x).
VIEWS = {
    'side': (np.array([0, 1, 0]), np.array([0, 0, -1]), np.array([1, 0, 0])),
    'front': (np.array([-1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])),
    'top': (np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1])),
}


def feature_edges(tri, toward, crease_deg):
  """Segments (k, 2, 3) on outlines and creases."""
  normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  area = np.linalg.norm(normals, axis=1)
  keep = area > 1e-9
  tri, normals = tri[keep], normals[keep] / area[keep, None]
  verts, index = np.unique(np.round(tri.reshape(-1, 3), 3), axis=0,
                           return_inverse=True)
  faces = index.reshape(-1, 3)
  edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
  owner = np.tile(np.arange(len(faces)), 3)
  edges.sort(axis=1)
  order = np.lexsort((edges[:, 1], edges[:, 0]))
  edges, owner = edges[order], owner[order]
  same = np.all(edges[1:] == edges[:-1], axis=1)
  facing = normals @ toward > 0
  cos_crease = np.cos(np.radians(crease_deg))
  out = []
  i = 0
  n = len(edges)
  while i < n:
    if i + 1 < n and same[i]:
      a, b = owner[i], owner[i + 1]
      sharp = normals[a] @ normals[b] < cos_crease
      outline = facing[a] != facing[b]
      if (sharp and (facing[a] or facing[b])) or outline:
        out.append(edges[i])
      i += 2
      while i < n and i > 0 and same[i - 1]:
        i += 1   # non-manifold extras
    else:
      if facing[owner[i]]:
        out.append(edges[i])   # open edge
      i += 1
  out = np.array(out)
  return verts[out], (tri, facing)


def depth_buffer(tri, ids, sx, sy, toward, res, lo, hi):
  """Per pixel, the nearest surface's depth along 'toward' and the index of
  the part it belongs to (-1: empty)."""
  w = int(np.ceil((hi[0] - lo[0]) / res)) + 1
  h = int(np.ceil((hi[1] - lo[1]) / res)) + 1
  zbuf = np.full((h, w), -np.inf)
  idbuf = np.full((h, w), -1)
  p = np.stack([tri @ sx, tri @ sy, tri @ toward], -1)   # (n, 3, 3)
  px = (p[..., 0] - lo[0]) / res
  py = (p[..., 1] - lo[1]) / res
  for t in range(len(p)):
    x0, x1 = int(np.floor(px[t].min())), int(np.ceil(px[t].max()))
    y0, y1 = int(np.floor(py[t].min())), int(np.ceil(py[t].max()))
    if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
      continue
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    (ax, bx, cx), (ay, by, cy) = px[t], py[t]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-12:
      continue
    l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
    l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
    l3 = 1 - l1 - l2
    inside = (l1 >= -0.01) & (l2 >= -0.01) & (l3 >= -0.01)
    if not inside.any():
      continue
    z = np.where(inside, l1 * p[t, 0, 2] + l2 * p[t, 1, 2] + l3 * p[t, 2, 2],
                 -np.inf)
    sub = zbuf[y0:y1 + 1, x0:x1 + 1]
    nearer = z > sub
    sub[nearer] = z[nearer]
    idbuf[y0:y1 + 1, x0:x1 + 1][nearer] = ids[t]
  return zbuf, idbuf


def outlines(idbuf, part, res, lo):
  """Boundary of the region where `part` is the nearest surface."""
  import contourpy
  mask = np.pad((idbuf == part).astype(float), 1)
  gen = contourpy.contour_generator(z=mask)
  return [np.stack([(c[:, 0] - 1) * res + lo[0], (c[:, 1] - 1) * res + lo[1]], -1)
          for c in gen.lines(0.5)]


def visible_segments(segs, zbuf, sx, sy, toward, res, lo, tol, step):
  """Split each 3D segment at hidden samples; return visible 2D pieces."""
  pieces = []
  h, w = zbuf.shape
  for a, b in segs:
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    t = np.linspace(0, 1, n)[:, None]
    pts = a + t * (b - a)
    x, y, z = pts @ sx, pts @ sy, pts @ toward
    ix = np.clip(np.round((x - lo[0]) / res).astype(int), 0, w - 1)
    iy = np.clip(np.round((y - lo[1]) / res).astype(int), 0, h - 1)
    # Visible if no surface in a small neighbourhood is clearly in front.
    vis = z >= zbuf[iy, ix] - tol
    start = None
    for k in range(n):
      if vis[k] and start is None:
        start = k
      if (not vis[k] or k == n - 1) and start is not None:
        end = k if vis[k] else k - 1
        if end > start:
          pieces.append(((x[start], y[start]), (x[end], y[end])))
        start = None
  return pieces


def chain(pieces, eps=0.05):
  """Join 2D segments sharing endpoints into polylines."""
  key = lambda p: (round(p[0] / eps), round(p[1] / eps))
  ends = {}
  for i, (a, b) in enumerate(pieces):
    ends.setdefault(key(a), []).append((i, 0))
    ends.setdefault(key(b), []).append((i, 1))
  used = [False] * len(pieces)
  lines = []
  for i in range(len(pieces)):
    if used[i]:
      continue
    used[i] = True
    line = [pieces[i][0], pieces[i][1]]
    for direction in (1, 0):
      while True:
        tip = line[-1] if direction else line[0]
        nxt = None
        for j, side in ends.get(key(tip), []):
          if not used[j]:
            nxt = (j, side)
            break
        if nxt is None:
          break
        j, side = nxt
        used[j] = True
        other = pieces[j][1 - side]
        line.append(other) if direction else line.insert(0, other)
    lines.append(np.array(line))
  return lines


def simplify(line, tol):
  """Ramer-Douglas-Peucker."""
  if len(line) < 3:
    return line
  a, b = line[0], line[-1]
  ab = b - a
  norm = np.linalg.norm(ab)
  if norm < 1e-9:
    dist = np.linalg.norm(line - a, axis=1)
  else:
    dist = np.abs(np.cross(ab, line - a)) / norm
  k = int(np.argmax(dist))
  if dist[k] <= tol:
    return np.array([a, b])
  return np.concatenate([simplify(line[:k + 1], tol)[:-1], simplify(line[k:], tol)])


def to_path(lines, min_len):
  out = []
  for line in lines:
    if np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)) < min_len:
      continue
    out.append('M' + 'L'.join(f'{x:.1f} {y:.1f}' for x, y in line))
  return ''.join(out)


def draw(parts, view, args, occluders=()):
  """Paths per part: its visible outline plus its visible creases. Parts
  hide each other; `occluders` hide them too but are not drawn."""
  sx, sy, toward = VIEWS[view]
  names = list(parts)
  tris = [parts[n] for n in names] + list(occluders)
  everything = np.concatenate(tris)
  ids = np.concatenate([np.full(len(t), i) for i, t in enumerate(tris)])
  flat = everything.reshape(-1, 3)
  lo = np.array([(flat @ sx).min(), (flat @ sy).min()]) - 2
  hi = np.array([(flat @ sx).max(), (flat @ sy).max()]) + 2
  zbuf, idbuf = depth_buffer(everything, ids, sx, sy, toward, args.res, lo, hi)
  paths = {}
  for i, name in enumerate(names):
    segs, _ = feature_edges(parts[name], toward, args.crease)
    pieces = visible_segments(segs, zbuf, sx, sy, toward, args.res, lo,
                              tol=args.res * 2.5, step=args.res * 0.7)
    lines = [simplify(l, args.simplify) for l in chain(pieces)]
    lines += [simplify(l, args.simplify) for l in outlines(idbuf, i, args.res, lo)]
    paths[name] = to_path(lines, args.min_len)
    print(f'{view}/{name}: {len(segs)} edges -> {len(lines)} lines, '
          f'{len(paths[name]) / 1e3:.0f} kB', file=sys.stderr)
  return paths


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('meshdir')
  ap.add_argument('out', nargs='?')
  ap.add_argument('--html', help='write into this page between its markers')
  ap.add_argument('--crease', type=float, default=40.0)
  ap.add_argument('--res', type=float, default=0.4, help='depth buffer mm/px')
  ap.add_argument('--simplify', type=float, default=0.25, help='mm')
  ap.add_argument('--min_len', type=float, default=2.0, help='mm')
  args = ap.parse_args()
  parts = mesh_parts(pathlib.Path(args.meshdir).resolve())
  body = np.concatenate([parts['spr_body'], parts['spr_phone']])
  left, right = parts['spr_wheel_left'], parts['spr_wheel_right']
  right_local = right - np.array([HALF_TRACK_MM, 0, 0])
  out = {}
  # Side: the right wheel is in front of the body and turns, so the body is
  # hidden behind the whole disc it sweeps -- hiding it behind the spokes at
  # one angle would bake a still copy of them into the body drawing.
  wheel_r = robot_mujoco.WHEEL_R * 1e3
  sweep = disc(wheel_r, right[..., 0].min(), right[..., 0].max())
  out['side'] = dict(
      draw({'body': body}, 'side', args, occluders=[sweep]),
      **draw({'wheel': right_local}, 'side', args))
  out['front'] = draw({'body': body, 'wheel_l': left, 'wheel_r': right},
                      'front', args)
  out['top'] = draw({'body': body, 'wheel_l': left, 'wheel_r': right},
                    'top', args)
  js = ('// Generated by dreamerv3 tools/robot/dev/blueprint_from_meshes.py from\n'
        '// the rlSim2Real meshes: SVG path data in mm, axle at the origin.\n'
        'const BLUEPRINT = ' + json.dumps(out) + ';\n')
  if args.out:
    pathlib.Path(args.out).write_text(js)
  if args.html:
    page = pathlib.Path(args.html)
    text = page.read_text()
    begin, end = '// BEGIN BLUEPRINT\n', '// END BLUEPRINT'
    i, j = text.index(begin) + len(begin), text.index(end)
    page.write_text(text[:i] + js + text[j:])


if __name__ == '__main__':
  main()
