#!/usr/bin/env python3
"""Square PNGs of the state-sim <-> code-sim mean+/-std curves (director vs
som+lip, hopper_stand), computed earlier straight from results_pairs/*.npz.
Standalone one-off -- not wired into the repo's figure scripts."""
import pathlib
import subprocess

TARGETS = [round(0.1 * k, 2) for k in range(1, 11)]

DATA1 = {  # state-sim bin -> soft-code sim
    'director': dict(
        mean=[0.140, 0.153, 0.152, 0.150, 0.168, 0.174, 0.166, 0.182, 0.298, 0.496],
        std=[0.064, 0.074, 0.074, 0.077, 0.084, 0.086, 0.085, 0.088, 0.130, 0.170],
        n=[3166, 30091, 22761, 9275, 5410, 4424, 4406, 8653, 197101, 237977]),
    'somlip': dict(
        mean=[0.310, 0.295, 0.321, 0.304, 0.322, 0.362, 0.432, 0.528, 0.690, 0.808],
        std=[0.110, 0.103, 0.126, 0.121, 0.104, 0.119, 0.117, 0.132, 0.120, 0.105],
        n=[24744, 17002, 12970, 7805, 9852, 18228, 20594, 23367, 175046, 212890]),
}

DATA2 = {  # hard-code sim bin -> state-sim
    'director': dict(
        mean=[0.798, 0.876, 0.876, 0.933, 0.959, 0.969, 0.975, 0.975, 0.979, 0.984],
        std=[0.268, 0.206, 0.206, 0.123, 0.058, 0.027, 0.017, 0.017, 0.014, 0.010],
        n=[146006, 127968, 127968, 83893, 46114, 20934, 7686, 7686, 1937, 264]),
    'somlip': dict(
        mean=[None, 0.180, 0.154, 0.208, 0.291, 0.542, 0.828, 0.915, 0.952, 0.970],
        std=[None, 0.109, 0.105, 0.154, 0.231, 0.319, 0.211, 0.101, 0.040, 0.020],
        n=[0, 6, 300, 6381, 21467, 64759, 146840, 194195, 111128, 10285]),
}

COLOR = {'director': '#2a78d6', 'somlip': '#eb6834'}
NAME = {'director': 'Director', 'somlip': 'SOM-line + LiP'}

S = 2000  # square canvas
PAD_L, PAD_R, PAD_T, PAD_B = 320, 260, 420, 320
PLOT = S - PAD_L - PAD_R  # also used for height since square interior


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build(data, title, sub, x_label, y_label, out_base, keys=('director', 'somlip')):
  plot_h = S - PAD_T - PAD_B

  def sx(i):
    return PAD_L + (i / (len(TARGETS) - 1)) * PLOT

  def sy(v):
    return PAD_T + (1 - v) * plot_h

  svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" '
         f'viewBox="0 0 {S} {S}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{S}" height="{S}" fill="white"/>']

  svg.append(f'<text x="{PAD_L}" y="90" font-size="66" font-weight="bold" '
             f'fill="#0b0b0b">{esc(title)}</text>')
  svg.append(f'<text x="{PAD_L}" y="155" font-size="40" fill="#52514e">'
             f'{esc(sub)}</text>')

  # legend -- skipped for a single series (its color is unambiguous from the
  # title/subtitle alone, per the one-series-no-legend-box rule)
  if len(keys) > 1:
    ly = 250
    lx = PAD_L
    for key in keys:
      svg.append(f'<line x1="{lx}" y1="{ly - 12}" x2="{lx + 70}" y2="{ly - 12}" '
                 f'stroke="{COLOR[key]}" stroke-width="10"/>')
      svg.append(f'<text x="{lx + 84}" y="{ly}" font-size="40" fill="#52514e">'
                 f'{esc(NAME[key])}</text>')
      lx += 84 + 25 * len(NAME[key]) + 90

  # gridlines + y ticks
  for v in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = sy(v)
    stroke = '#c3c2b7' if v == 0 else '#e1e0d9'
    svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{S - PAD_R}" y2="{y:.1f}" '
               f'stroke="{stroke}" stroke-width="1.5"/>')
    svg.append(f'<text x="{PAD_L - 26}" y="{y + 13:.1f}" font-size="34" '
               f'text-anchor="end" fill="#898781">{v:.2f}</text>')

  # x ticks
  for i, t in enumerate(TARGETS):
    x = sx(i)
    svg.append(f'<text x="{x:.1f}" y="{S - PAD_B + 56:.1f}" font-size="34" '
               f'text-anchor="middle" fill="#898781">{t:.1f}</text>')
  svg.append(f'<text x="{PAD_L + PLOT / 2:.1f}" y="{S - 60}" font-size="40" '
             f'text-anchor="middle" fill="#52514e">{esc(x_label)}</text>')
  svg.append(f'<text x="60" y="{PAD_T + plot_h / 2:.1f}" font-size="40" '
             f'text-anchor="middle" fill="#52514e" '
             f'transform="rotate(-90 60 {PAD_T + plot_h / 2:.1f})">'
             f'{esc(y_label)}</text>')

  for key in keys:
    d = data[key]
    color = COLOR[key]
    segs, cur = [], []
    for i, m in enumerate(d['mean']):
      if m is None:
        if cur:
          segs.append(cur)
        cur = []
      else:
        cur.append(i)
    if cur:
      segs.append(cur)
    for seg in segs:
      top = [(sx(i), sy(min(1.0, d['mean'][i] + d['std'][i]))) for i in seg]
      bot = [(sx(i), sy(max(0.0, d['mean'][i] - d['std'][i]))) for i in reversed(seg)]
      pts = top + bot
      poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
      svg.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.13" '
                 f'stroke="none"/>')
      line_pts = ' '.join(f'{sx(i):.1f},{sy(d["mean"][i]):.1f}' for i in seg)
      svg.append(f'<polyline points="{line_pts}" fill="none" stroke="{color}" '
                 f'stroke-width="7" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
    for i, m in enumerate(d['mean']):
      if m is None:
        continue
      svg.append(f'<circle cx="{sx(i):.1f}" cy="{sy(m):.1f}" r="13" '
                 f'fill="{color}" stroke="white" stroke-width="5"/>')
    last = max(i for i, m in enumerate(d['mean']) if m is not None)
    # nudge the two end labels apart vertically so close-converging curves
    # (both arms end near 1.0 in the codes->state chart) don't overlap
    nudge = -18 if key == 'director' else 38
    svg.append(f'<text x="{sx(last) + 24:.1f}" '
               f'y="{sy(d["mean"][last]) + 14 + nudge:.1f}" '
               f'font-size="46" font-weight="bold" fill="{color}">'
               f'{d["mean"][last]:.2f}</text>')

  svg.append('</svg>')
  base = pathlib.Path(out_base)
  base.with_suffix('.svg').write_text('\n'.join(svg))
  subprocess.run(['rsvg-convert', '-f', 'png', '-w', str(S), '-h', str(S),
                  '-o', str(base.with_suffix('.png')), str(base.with_suffix('.svg'))],
                 check=True)
  print('wrote', base.with_suffix('.png'))


HERE = pathlib.Path(__file__).resolve().parent

build(DATA1, 'States this similar -> code similarity',
      'hopper_stand, pooled over 4 seeds, error bands = +/-1 std',
      'goal-state similarity bin', 'soft-code similarity',
      str(HERE / 'state_to_code_sim'))

build(DATA2, 'Codes this similar -> state similarity',
      'hopper_stand, pooled over 4 seeds, error bands = +/-1 std',
      'hard-code similarity bin', 'goal-state similarity',
      str(HERE / 'code_to_state_sim'))

build(DATA1, 'States this similar -> code similarity (Director)',
      'hopper_stand, pooled over 4 seeds, error band = +/-1 std',
      'goal-state similarity bin', 'soft-code similarity',
      str(HERE / 'state_to_code_sim_director_only'), keys=('director',))

build(DATA2, 'Codes this similar -> state similarity (Director)',
      'hopper_stand, pooled over 4 seeds, error band = +/-1 std',
      'hard-code similarity bin', 'goal-state similarity',
      str(HERE / 'code_to_state_sim_director_only'), keys=('director',))
