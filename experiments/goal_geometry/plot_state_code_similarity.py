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

S = 1000  # viewBox units == output pixels (1000x1000, per the request)
DPI = 300  # embedded density tag (rsvg-convert default is 90) -- print-quality
PAD_L, PAD_R, PAD_T, PAD_B = 190, 190, 260, 190
PLOT = S - PAD_L - PAD_R  # also used for height since square interior

# High-contrast ink: pure/near-black text and darker gridlines instead of the
# lighter secondary/muted grays used at the previous size, so the chart still
# reads at a glance when scaled down or viewed on a dim screen.
INK = '#000000'
INK_2 = '#1a1a1a'
GRID = '#b0afa8'
BASELINE = '#5c5b56'


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

  # Fit the title to the fixed 1000px canvas -- the "(Director)" suffix
  # otherwise runs the bold title past the right edge at a flat font size.
  title_budget = S - PAD_L - 20
  title_size = min(42, title_budget / (0.62 * len(title)))
  svg.append(f'<text x="{PAD_L}" y="58" font-size="{title_size:.1f}" '
             f'font-weight="bold" fill="{INK}">{esc(title)}</text>')
  svg.append(f'<text x="{PAD_L}" y="98" font-size="24" fill="{INK_2}">'
             f'{esc(sub)}</text>')

  # legend -- skipped for a single series (its color is unambiguous from the
  # title/subtitle alone, per the one-series-no-legend-box rule)
  if len(keys) > 1:
    ly = 152
    lx = PAD_L
    for key in keys:
      svg.append(f'<line x1="{lx}" y1="{ly - 8}" x2="{lx + 42}" y2="{ly - 8}" '
                 f'stroke="{COLOR[key]}" stroke-width="7"/>')
      svg.append(f'<text x="{lx + 52}" y="{ly}" font-size="24" '
                 f'font-weight="600" fill="{INK}">{esc(NAME[key])}</text>')
      lx += 52 + 15 * len(NAME[key]) + 54

  # gridlines + y ticks
  for v in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = sy(v)
    stroke = BASELINE if v == 0 else GRID
    svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{S - PAD_R}" y2="{y:.1f}" '
               f'stroke="{stroke}" stroke-width="{1.5 if v else 2.5}"/>')
    svg.append(f'<text x="{PAD_L - 16}" y="{y + 8:.1f}" font-size="21" '
               f'font-weight="600" text-anchor="end" fill="{INK}">{v:.2f}</text>')

  # x ticks
  for i, t in enumerate(TARGETS):
    x = sx(i)
    svg.append(f'<text x="{x:.1f}" y="{S - PAD_B + 34:.1f}" font-size="21" '
               f'font-weight="600" text-anchor="middle" fill="{INK}">{t:.1f}</text>')
  svg.append(f'<text x="{PAD_L + PLOT / 2:.1f}" y="{S - 36}" font-size="24" '
             f'font-weight="600" text-anchor="middle" fill="{INK}">'
             f'{esc(x_label)}</text>')
  svg.append(f'<text x="34" y="{PAD_T + plot_h / 2:.1f}" font-size="24" '
             f'font-weight="600" text-anchor="middle" fill="{INK}" '
             f'transform="rotate(-90 34 {PAD_T + plot_h / 2:.1f})">'
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
      svg.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.20" '
                 f'stroke="none"/>')
      line_pts = ' '.join(f'{sx(i):.1f},{sy(d["mean"][i]):.1f}' for i in seg)
      svg.append(f'<polyline points="{line_pts}" fill="none" stroke="{color}" '
                 f'stroke-width="5.5" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
    for i, m in enumerate(d['mean']):
      if m is None:
        continue
      svg.append(f'<circle cx="{sx(i):.1f}" cy="{sy(m):.1f}" r="9" '
                 f'fill="{color}" stroke="white" stroke-width="3.5"/>')
    last = max(i for i, m in enumerate(d['mean']) if m is not None)
    # nudge the two end labels apart vertically so close-converging curves
    # (both arms end near 1.0 in the codes->state chart) don't overlap
    nudge = -12 if key == 'director' else 24
    svg.append(f'<text x="{sx(last) + 16:.1f}" '
               f'y="{sy(d["mean"][last]) + 9 + nudge:.1f}" '
               f'font-size="30" font-weight="bold" fill="{color}">'
               f'{d["mean"][last]:.2f}</text>')

  svg.append('</svg>')
  base = pathlib.Path(out_base)
  base.with_suffix('.svg').write_text('\n'.join(svg))
  subprocess.run(['rsvg-convert', '-f', 'png', '-w', str(S), '-h', str(S),
                  '--dpi-x', str(DPI), '--dpi-y', str(DPI),
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
