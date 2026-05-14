#!/usr/bin/env python3
"""Plot latest v3 vs single-flag vs all-v4 runs from scores.jsonl."""

import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.environ.get('DREAMERV3_WORK', '/work/DoyaU/vasilache/work')
PREFIX = 'dreamerv3_online_vis1m32_1env'

# Use single-quoted family names so font-family="..." attributes stay valid XML.
FONT_STACK = (
    "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
    "Arial, 'Noto Sans', sans-serif")


def classify(name: str) -> Optional[str]:
  if f'{PREFIX}_v4_' in name:
    return 'v4_all'
  if f'{PREFIX}_ab_pmpo_' in name:
    return 'pmpo'
  if f'{PREFIX}_ab_rollout_' in name:
    return 'rollout'
  if f'{PREFIX}_ab_rms_' in name:
    return 'rms'
  if f'{PREFIX}_ab_' in name:
    return None
  if name.startswith(PREFIX + '_'):
    return 'v3'
  return None


def latest_per_category() -> Dict[str, str]:
  dirs = [
      d for d in glob.glob(os.path.join(ROOT, PREFIX + '*'))
      if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'logdir', 'scores.jsonl'))
  ]
  by_cat: Dict[str, List[Tuple[float, str]]] = {}
  for d in dirs:
    cat = classify(os.path.basename(d))
    if cat is None:
      continue
    m = os.path.getmtime(d)
    by_cat.setdefault(cat, []).append((m, d))
  out = {}
  for cat, items in by_cat.items():
    out[cat] = max(items, key=lambda x: x[0])[1]
  return out


def load_scores(path: str) -> Tuple[List[int], List[float]]:
  xs, ys = [], []
  p = os.path.join(path, 'logdir', 'scores.jsonl')
  with open(p, encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      o = json.loads(line)
      xs.append(int(o['step']))
      ys.append(float(o['episode/score']))
  return xs, ys


def svg_escape(s: str) -> str:
  return (
      s.replace('&', '&amp;')
      .replace('<', '&lt;')
      .replace('>', '&gt;')
      .replace('"', '&quot;'))


def format_step_tick(n: int) -> str:
  """Step labels on 500k grid: 0, 500k, 1M, 1.5M, …"""
  if n == 0:
    return '0'
  if n >= 1_000_000:
    m = n / 1_000_000.0
    if abs(m - round(m)) < 1e-6:
      return '%dM' % int(round(m))
    t = '%.1f' % m
    if t.endswith('.0'):
      t = t[:-2]
    return t + 'M'
  if n % 1_000 == 0:
    return '%dk' % (n // 1_000)
  return str(n)


def smooth_moving_average(values: List[float], window: int) -> List[float]:
  """Centered moving average; short series unchanged."""
  n = len(values)
  if n < 3 or window < 3:
    return list(values)
  w = min(window, n)
  if w % 2 == 0:
    w -= 1
  if w < 3:
    return list(values)
  half = w // 2
  out: List[float] = []
  for i in range(n):
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    chunk = values[lo:hi]
    out.append(sum(chunk) / len(chunk))
  return out


def legend_label_lines(label: str, first_line_max: int = 15) -> List[str]:
  """Split long legend labels at a space; avoid orphan last word on line 2."""
  if len(label) <= first_line_max:
    return [label]
  cut = label.rfind(' ', 0, first_line_max + 1)
  if cut < 1:
    cut = first_line_max
  a, b = label[:cut].strip(), label[cut:].strip()
  if not b:
    return [a]
  if len(b) <= 4 and ' ' in a:
    cut2 = a.rfind(' ')
    if cut2 > 0:
      b = (a[cut2:] + ' ' + b).strip()
      a = a[:cut2].strip()
  return [a, b] if b else [a]


def main() -> int:
  cats = latest_per_category()
  need = ('v3', 'pmpo', 'rollout', 'rms', 'v4_all')
  missing = [c for c in need if c not in cats]
  if missing:
    print('Missing categories (no scores.jsonl under matching RUN_DIR):', missing, file=sys.stderr)
    print('Found:', {k: os.path.basename(cats[k]) for k in cats}, file=sys.stderr)
    return 1

  # Okabe–Ito (Wong 2011): five hues spread across warm/cool (avoid sky + blue
  # together; use orange instead of yellow for contrast on a white field).
  order = [
      ('v3', 'v3 baseline', '#0072B2'),  # blue
      ('pmpo', 'PMPO only', '#D55E00'),  # vermillion
      ('rollout', 'single rollout only', '#009E73'),  # bluish green
      ('rms', 'RMS loss norm only', '#CC79A7'),  # reddish purple
      ('v4_all', 'all V4 flags', '#E69F00'),  # orange
  ]

  series = []
  smooth_window = 55
  for key, label, color in order:
    d = cats[key]
    xs, ys = load_scores(d)
    ys_plot = smooth_moving_average(ys, smooth_window)
    series.append((label, color, xs, ys_plot))

  all_x = [x for _, _, xs, _ in series for x in xs]
  all_y = [y for _, _, _, ys in series for y in ys]
  if not all_x:
    print('No data', file=sys.stderr)
    return 1

  # X-axis spans data only (right edge = max step); Y still uses a fixed nice range.
  x_tick_step = 500_000
  x_min = 0
  data_x_max = max(all_x)
  data_y_max = max(all_y)
  data_y_min = min(all_y)
  x_max = int(max(data_x_max, 1))
  y_min, y_max = 0, 1000
  if data_y_max > y_max:
    y_max = int(math.ceil(data_y_max / 250.0) * 250)
  if data_y_min < y_min:
    y_min = math.floor(data_y_min / 250.0) * 250

  x_span = max(x_max - x_min, 1)
  y_span = max(y_max - y_min, 1)

  x_ticks = list(range(x_min, x_max + 1, x_tick_step))
  y_tick_step = 250
  y_ticks = list(range(y_min, y_max + 1, y_tick_step))

  # Fixed canvas W×H; larger type + thicker strokes → more margin, smaller plot area.
  W, H = 3400, 1850
  margin_l, margin_r = 378, 862
  margin_t, margin_b = 322, 358
  plot_w = W - margin_l - margin_r
  plot_h = H - margin_t - margin_b

  fs_title = 130
  fs_axis = 90
  fs_tick = 82
  fs_leg_title = 76
  fs_leg_heading = 92

  line_stroke = 18.0

  def tx(x: float) -> float:
    return margin_l + (x - x_min) / x_span * plot_w

  def ty(y: float) -> float:
    return margin_t + (1.0 - (y - y_min) / y_span) * plot_h

  x0, y0 = margin_l, margin_t
  parts = []

  parts.append(
      '<defs>'
      '<clipPath id="plotclip">'
      f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}"/>'
      '</clipPath>'
      '</defs>')

  # Background panels
  parts.append(
      f'<rect width="{W}" height="{H}" fill="#f8fafc"/>')
  parts.append(
      f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
      f'fill="#ffffff" stroke="#e2e8f0" stroke-width="2" rx="6"/>')

  # Grid at tick positions (skip edges; frame draws border).
  grid_color = '#e2e8f0'
  for yv in y_ticks:
    if yv in (y_min, y_max):
      continue
    py = ty(float(yv))
    parts.append(
        f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x0 + plot_w:.1f}" y2="{py:.1f}" '
        f'stroke="{grid_color}" stroke-width="1.5"/>')
  for xv in x_ticks:
    if xv in (x_min, x_max):
      continue
    px = tx(float(xv))
    parts.append(
        f'<line x1="{px:.1f}" y1="{y0:.1f}" x2="{px:.1f}" y2="{y0 + plot_h:.1f}" '
        f'stroke="{grid_color}" stroke-width="1.5"/>')

  # Series lines (on top of grid, clipped)
  parts.append(f'<g clip-path="url(#plotclip)">')
  for _, color, xs, ys in series:
    if not xs:
      continue
    pts = ' '.join(f'{tx(x):.2f},{ty(y):.2f}' for x, y in zip(xs, ys))
    parts.append(
        f'<polyline fill="none" stroke="{color}" stroke-width="{line_stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{pts}" />')
  parts.append('</g>')

  # Plot border (on top of lines at edges — redraw frame)
  parts.append(
      f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
      f'fill="none" stroke="#64748b" stroke-width="2.5" rx="6"/>')

  tick_major = 50
  axis_text = '#334155'
  y_label_dx = 34
  for yv in y_ticks:
    py = ty(float(yv))
    parts.append(
        f'<line x1="{x0 - tick_major:.1f}" y1="{py:.1f}" x2="{x0:.1f}" y2="{py:.1f}" '
        f'stroke="{axis_text}" stroke-width="2.5"/>')
    parts.append(
        f'<text x="{x0 - tick_major - y_label_dx:.1f}" y="{py + fs_tick * 0.35:.1f}" '
        f'text-anchor="end" font-family="{FONT_STACK}" font-size="{fs_tick}" '
        f'fill="{axis_text}" font-weight="500">{yv:d}</text>')

  x_tick_dy = tick_major + int(fs_tick * 1.15)
  # Extra vertical gap only below tick *numbers* for "Environment steps" (numbers unchanged).
  x_axis_title_extra = 20
  axis_x_title_y = y0 + plot_h + x_tick_dy + int(fs_axis * 1.08) + x_axis_title_extra
  # Rotated "Episode score": farther left of the plot edge (tick numbers unchanged).
  y_axis_label_tx = max(72, margin_l - int(fs_axis * 3.05))
  title_baseline = int(margin_t * 0.46)

  for xv in x_ticks:
    px = tx(float(xv))
    parts.append(
        f'<line x1="{px:.1f}" y1="{y0 + plot_h:.1f}" x2="{px:.1f}" '
        f'y2="{y0 + plot_h + tick_major:.1f}" stroke="{axis_text}" stroke-width="2.5"/>')
    parts.append(
        f'<text x="{px:.1f}" y="{y0 + plot_h + x_tick_dy:.1f}" text-anchor="middle" '
        f'font-family="{FONT_STACK}" font-size="{fs_tick}" fill="{axis_text}" '
        f'font-weight="500">{svg_escape(format_step_tick(xv))}</text>')

  # Axis titles
  parts.append(
      f'<text x="{x0 + plot_w / 2:.1f}" y="{axis_x_title_y:.1f}" text-anchor="middle" '
      f'font-family="{FONT_STACK}" font-size="{fs_axis}" fill="{axis_text}" '
      f'font-weight="600">{svg_escape("Environment steps")}</text>')
  parts.append(
      f'<text transform="translate({y_axis_label_tx},{y0 + plot_h / 2:.1f}) rotate(-90)" '
      f'text-anchor="middle" font-family="{FONT_STACK}" font-size="{fs_axis}" '
      f'fill="{axis_text}" font-weight="600">{svg_escape("Episode score")}</text>')

  # Title block
  title = 'Cartpole swingup'
  parts.append(
      f'<text x="{W / 2:.1f}" y="{title_baseline:d}" text-anchor="middle" font-family="{FONT_STACK}" '
      f'font-size="{fs_title}" font-weight="700" fill="#0f172a">'
      f'{svg_escape(title)}</text>')

  # Legend column (right panel): wide box + word-wrapped lines for long labels.
  leg_pad = 22
  leg_x = margin_l + plot_w + leg_pad
  leg_inner_w = margin_r - 2 * leg_pad
  leg_top = margin_t + 12
  parts.append(
      f'<rect x="{leg_x:.1f}" y="{leg_top:.1f}" width="{leg_inner_w:.1f}" height="{plot_h - 24:.1f}" '
      f'fill="#ffffff" stroke="#e2e8f0" stroke-width="2" rx="8"/>')
  parts.append(
      f'<text x="{leg_x + leg_pad:.1f}" y="{leg_top + int(fs_leg_heading * 0.75):d}" font-family="{FONT_STACK}" '
      f'font-size="{fs_leg_heading}" font-weight="700" fill="#0f172a">'
      f'{svg_escape("Runs")}</text>')

  leg_line_gap = int(fs_leg_title * 1.12)
  row_gap = int(fs_leg_title * 0.5)
  sw, sh = 44, 44
  y_cursor = leg_top + int(fs_leg_heading * 1.35)
  for label, color, _, _ in series:
    lines = legend_label_lines(label, first_line_max=19)
    row_h = sh + len(lines) * leg_line_gap + row_gap
    parts.append(
        f'<rect x="{leg_x + leg_pad:.1f}" y="{y_cursor:.1f}" width="{sw}" height="{sh}" '
        f'rx="4" fill="{color}"/>')
    tx0 = leg_x + leg_pad + sw + 18
    for li, ln in enumerate(lines):
      ty_ln = y_cursor + sh - 6 + li * leg_line_gap
      parts.append(
          f'<text x="{tx0:.1f}" y="{ty_ln:.1f}" font-family="{FONT_STACK}" '
          f'font-size="{fs_leg_title}" font-weight="600" fill="#0f172a">'
          f'{svg_escape(ln)}</text>')
    y_cursor += row_h

  svg = (
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}">'
      f'<style>text {{ text-rendering: geometricPrecision; }}</style>'
      + ''.join(parts) +
      '</svg>\n')

  out = os.path.join(ROOT, 'dreamerv3_latest_ab_score_compare.svg')
  with open(out, 'w', encoding='utf-8') as f:
    f.write(svg)
  print(out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
