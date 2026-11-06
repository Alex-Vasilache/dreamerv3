"""Layout of /work/DoyaU/vasilache/work/collapse_figs: topic folders, an archive, an index.

  collapse_figs/
    INDEX.md            newest first per topic: date, figure, table (rebuilt by `tidy`)
    tce_kl/ tce_hash/ tce/ ucb/ som/ worker/ baselines/ wm_collapse/ misc/
    archive/<topic>/    figure groups not touched for ARCHIVE_DAYS days

Report scripts save through `figdir(name)`, which returns (and creates) the topic folder
for a figure name. `tidy` moves anything left at the top level into its topic, archives
groups older than ARCHIVE_DAYS (by the newest file of the group) and rewrites INDEX.md:

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/figs.py
"""
import collections, os, re, shutil, sys, time

ROOT = '/work/DoyaU/vasilache/work/collapse_figs'
ARCHIVE_DAYS = 28
# First matching prefix wins. Topic -> (title, prefixes).
TOPICS = collections.OrderedDict([
    ('tce_kl', ('TCE-KL: log-count bonus inside the entropy term (c sweep, mechanism)', ['tcekl', 'tce_kl'])),
    ('tce_hash', ('TCE start-cell hash: 1x16 vs 4x12, adaptive / smoothed weight', ['hash', 'final4x12', 'wm_hash'])),
    ('tce', ('TCE before the hash: codebook start, weight schemes, anneal / grow, weekend',
             ['tce', 'weekend', 'fromto', 'somtce', 'pp5_failure', 'published_og_tce'])),
    ('ucb', ('UCB goal-selection exploration', ['ucb'])),
    ('som', ('Goal codebook: SOM, US-SOM, masks, goal spread', ['someven', 'even_train', 'goal_spread', 'mask',
                                                               'now_vs_goal_ae'])),
    ('worker', ('Worker: entropy, v3 settings, goal code input', ['wkr'])),
    ('baselines', ('Baselines: OG, OG+v3, DreamerV3, published curves, scale',
                   ['base3', 'ogv', 'now_vs', 'pubmatch_vs', 'make_pubmatch', 'size6m', 'b64'])),
    ('wm_collapse', ('Pin pad collapse study: freezes, resets, WM fixes, deter scale',
                     ['freeze', 'reset', 'deter', 'wmfix', 'plot_', 'compare_arms'])),
])
EXT = re.compile(r'\.(png|pdf|md|txt|json|csv|npz|log|py|pkl)$')


def topic(name):
  for t, (_, prefixes) in TOPICS.items():
    if any(name.startswith(p) for p in prefixes):
      return t
  return 'misc'


def figdir(name):
  d = f'{ROOT}/{topic(name)}'
  os.makedirs(d, exist_ok=True)
  return d


def groups(folder):
  """basename -> [entries] (files and data dirs) in one folder."""
  g = collections.defaultdict(list)
  for e in os.listdir(folder):
    if e in ('INDEX.md', 'archive') or e in TOPICS or e == 'misc':
      continue
    g[re.sub(r'_table$', '', EXT.sub('', e))].append(e)
  return g


def newest(paths):
  return max(os.path.getmtime(p) for p in paths)


def tidy():
  os.makedirs(f'{ROOT}/archive', exist_ok=True)
  moved = archived = 0
  for name, entries in groups(ROOT).items():          # 1. top level -> topic folders
    d = figdir(name)
    for e in entries:
      shutil.move(f'{ROOT}/{e}', f'{d}/{e}'); moved += 1
  for t in list(TOPICS) + ['misc']:                   # 1b. re-home after a topic map change
    src = f'{ROOT}/{t}'
    if os.path.isdir(src):
      for name, entries in groups(src).items():
        if topic(name) != t:
          d = figdir(name)
          for e in entries:
            shutil.move(f'{src}/{e}', f'{d}/{e}'); moved += 1
      if not os.listdir(src):
        os.rmdir(src)
  cutoff = time.time() - ARCHIVE_DAYS * 86400
  for t in list(TOPICS) + ['misc']:                   # 2. old groups -> archive/<topic>
    src = f'{ROOT}/{t}'
    if not os.path.isdir(src):
      continue
    for name, entries in groups(src).items():
      if newest([f'{src}/{e}' for e in entries]) < cutoff:
        dst = f'{ROOT}/archive/{t}'
        os.makedirs(dst, exist_ok=True)
        for e in entries:
          shutil.move(f'{src}/{e}', f'{dst}/{e}'); archived += 1
  index()
  print(f'moved {moved}, archived {archived}; {ROOT}/INDEX.md')


def index():
  lines = ['# collapse_figs index', '',
           f'Newest first within each topic. Groups untouched for {ARCHIVE_DAYS} days live in `archive/<topic>/`. '
           'Rebuild with `tools/collapse_study/figs.py`.', '']
  for t in list(TOPICS) + ['misc']:
    src = f'{ROOT}/{t}'
    if not os.path.isdir(src) or not os.listdir(src):
      continue
    title = TOPICS[t][0] if t in TOPICS else 'Other'
    lines += [f'## {t}/ — {title}', '', '| date | figure | table / data |', '|---|---|---|']
    rows = []
    for name, entries in groups(src).items():
      ts = newest([f'{src}/{e}' for e in entries])
      fig = [e for e in entries if e.endswith('.png')]
      other = [e for e in entries if not e.endswith(('.png', '.pdf'))]
      rows.append((ts, f'| {time.strftime("%m-%d", time.localtime(ts))} | '
                       f'{" ".join(f"[{e}]({t}/{e})" for e in sorted(fig)) or name} | '
                       f'{" ".join(f"[{e}]({t}/{e})" for e in sorted(other))} |'))
    lines += [r for _, r in sorted(rows, reverse=True)] + ['']
  arch = f'{ROOT}/archive'
  if os.path.isdir(arch):
    n = sum(len(groups(f'{arch}/{t}')) for t in os.listdir(arch) if os.path.isdir(f'{arch}/{t}'))
    lines += [f'## archive/ — {n} older figure groups, same topic folders', '']
  open(f'{ROOT}/INDEX.md', 'w').write('\n'.join(lines))


if __name__ == '__main__':
  tidy()
